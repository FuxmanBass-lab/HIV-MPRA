#!/usr/bin/env python3

"""
Plot variation between four replicates (patients) as a function of activity.

Usage:
  python 10_patients_activity_variation.py \
      -i ../data/activity/OL53_T_primaryT_activity_withreplicates.tsv \
      -o ../results/patients_variation/activity_vs_variance --base-id-reps 3

Notes:
  - The script expects a column called 'ID' and four per-patient
    activity columns (see PATIENT_COLS below).
  - Tiles with IDs starting with:
        HIV-1:<something>:6:
        HIV-1:<something>:13:
    are excluded from the basic SD/CV plots, but are included in the
    mean–variance modeling and in the Ftrend plots.
"""

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ======================================================================
# Configuration – EDIT THIS to match your column names
# ======================================================================
# Prefix for the columns that contain activity for each patient/replicate.
# With the new activity output, per-replicate activity columns are named like:
#   activity_Plasmid_r1 ... activity_Plasmid_rN
#   activity_primaryT_r1 ... activity_primaryT_r4
# We only want the primary T (patient) activity replicates here.
PATIENT_ACTIVITY_PREFIX = "activity_primaryT_"


def exclude_hiv_tiles_6_and_13(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude rows whose ID matches:
        HIV-1:<something>:6:
        HIV-1:<something>:13:
    """
    if "ID" not in df.columns:
        raise ValueError("Expected an 'ID' column in the input file.")

    # Regex: start with HIV-1:, then anything up to next colon, then 6 or 13, then colon
    pattern = r"^HIV-1:[^:]+:(6|13):"

    mask_exclude = df["ID"].astype(str).str.match(pattern)
    kept = df.loc[~mask_exclude].copy()

    return kept


# ----------------------------------------------------------------------
# Helper: deduplicate tiles by base ID (remove isolate/mutant suffix)
# ----------------------------------------------------------------------
def deduplicate_tiles_by_base_id(df: pd.DataFrame, n_reps: int | None = None, random_state: int = 1337) -> pd.DataFrame:
    """Return a deduplicated view of df where tiles that differ only by the
    final colon-separated suffix in ID are collapsed to up to n_reps random
    representative rows per base ID (reproducibly).

    Example collapsed IDs:
      - HIV-1:REJO:287:-_MT190941.1  -> base ID 'HIV-1:REJO:287'
      - HIV-1:REJO:287:-_MT190978.1  -> base ID 'HIV-1:REJO:287'
      - Papilloma_Virus:Type_16:139:+_Modified_LC456194.1 -> 'Papilloma_Virus:Type_16:139'
      - Papilloma_Virus:Type_16:139:+_Modified_OP971067.1 -> 'Papilloma_Virus:Type_16:139'

    If n_reps is None, all rows are returned (no deduplication). If n_reps
    is a positive integer, up to n_reps random representatives per base ID
    are kept.

    The returned DataFrame has up to n_reps rows per base ID, but the original df is
    left unchanged so that all tiles can still be written to the metrics TSV.
    """
    if "ID" not in df.columns or n_reps is None:
        return df

    if n_reps < 1:
        # Nothing sensible to do; fall back to returning all rows
        return df

    # Compute base IDs by removing everything after the final ':'
    base_id = df["ID"].astype(str).str.replace(r":[^:]+$", "", regex=True)

    # Shuffle once for reproducible random choice within each base_id group
    shuffled = df.sample(frac=1.0, random_state=random_state)
    base_id_shuffled = base_id.loc[shuffled.index]

    # Take up to n_reps rows per base_id in the shuffled order
    keep_idx = shuffled.groupby(base_id_shuffled, sort=False).head(n_reps).index

    return df.loc[keep_idx].copy()


# ----------------------------------------------------------------------
# Helper: stratified sampling by activity for mean-variance trend fitting
# ----------------------------------------------------------------------
def stratified_sample_by_activity(
    df: pd.DataFrame,
    activity_col: str = "activity",
    n_bins: int | None = None,
    n_per_bin: int | None = None,
    random_state: int = 1337,
) -> pd.Index:
    """Return a subset of df index selected via stratified sampling over activity.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame (already filtered to rows eligible for fitting).
    activity_col : str
        Column name containing the activity values (assumed finite for rows in df).
    n_bins : int or None
        Number of quantile-based bins over the activity range. If None, no
        stratification is performed and all indices are returned.
    n_per_bin : int or None
        Maximum number of rows to sample from each bin. If None, no
        stratification is performed and all indices are returned.

    Returns
    -------
    pandas.Index
        Index of df corresponding to selected rows. If n_bins or n_per_bin is
        None, returns df.index (no downsampling).
    """
    if n_bins is None or n_per_bin is None:
        return df.index

    if activity_col not in df.columns:
        return df.index

    # Work on a copy of the relevant columns to avoid SettingWithCopy warnings
    activity = pd.to_numeric(df[activity_col], errors="coerce")
    # Drop rows with non-finite activity from stratification; they will be
    # excluded from the fit upstream.
    valid = np.isfinite(activity.to_numpy())
    df_valid = df.loc[activity.index[valid]].copy()
    if df_valid.empty:
        return df.index

    # Quantile-based bins over activity
    try:
        bins = pd.qcut(df_valid[activity_col], q=n_bins, duplicates="drop")
    except ValueError:
        # If qcut fails (e.g. not enough unique values), fall back to no
        # stratification.
        return df.index

    selected_idx = []
    rng = np.random.default_rng(random_state)

    for level, group in df_valid.groupby(bins):
        if group.empty:
            continue
        n_take = min(n_per_bin, len(group))
        # Use numpy RNG for reproducibility independent of pandas version
        chosen = rng.choice(group.index.to_numpy(), size=n_take, replace=False)
        selected_idx.extend(chosen.tolist())

    if not selected_idx:
        return df.index

    return pd.Index(selected_idx)


def compute_activity_and_variation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with per-replicate activity columns, compute:
      - activity: log2 of the mean primary T activity across patient replicates
      - sd: standard deviation of log2 activity across patient replicates
      - cv: coefficient of variation in log-activity space (sd / log2-mean activity)
      - mean_activity_linear: mean activity across patient replicates on the original (linear) scale

    Patient activity columns are detected by the PATIENT_ACTIVITY_PREFIX.
    """
    # Detect patient activity columns dynamically (e.g. activity_primaryT_r1..r4)
    patient_cols = [c for c in df.columns if c.startswith(PATIENT_ACTIVITY_PREFIX)]

    if len(patient_cols) == 0:
        raise ValueError(
            f"No patient activity columns found with prefix '{PATIENT_ACTIVITY_PREFIX}'. "
            "Check that your activity file has columns like 'activity_primaryT_r1'..."
        )
    if len(patient_cols) < 2:
        raise ValueError(
            f"Expected at least 2 patient activity columns with prefix '{PATIENT_ACTIVITY_PREFIX}', "
            f"found {len(patient_cols)}: {patient_cols}"
        )

    df = df.copy()

    # Mean activity across patient replicates on the original (linear) activity scale
    mean_linear = df[patient_cols].mean(axis=1)
    df["mean_activity_linear"] = mean_linear

    # Log2-mean activity for the x-axis; values <= 0 are set to NaN
    mean_linear_safe = mean_linear.where(mean_linear > 0, np.nan)
    df["activity"] = np.log2(mean_linear_safe)

    # Log2-transform per-replicate activities, guarding against non-positive values
    log_rep = df[patient_cols].where(df[patient_cols] > 0)
    log_rep = np.log2(log_rep)

    # Standard deviation of log2 activity across patient replicates
    df["sd"] = log_rep.std(axis=1, skipna=True)

    # Coefficient of variation in log-activity space – use np.nan to avoid division by zero
    df["cv"] = df["sd"] / df["activity"].replace(0, np.nan)

    return df


# ----------------------------------------------------------------------
# Helper: regress sd on predictors to compute noise-adjusted variation
# ----------------------------------------------------------------------
def compute_noise_adjusted_variation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regress 'sd' on available predictors and compute expected sd, residuals, and F-statistic.
    Predictors: intercept, 'activity', 'log2FoldChange' (if present), log10('ctrl_mean') (if present and >0).
    Adds columns:
      - sd_expected: fitted value from regression
      - sd_resid: residual (observed sd minus expected)
      - sd_Fstat: (sd ** 2) / sigma2 (global residual variance)
    Returns modified DataFrame.
    """
    import numpy as np
    import pandas as pd
    df = df.copy()
    # Build predictors
    predictors = []
    X_cols = []
    # Intercept
    X_cols.append("intercept")
    predictors.append(np.ones(len(df)))
    # activity
    if "activity" in df.columns:
        X_cols.append("activity")
        predictors.append(df["activity"].to_numpy())
    # log2FoldChange
    if "log2FoldChange" in df.columns:
        X_cols.append("log2FoldChange")
        predictors.append(df["log2FoldChange"].to_numpy())
    # log10(ctrl_mean)
    if "ctrl_mean" in df.columns:
        ctrl_mean = df["ctrl_mean"]
        ctrl_mean_log10 = np.full(len(df), np.nan)
        mask = ctrl_mean > 0
        ctrl_mean_log10[mask] = np.log10(ctrl_mean[mask])
        X_cols.append("log10_ctrl_mean")
        predictors.append(ctrl_mean_log10)
    # Stack predictors column-wise
    X = np.column_stack(predictors)
    # Response
    y = df["sd"].to_numpy()
    # Mask: rows where y and all predictors are finite
    mask = np.isfinite(y)
    for arr in predictors:
        mask = mask & np.isfinite(arr)
    # Fit linear model using least squares
    if mask.sum() >= len(X_cols):
        X_fit = X[mask]
        y_fit = y[mask]
        # lstsq returns: coef, residuals, rank, s
        coef, _, _, _ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
        # Fitted values
        sd_hat = np.full(len(df), np.nan)
        sd_hat[mask] = X_fit @ coef
        # Residuals
        resid = np.full(len(df), np.nan)
        resid[mask] = y_fit - sd_hat[mask]
        # Global residual variance
        sigma2 = np.nanmean((y_fit - sd_hat[mask]) ** 2)
        # sd_expected
        df["sd_expected"] = np.full(len(df), np.nan)
        df.loc[mask, "sd_expected"] = sd_hat[mask]
        # sd_resid
        df["sd_resid"] = np.full(len(df), np.nan)
        df.loc[mask, "sd_resid"] = y_fit - sd_hat[mask]
        # sd_Fstat
        df["sd_Fstat"] = np.nan
        df.loc[mask, "sd_Fstat"] = (y_fit ** 2) / sigma2 if sigma2 > 0 else np.nan
    else:
        # Not enough rows to fit model; fill with NaN
        df["sd_expected"] = np.nan
        df["sd_resid"] = np.nan
        df["sd_Fstat"] = np.nan
    return df


def compute_meanvariance_trend(
    df: pd.DataFrame,
    base_id_reps: int | None = None,
    activity_bins: int | None = None,
    activity_per_bin: int | None = None,
) -> pd.DataFrame:
    """\
    Mean–variance trend (DESeq2-style, but simple):

      log(var_obs) ~ beta0 + beta1 * activity + [beta2 * log10(ctrl_mean)]

    where:
      - var_obs = variance of log2 primary T activity across patients
                  (here computed as sd**2, where sd is the SD of log2 activity)
      - activity = log2 mean primary T activity across patient replicates
                   (column `activity` produced by `compute_activity_and_variation`)
      - ctrl_mean = control DNA/plasmid mean on the original (linear) scale,
                    modeled as log10(ctrl_mean) if present and > 0.

    Adds to the DataFrame:
      - var_trend : expected variance of log2 activity from the fitted trend
      - sd_Ftrend : F-like statistic = var_obs / var_trend
    """
    df = df.copy()

    # We require sd (SD of log2 activity) and activity (log2 mean activity)
    if "sd" not in df.columns or "activity" not in df.columns:
        df["var_trend"] = np.nan
        df["sd_Ftrend"] = np.nan
        return df

    # Observed variance of log2 activity across patients
    sd = df["sd"].to_numpy()
    var_obs = sd ** 2

    # Predictors
    activity = df["activity"].to_numpy()  # log2 mean activity

    # ------------------------------------------------------------------
    # Build a mask of rows to use for fitting the global mean–variance
    # trend:
    #   - positive, finite variance
    #   - finite activity
    #   - finite, positive ctrl_mean (if available)
    #   - deduplicated by base tile ID (collapse isolates/mutants)
    #   - (all tiles, including HIV-1 tiles 6 and 13, are included)
    # ------------------------------------------------------------------
    valid_var = (var_obs > 0) & np.isfinite(var_obs)
    valid_act = np.isfinite(activity)

    if "ctrl_mean" in df.columns:
        ctrl = df["ctrl_mean"].to_numpy()
        valid_ctrl = (ctrl > 0) & np.isfinite(ctrl)
    else:
        ctrl = None
        valid_ctrl = np.ones(len(df), dtype=bool)

    # Deduplicate by base ID (ID without the final colon-separated suffix)
    # if base_id_reps is provided; otherwise use all tiles.
    if "ID" in df.columns and base_id_reps is not None:
        df_dedup = deduplicate_tiles_by_base_id(df, n_reps=base_id_reps)
        dedup_idx = df_dedup.index
        is_dedup_rep = df.index.isin(dedup_idx)
    else:
        is_dedup_rep = np.ones(len(df), dtype=bool)

    # Final mask for fitting (before optional activity stratification).
    # All tiles, including HIV-1 tiles 6 and 13, are eligible here as long
    # as they pass the variance/activity/ctrl_mean filters and (optionally)
    # the base-ID deduplication.
    base_mask = is_dedup_rep
    mask_fit = base_mask & valid_var & valid_act & valid_ctrl

    # Optionally perform stratified sampling over activity so that the
    # global trend is fit on a more balanced set of tiles across the
    # activity range.
    if mask_fit.any() and (activity_bins is not None and activity_per_bin is not None):
        df_fit_candidates = df.loc[mask_fit].copy()
        idx_strat = stratified_sample_by_activity(
            df_fit_candidates,
            activity_col="activity",
            n_bins=activity_bins,
            n_per_bin=activity_per_bin,
        )
        # Reset mask_fit to only keep stratified subset
        new_mask = np.zeros_like(mask_fit, dtype=bool)
        new_mask[df.index.get_indexer(idx_strat)] = True
        mask_fit = mask_fit & new_mask

    # Need at least a few points to fit
    if mask_fit.sum() < 3:
        df["var_trend"] = np.nan
        df["sd_Ftrend"] = np.nan
        return df

    # Design matrix for the fit
    X_list = [
        np.ones(mask_fit.sum()),          # intercept
        activity[mask_fit],               # activity is already log2(mean activity)
    ]

    if ctrl is not None:
        # Use log10(ctrl_mean) as the plasmid predictor
        log10_ctrl = np.log10(ctrl[mask_fit])
        X_list.append(log10_ctrl)

    X = np.column_stack(X_list)

    # Response: log of the observed variance (natural log)
    var_fit = var_obs[mask_fit]
    log_var_fit = np.log(var_fit)

    # Fit: log(var_obs) = X * beta
    beta, _, _, _ = np.linalg.lstsq(X, log_var_fit, rcond=None)

    # ------------------------------------------------------------------
    # Predict expected variance for *all* rows with valid predictors,
    # including HIV-1 tiles 6/13 and duplicated isolates. Only the fit
    # was restricted; prediction is global so all tiles get var_trend.
    # ------------------------------------------------------------------
    var_hat = np.full(len(df), np.nan)

    if ctrl is not None:
        valid_pred = valid_act & ((ctrl > 0) & np.isfinite(ctrl))
        if valid_pred.sum() > 0:
            X_pred = np.column_stack([
                np.ones(valid_pred.sum()),
                activity[valid_pred],
                np.log10(ctrl[valid_pred]),
            ])
        else:
            X_pred = None
    else:
        valid_pred = valid_act
        if valid_pred.sum() > 0:
            X_pred = np.column_stack([
                np.ones(valid_pred.sum()),
                activity[valid_pred],
            ])
        else:
            X_pred = None

    if X_pred is not None:
        log_var_hat = X_pred @ beta
        var_hat_pred = np.exp(log_var_hat)
        var_hat[valid_pred] = var_hat_pred

    # Guard against extreme tiny / huge expectations by clipping
    positive_vals = var_hat[(var_hat > 0) & np.isfinite(var_hat)]
    if positive_vals.size == 0:
        df["var_trend"] = np.nan
        df["sd_Ftrend"] = np.nan
        return df

    low, high = np.nanpercentile(positive_vals, [1, 99])
    var_hat = np.clip(var_hat, low, high)

    df["var_trend"] = var_hat

    # F-like statistic: observed variance / expected variance
    with np.errstate(divide="ignore", invalid="ignore"):
        f_stat = var_obs / var_hat
        f_stat[~np.isfinite(f_stat)] = np.nan
    df["sd_Ftrend"] = f_stat

    return df

def make_scatter(x, y, x_label, y_label, title, out_path: Path, y_clip=None, ma_window=None, log_x=False):
    """Simple scatter plot helper.
    Drops NA values before plotting to avoid NAType issues.
    """
    # Drop NA / non-numeric values
    mask = pd.notna(x) & pd.notna(y)
    x_clean = pd.to_numeric(x[mask], errors="coerce")
    y_clean = pd.to_numeric(y[mask], errors="coerce")
    finite_mask = np.isfinite(x_clean) & np.isfinite(y_clean)
    x_clean = x_clean[finite_mask].to_numpy()
    y_clean = y_clean[finite_mask].to_numpy()

    # Sort by x (native scale)
    order = np.argsort(x_clean)
    x_plot = x_clean[order]
    y_sorted = y_clean[order]

    # Optionally clip y-values to reduce extreme outliers
    if y_clip is not None:
        if isinstance(y_clip, (list, tuple)) and len(y_clip) == 2:
            y_sorted = np.clip(y_sorted, y_clip[0], y_clip[1])
        elif isinstance(y_clip, (int, float)):
            # symmetric clipping: [-y_clip, y_clip]
            y_sorted = np.clip(y_sorted, -abs(y_clip), abs(y_clip))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x_plot, y_sorted, s=5, alpha=0.6, linewidth=0)

    # Moving-average: window = ±ma_window on the native x scale
    if ma_window is not None:
        xs = x_plot
        ys = y_sorted
        y_ma = np.full_like(ys, np.nan, dtype=float)

        for i, x0 in enumerate(xs):
            lo = x0 - ma_window
            hi = x0 + ma_window
            mask_win = (xs >= lo) & (xs <= hi)
            if np.any(mask_win):
                y_ma[i] = np.nanmean(ys[mask_win])

        ax.plot(x_plot, y_ma, linewidth=2.0, alpha=1.0, color="#d62728", zorder=10)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(-2,10)

    # Apply log scaling to the x-axis if requested (does not affect windowing)
    from matplotlib.ticker import ScalarFormatter
    if log_x == "log10":
        ax.set_xscale("log", base=10)
    elif log_x == "log2" or log_x is True:
        ax.set_xscale("log", base=2)
    # Format ticks as plain numbers when log scale is used
    if log_x:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_ftrend_with_highlights(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
    ma_window: float | None = None,
) -> None:
    """Scatter plot of sd_Ftrend with HIV-1 tiles 6, 9, and 13 highlighted.

    This uses the full DataFrame (no exclusion of tiles 6/13, no base-ID
    deduplication) so that those tiles appear in the plot, even though they
    are excluded from the global trend fit.
    """
    if x_col not in df.columns or y_col not in df.columns:
        return

    # Drop NA / non-numeric values
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())

    x = x[mask]
    y = y[mask]
    df_valid = df.loc[mask].copy()

    # Sort by x
    order = np.argsort(x.to_numpy())
    x_plot = x.to_numpy()[order]
    y_plot = y.to_numpy()[order]
    df_valid = df_valid.iloc[order]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Base scatter for all tiles
    ax.scatter(x_plot, y_plot, s=5, alpha=0.4, linewidth=0, color="#bbbbbb")

    # Identify HIV-1 tiles 6, 9, and 13
    if "ID" in df_valid.columns:
        ids = df_valid["ID"].astype(str)
        mask_6 = ids.str.match(r"^HIV-1:[^:]+:6:")
        mask_9 = ids.str.match(r"^HIV-1:[^:]+:9:")
        mask_13 = ids.str.match(r"^HIV-1:[^:]+:13:")

        # Plot and label each group if present
        for m, color, label in [
            (mask_6, "#1f77b4", "HIV-1 tile 6"),
            (mask_9, "#2ca02c", "HIV-1 tile 9"),
            (mask_13, "#d62728", "HIV-1 tile 13"),
        ]:
            if m.any():
                ax.scatter(
                    x_plot[m.to_numpy()],
                    y_plot[m.to_numpy()],
                    s=5,              # same size as base scatter
                    alpha=0.9,
                    linewidth=0.0,    # no bold outline
                    edgecolor="none",
                    label=label,
                    color=color,
                )
                # Add simple text labels next to the first point in each group
                idx_first = np.where(m.to_numpy())[0][0]
                ax.text(
                    x_plot[idx_first],
                    y_plot[idx_first],
                    label.replace("HIV-1 ", ""),
                    fontsize=8,
                    ha="left",
                    va="bottom",
                )

    # Moving-average trend line over all points
    if ma_window is not None:
        xs = x_plot
        ys = y_plot
        y_ma = np.full_like(ys, np.nan, dtype=float)
        for i, x0 in enumerate(xs):
            lo = x0 - ma_window
            hi = x0 + ma_window
            mask_win = (xs >= lo) & (xs <= hi)
            if np.any(mask_win):
                y_ma[i] = np.nanmean(ys[mask_win])
        ax.plot(xs, y_ma, linewidth=2.0, alpha=1.0, color="#ff7f0e", zorder=10)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(0, 10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot variation between four replicates as a function of activity."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input results file (tab-delimited by default).",
    )
    parser.add_argument(
        "-o",
        "--out-prefix",
        default="patients_variation",
        help="Output file prefix (default: patients_variation).",
    )
    parser.add_argument(
        "--sep",
        default="\t",
        help="Column separator for the input file (default: '\\t').",
    )
    parser.add_argument(
        "--base-id-reps",
        type=int,
        default=None,
        help=(
            "Number of random representatives per base tile ID to use for "
            "fitting the global mean–variance trend and for plotting. "
            "If not set, use all tiles (no deduplication)."
        ),
    )
    parser.add_argument(
        "--activity-bins",
        type=int,
        default=None,
        help=(
            "Number of quantile-based bins over log2 activity to use for "
            "stratified sampling when fitting the mean–variance trend. "
            "If not set, all eligible tiles are used."
        ),
    )
    parser.add_argument(
        "--activity-per-bin",
        type=int,
        default=None,
        help=(
            "Maximum number of tiles to sample from each activity bin when "
            "fitting the mean–variance trend. If not set, all eligible "
            "tiles are used."
        ),
    )

    args = parser.parse_args()
    base_id_reps = args.base_id_reps
    activity_bins = args.activity_bins
    activity_per_bin = args.activity_per_bin

    in_path = Path(args.input)
    out_prefix = Path(args.out_prefix)

    # Ensure output directory exists
    out_dir = out_prefix.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    df = pd.read_csv(in_path, sep=args.sep)
    # Exclude tiles with low control mean (< 50) if ctrl_mean is present
    if "ctrl_mean" in df.columns:
        df = df[(df["ctrl_mean"].isna()) | (df["ctrl_mean"] >= 50)].copy()

    # ------------------------------------------------------------------
    # Compute activity and variation across the four patients
    #   (for ALL tiles; we apply HIV-1:6/13 exclusion only for plotting)
    # ------------------------------------------------------------------
    df = compute_activity_and_variation(df)
    df = compute_noise_adjusted_variation(df)
    df = compute_meanvariance_trend(
        df,
        base_id_reps=base_id_reps,
        activity_bins=activity_bins,
        activity_per_bin=activity_per_bin,
    )

    # Make a plotting-only subset that excludes HIV-1 tiles 6 and 13,
    # then deduplicate tiles that differ only by isolate/mutation.
    df_plot = exclude_hiv_tiles_6_and_13(df)
    df_plot = deduplicate_tiles_by_base_id(df_plot, n_reps=base_id_reps)

    # Sort plotting DataFrame by activity for nicer-looking plots
    df_plot = df_plot.sort_values("activity")

    # ------------------------------------------------------------------
    # Plots
    #   1) SD vs activity
    #   2) CV vs activity
    # ------------------------------------------------------------------
    stem = out_prefix.stem
    sd_plot = out_dir / f"{stem}_sd_vs_activity.png"
    cv_plot = out_dir / f"{stem}_cv_vs_activity.png"

    make_scatter(
        x=df_plot["activity"],
        y=df_plot["sd"],
        x_label="log2 mean CD4 activity (across patient replicates)",
        y_label="SD of log2 CD4 activity",
        title="Variation between patients vs log2 mean activity",
        out_path=sd_plot,
        ma_window=0.5,  # moving average over ±0.5 log2 window (1 log2 total width)
    )
    # SD vs ctrl_mean
    if "ctrl_mean" in df_plot.columns:
        sd_ctrl_plot = out_dir / f"{stem}_sd_vs_ctrl_mean.png"
        ctrl_mean_log10 = np.log10(df_plot["ctrl_mean"].where(df_plot["ctrl_mean"] > 0))
        make_scatter(
            x=ctrl_mean_log10,
            y=df_plot["sd"],
            x_label="log10 control mean",
            y_label="SD of log2 CD4 activity",
            title="Variation between patients vs log10 control mean",
            out_path=sd_ctrl_plot,
            ma_window=0.5,
        )

    make_scatter(
        x=df_plot["activity"],
        y=df_plot["cv"],
        x_label="log2 mean CD4 activity (across patient replicates)",
        y_label="Coefficient of variation (SD / log2 mean activity)",
        title="Relative variation between patients vs log2 mean activity",
        out_path=cv_plot,
    )
    # CV vs ctrl_mean
    if "ctrl_mean" in df_plot.columns:
        cv_ctrl_plot = out_dir / f"{stem}_cv_vs_ctrl_mean.png"
        ctrl_mean_log10 = np.log10(df_plot["ctrl_mean"].where(df_plot["ctrl_mean"] > 0))
        make_scatter(
            x=ctrl_mean_log10,
            y=df_plot["cv"],
            x_label="log10 control mean",
            y_label="Coefficient of variation",
            title="Relative variation between patients vs log10 control mean",
            out_path=cv_ctrl_plot,
            ma_window=0.5,
        )

    cv_plot_clipped = out_dir / f"{stem}_cv_vs_activity_clipped.png"
    make_scatter(
        x=df_plot["activity"],
        y=df_plot["cv"],
        x_label="log2 mean CD4 activity (across patient replicates)",
        y_label="Coefficient of variation (clipped)",
        title="Relative variation (clipped) between patients vs log2 mean activity",
        out_path=cv_plot_clipped,
        y_clip=(-2, 2)
    )

    sd_fstat_plot = out_dir / f"{stem}_sdFstat_vs_log2FC.png"
    make_scatter(
        x=df_plot["log2FoldChange"],
        y=df_plot["sd_Fstat"],
        x_label="log2FoldChange (DESeq2)",
        y_label="Noise-adjusted variability (sd_Fstat)",
        title="Noise-adjusted variability vs log2FoldChange",
        out_path=sd_fstat_plot,
        ma_window=0.5,
    )

    # sd_Fstat vs ctrl_mean
    if "ctrl_mean" in df_plot.columns:
        sd_fstat_ctrl_plot = out_dir / f"{stem}_sdFstat_vs_ctrl_mean.png"
        ctrl_mean_log10 = np.log10(df_plot["ctrl_mean"].where(df_plot["ctrl_mean"] > 0))
        make_scatter(
            x=ctrl_mean_log10,
            y=df_plot["sd_Fstat"],
            x_label="log10 control mean",
            y_label="Noise-adjusted variability (sd_Fstat)",
            title="Noise-adjusted variability vs log10 control mean",
            out_path=sd_fstat_ctrl_plot,
            ma_window=0.5,
        )

    # Additional plots for mean-variance trend adjusted metric sd_Ftrend.
    # Here we use the full df (no HIV-1 6/13 exclusion, no deduplication)
    # so that those tiles can be included and highlighted, while still
    # having been excluded from the trend fitting itself.
    sd_ftrend_plot = out_dir / f"{stem}_sdFtrend_vs_log2FC.png"
    plot_ftrend_with_highlights(
        df=df,
        x_col="log2FoldChange",
        y_col="sd_Ftrend",
        x_label="log2FoldChange (DESeq2)",
        y_label="Noise-adjusted variability (mean–variance F, sd_Ftrend)",
        title="Noise-adjusted variability (trend) vs log2FoldChange",
        out_path=sd_ftrend_plot,
        ma_window=0.5,
    )

    if "ctrl_mean" in df.columns:
        sd_ftrend_ctrl_plot = out_dir / f"{stem}_sdFtrend_vs_ctrl_mean.png"
        ctrl_mean_log10 = np.log10(df["ctrl_mean"].where(df["ctrl_mean"] > 0))
        # Reuse the same helper, but pass a temporary DataFrame with the
        # x-column set to log10(ctrl_mean)
        df_ftrend_ctrl = df.copy()
        df_ftrend_ctrl["_log10_ctrl_mean"] = ctrl_mean_log10
        plot_ftrend_with_highlights(
            df=df_ftrend_ctrl,
            x_col="_log10_ctrl_mean",
            y_col="sd_Ftrend",
            x_label="log10 control mean",
            y_label="Noise-adjusted variability (mean–variance F, sd_Ftrend)",
            title="Noise-adjusted variability (trend) vs log10 control mean",
            out_path=sd_ftrend_ctrl_plot,
            ma_window=0.5,
        )

    # Additional plots: log10 of sd_Ftrend to visualize the F-like metric on a log scale.
    # We restrict to sd_Ftrend > 0 for the log10 transform.
    if "sd_Ftrend" in df.columns:
        df_logF = df.copy()
        df_logF["log10_sd_Ftrend"] = np.nan
        mask_pos = df_logF["sd_Ftrend"] > 0
        df_logF.loc[mask_pos, "log10_sd_Ftrend"] = np.log10(df_logF.loc[mask_pos, "sd_Ftrend"])

        sd_logF_plot = out_dir / f"{stem}_log10_sdFtrend_vs_log2FC.png"
        plot_ftrend_with_highlights(
            df=df_logF,
            x_col="log2FoldChange",
            y_col="log10_sd_Ftrend",
            x_label="log2FoldChange (DESeq2)",
            y_label="log10 noise-adjusted variability (mean–variance F, sd_Ftrend)",
            title="log10 noise-adjusted variability (trend) vs log2FoldChange",
            out_path=sd_logF_plot,
            ma_window=0.5,
        )

        if "ctrl_mean" in df.columns:
            sd_logF_ctrl_plot = out_dir / f"{stem}_log10_sdFtrend_vs_ctrl_mean.png"
            df_logF_ctrl = df_logF.copy()
            ctrl_mean_log10 = np.log10(df_logF_ctrl["ctrl_mean"].where(df_logF_ctrl["ctrl_mean"] > 0))
            df_logF_ctrl["_log10_ctrl_mean"] = ctrl_mean_log10
            plot_ftrend_with_highlights(
                df=df_logF_ctrl,
                x_col="_log10_ctrl_mean",
                y_col="log10_sd_Ftrend",
                x_label="log10 control mean",
                y_label="log10 noise-adjusted variability (mean–variance F, sd_Ftrend)",
                title="log10 noise-adjusted variability (trend) vs log10 control mean",
                out_path=sd_logF_ctrl_plot,
                ma_window=0.5,
            )

    # ------------------------------------------------------------------
    # Export per-tile variation metrics
    #   Columns:
    #     - ID
    #     - log2FoldChange (from DESeq2 output)
    #     - sd  (SD of log2 CD4 activity across patients)
    #     - cv  (CV in log-activity space)
    #     - mean_activity_linear
    #     - var_trend
    #     - sd_Ftrend
    # ------------------------------------------------------------------
    metrics_path = out_dir / f"{stem}_activity_variation_metrics.tsv"
    cols_to_keep = []
    for c in ["ID", "log2FoldChange", "sd", "cv", "sd_expected", "sd_resid", "sd_Fstat", "mean_activity_linear", "var_trend", "sd_Ftrend"]:
        if c in df.columns:
            cols_to_keep.append(c)
    df[cols_to_keep].to_csv(metrics_path, sep="\t", index=False)
    print(f"Wrote variation metrics table: {metrics_path}")


if __name__ == "__main__":
    main()
