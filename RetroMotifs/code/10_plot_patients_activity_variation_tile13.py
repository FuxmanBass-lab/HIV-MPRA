#!/usr/bin/env python3
"""
Plot activity distributions (log2FoldChange) per motif grammar for Tile 13.

Inputs expected (produced by 06_tile13_motif_grammar.py):
  - ../results/motif_counts/tile13_site_presence_fixedbins.tsv
  - ../results/motif_counts/tile13_site_combination_counts_fixedbins.tsv
And activity metrics:
  - ../data/activity/OL53_run_Jurkat_berkay_activity.tsv
  - ../data/activity/comparison_INFg_vs_Ctrl.tsv

Behavior
--------
- Keeps only motif grammars (signatures) with ≥ 10 isolates.
- Y-axis lists grammars; for each y-row we draw a 4-slot rectangle representing
  sites (half‑open reference bins):
    1) slot2 bin [40,55): IRF_x2_site2 **or** E2F_site1 (mutually exclusive)
    2) slot3 bin [60,80): IRF_x3_site1 **or** IRF_x2_site3 (mutually exclusive)
    3) E2F_site2
    4) SP/KLF_site1
  Absent sites are light gray (#eeeeee) with a thin edge. If both families are present
  in a mutually exclusive slot (rare), priority is E2F over IRF_x2 for [40,55) and IRF_x3
  over IRF_x2 for [60,80).
- X-axis is activity (log2FoldChange) as horizontal violins for isolates within
  each grammar.
- Shows two violin panels: baseline (Jurkat) and INFg.
- Saves PDF and PNG.

Usage
-----
python 07_tile13_plot_grammar_activity.py \
  --presence ../results/motif_grammar/tile13_site_presence_fixedbins.tsv \
  --counts   ../results/motif_grammar/tile13_site_combination_counts_fixedbins.tsv \
  --baseline ../data/activity/OL53_run_Jurkat_berkay_activity.tsv \
  --ifng     ../data/activity/comparison_INFg_vs_Ctrl.tsv \
  --min-n 3 \
  --order-by ifng \
  --outfig ../results/figures/tile13_grammar_activity_3iso

python 10_plot_patients_activity_variation_tile13.py \
  --presence ../results/motif_grammar/tile13_site_presence_fixedbins.tsv \
  --counts   ../results/motif_grammar/tile13_site_combination_counts_fixedbins.tsv \
  --baseline ../data/activity/OL53_T_primaryT_activity.tsv \
  --variation ../results/patients_variation/activity_vs_variance_activity_variation_metrics.tsv \
  --min-n 3 \
  --order-by baseline \
  --baseline-and-variation \
  --outfig ../results/figures/variation/tile13_grammar_activity_CD4_3iso


Paired SP/KLF on/off comparison (F-statistic variation):
python 10_plot_patients_activity_variation_tile13.py \
  --presence ../results/motif_grammar/tile13_site_presence_fixedbins.tsv \
  --counts   ../results/motif_grammar/tile13_site_combination_counts_fixedbins.tsv \
  --baseline ../data/activity/OL53_T_primaryT_activity.tsv \
  --variation ../results/patients_variation/activity_vs_variance_activity_variation_metrics.tsv \
  --variation-metric fstat \
  --baseline-and-variation \
  --pair-sp-only \
  --pair-test brunnermunzel \
  --min-n 3 \
  --order-by baseline \
  --outfig ../results/figures/variation/tile13_grammar_activity_CD4_pairs

# Welch version:
python 10_plot_patients_activity_variation_tile13.py \
  --presence ../results/motif_grammar/tile13_site_presence_fixedbins.tsv \
  --counts   ../results/motif_grammar/tile13_site_combination_counts_fixedbins.tsv \
  --baseline ../data/activity/OL53_T_primaryT_activity.tsv \
  --variation ../results/patients_variation/activity_vs_variance_activity_variation_metrics.tsv \
  --variation-metric fstat \
  --baseline-and-variation \
  --pair-sp-only \
  --pair-test welch \
  --min-n 3 \
  --order-by baseline \
  --outfig ../results/figures/variation/tile13_grammar_activity_CD4_pairs_welch

Global SP pooled comparison (2 violins; Brunner–Munzel):
python 10_plot_patients_activity_variation_tile13.py \
 --presence ../results/motif_grammar/tile13_site_presence_fixedbins.tsv \
  --counts   ../results/motif_grammar/tile13_site_combination_counts_fixedbins.tsv \
  --baseline ../data/activity/OL53_T_primaryT_activity.tsv \
  --variation ../results/patients_variation/activity_vs_variance_activity_variation_metrics.tsv \
  --min-n 3 \
  --baseline-and-variation \
  --variation-metric fstat \
  --outfig ../results/figures/variation/tile13_grammar_activity_CD4_pairs
# This also writes: *_SPglobal_<metric>.pdf/png
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None
from matplotlib.patches import Rectangle, Wedge


# Colors
COL_ABS  = "#eeeeee"
EDGE     = "#555555"

# Family color mapping for Tile 13
FAMILY_COLORS = {
    "IRF_x2": "#7FB6E0",  # softer sky blue (muted)
    "IRF_x3": "#005F73",  # intense blue‑teal (more saturated/darker)
    "E2F":     "#8FBF88",  # muted green
    "SP/KLF":  "#9370db",  # muted purple (kept)
}

CLADE_ORDER = ["A", "B", "C", "D", "AE", "F", "G", "H", "O", "Other"]
CLADE_COLORS = {
    "A":  "#4E79A7",   # blue
    "B":  "#E15759",   # red
    "C":  "#59A14F",   # green
    "D":  "#B07AA1",   # purple
    "AE": "#EDC948",   # yellow
    "F":  "#76B7B2",   # teal
    "G":  "#F28E2B",   # orange
    "H":  "#9C755F",   # brown
    "O":  "#FF9DA7",   # pink
    "Other": "#BAB0AC", # gray
}
SUBTYPE_MAP = {
    "A1": "A", "A6": "A", "A4": "A", "A2": "A", "A": "A",
    "B": "B", "C": "C", "D": "D",
    "F1": "F", "F2": "F", "F1F2": "F",
    "G": "G", "O": "O", "U": "Other", "N": "Other",
    "H": "H", "L": "Other", "J": "Other",
    "AE": "AE", "CRF01_AE": "AE"
}

# Four-slot grammar layout for Tile 13 (excluding IRF_x2_site1)
#  slot1: IRF_x2_site2 OR E2F_site1 (mutually exclusive bin [40,55))
#  slot2: IRF_x3_site1  OR IRF_x2_site3 (mutually exclusive bin [60,80))
#  slot3: E2F_site2
#  slot4: SP/KLF_site1

FOUR_SLOTS = (
    ("IRF_x2_site2", "E2F_site1"),   # slot 1 (either)
    ("IRF_x3_site1", "IRF_x2_site3"),# slot 2 (either)
    ("E2F_site2",),                    # slot 3
    ("SP/KLF_site1",),                 # slot 4
)

# Priority when both families appear in a mutually exclusive slot (rare):
#  - slot1 ([40,55)): prefer E2F over IRF_x2
#  - slot2 ([60,80)): prefer IRF_x3 over IRF_x2
SLOT_PRIORITY = {
    1: ("E2F", "IRF_x2"),
    2: ("IRF_x3", "IRF_x2"),
    3: ("E2F",),
    4: ("SP/KLF",),
}

def pooled_variation_by_sp(
    pres: pd.DataFrame,
    var_df: pd.DataFrame,
    metric: str = "sd_ftrend",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vals_sp_minus, vals_sp_plus) pooled across all configurations.

    Pools per-isolate variation values based on whether the isolate's grammar has SP/KLF_site1.
    """
    metric = metric.lower().strip()
    vcol = "sd_ftrend" if metric in {"sd", "sd_ftrend", "ftrend"} else "fstat"

    # isolate -> has_sp (from presence table)
    iso_sp = (
        pres[["isolate", "SP/KLF_site1"]]
        .dropna()
        .drop_duplicates(subset=["isolate"], keep="last")
        .rename(columns={"SP/KLF_site1": "has_sp"})
    )
    iso_sp["has_sp"] = iso_sp["has_sp"].astype(bool)

    v = var_df[["isolate", vcol]].dropna().copy()
    v[vcol] = v[vcol].astype(float)

    merged = v.merge(iso_sp, on="isolate", how="inner")
    vals_minus = merged.loc[~merged["has_sp"], vcol].to_numpy(dtype=float)
    vals_plus = merged.loc[merged["has_sp"], vcol].to_numpy(dtype=float)

    return vals_minus, vals_plus


def extract_isolate_from_id(s: str) -> str:
    # ID looks like: HIV-1:REJO:6:+_Modified_MT929400.1 or similar
    # isolate is the suffix after the last underscore
    return str(s).split("_")[-1]


def load_presence_counts(presence_path: Path, counts_path: Path, min_n: int):
    pres = pd.read_csv(presence_path, sep="\t")
    cnts = pd.read_csv(counts_path, sep="\t")
    cnts = cnts[cnts["n_isolates"] >= min_n].copy()
    keep = set(cnts["signature"].tolist())
    pres = pres[pres["signature"].isin(keep)].copy()
    # order signatures by count desc
    order = (
        cnts.sort_values(["n_isolates", "signature"], ascending=[False, True])["signature"].tolist()
    )
    return pres, cnts, order



def load_activity(activity_path: Path):
    act = pd.read_csv(activity_path, sep="\t")
    # Filter to tile 13 only using the pattern :13:
    act = act[act["ID"].astype(str).str.contains(r":13:")].copy()
    act["isolate"] = act["ID"].map(extract_isolate_from_id)
    return act[["ID", "isolate", "log2FoldChange"]]


# --- Variation loading helper (updated for F-stat support) ---
def load_variation(variation_path: Path, metric: str = "sd_ftrend") -> pd.DataFrame:
    """Load variation metrics for Tile 13 and return ID, isolate, and a chosen metric.

    Parameters
    ----------
    variation_path:
        TSV with per-tile variation metrics.
    metric:
        Which metric to use. Supported values:
          - "sd_ftrend"  : trend-adjusted SD-style metric (legacy)
          - "fstat"      : F-statistic style metric

    Notes
    -----
    The function accepts several possible column spellings and normalizes them to:
      - 'sd_ftrend' when metric == "sd_ftrend"
      - 'fstat'     when metric == "fstat"
    """
    var = pd.read_csv(variation_path, sep="\t")
    # Filter to tile 13 only using the pattern :13:
    var = var[var["ID"].astype(str).str.contains(r":13:")].copy()
    var["isolate"] = var["ID"].map(extract_isolate_from_id)

    metric = metric.lower().strip()
    if metric in {"sd", "sd_ftrend", "ftrend", "sd_ftrend_activity"}:
        candidate_cols = [
            "sd_ftrend",
            "sd_ftrend_ctrl",
            "sd_Ftrend",
            "sd_Ftrend_ctrl",
            "sd_ftrend_activity",
        ]
        found = [c for c in candidate_cols if c in var.columns]
        if not found:
            raise ValueError(
                "variation file must contain one of these sd_ftrend columns: "
                f"{candidate_cols}. Found: {list(var.columns)}"
            )
        chosen = found[0]
        var = var.rename(columns={chosen: "sd_ftrend"})
        return var[["ID", "isolate", "sd_ftrend"]]

    if metric in {"fstat", "f", "f_stat", "f_statistic"}:
        candidate_cols = [
            "fstat",
            "Fstat",
            "F_stat",
            "F_statistic",
            "f_stat",
            "f_statistic",
            "sd_Fstat",  # legacy name sometimes used
        ]
        found = [c for c in candidate_cols if c in var.columns]
        if not found:
            raise ValueError(
                "variation file must contain one of these F-stat columns: "
                f"{candidate_cols}. Found: {list(var.columns)}"
            )
        chosen = found[0]
        var = var.rename(columns={chosen: "fstat"})
        return var[["ID", "isolate", "fstat"]]

    raise ValueError(f"Unknown variation metric: {metric}. Use 'sd_ftrend' or 'fstat'.")

def collect_activity_by_signature(pres: pd.DataFrame, order: list[str], act: pd.DataFrame):
    # pres has one row per isolate with boolean columns and signature
    data = []
    for sig in order:
        iso = pres.loc[pres["signature"] == sig, "isolate"].astype(str).tolist()
        vals = act.loc[act["isolate"].isin(iso), "log2FoldChange"].astype(float).dropna().values
        data.append((sig, iso, vals))
    return data


# --- Variation by signature helper (with metric support) ---
def collect_variation_by_signature(
    pres: pd.DataFrame,
    order: list[str],
    var: pd.DataFrame,
    metric: str = "sd_ftrend",
):
    """Collect variation values for each grammar signature.

    Expects var to contain either 'sd_ftrend' or 'fstat' depending on metric.
    """
    metric = metric.lower().strip()
    col = "sd_ftrend" if metric in {"sd", "sd_ftrend", "ftrend"} else "fstat"

    data = []
    for sig in order:
        iso = pres.loc[pres["signature"] == sig, "isolate"].astype(str).tolist()
        vals = var.loc[var["isolate"].isin(iso), col].astype(float).dropna().values
        data.append((sig, iso, vals))
    return data


def _slot_family(pres_row: pd.Series, slot_idx_1based: int) -> str | None:
    """Return the chosen family for a given slot according to SLOT_PRIORITY.

    For mutually-exclusive slots, we resolve ties using SLOT_PRIORITY.
    Returns one of: 'IRF_x2', 'IRF_x3', 'E2F', 'SP/KLF', or None.
    """
    site_names = FOUR_SLOTS[slot_idx_1based - 1]
    present_fams = []
    for sname in site_names:
        if bool(pres_row.get(sname, False)):
            fam = sname.split("_site")[0]
            present_fams.append(fam)
    if not present_fams:
        return None
    pri = SLOT_PRIORITY.get(slot_idx_1based, ())
    for fam in pri:
        if fam in present_fams:
            return fam
    return present_fams[0]


def _arch_key_no_sp(pres_row: pd.Series) -> tuple:
    """Architecture key excluding SP/KLF slot (slot 4).

    We key by the *displayed* 4-slot grammar representation for slots 1-3:
      - slot1 family (IRF_x2 vs E2F vs None)
      - slot2 family (IRF_x3 vs IRF_x2 vs None)
      - slot3 presence (E2F vs None)

    This avoids incorrect pairing when raw booleans differ but the resolved
    slot representation is the same (or vice versa).
    """
    s1 = _slot_family(pres_row, 1)
    s2 = _slot_family(pres_row, 2)
    s3 = _slot_family(pres_row, 3)  # E2F or None
    return (s1, s2, s3)


def build_sp_pairs(pres: pd.DataFrame, signatures: list[str]) -> list[tuple[str, str]]:
    """Return list of (sig_no_sp, sig_with_sp) pairs.

    Only includes architectures that have exactly one signature with SP/KLF_site1==False
    and one with SP/KLF_site1==True.
    """
    # one representative row per signature
    rep = pres[pres["signature"].isin(signatures)].groupby("signature", as_index=False).head(1)

    by_key: dict[tuple, dict[bool, str]] = {}
    for _i, r in rep.iterrows():
        sig = str(r["signature"])
        key = _arch_key_no_sp(r)
        has_sp = bool(r.get("SP/KLF_site1", False))
        by_key.setdefault(key, {})[has_sp] = sig

    pairs = []
    for _key, d in by_key.items():
        if (False in d) and (True in d):
            pairs.append((d[False], d[True]))
    return pairs


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size (x vs y). Range [-1, 1]."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        return np.nan
    # O(n*m) but small groups here.
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (x.size * y.size))


def compare_two_distributions(
    x: np.ndarray,
    y: np.ndarray,
    test: str = "brunnermunzel",
    log_transform_for_t: bool = True,
) -> tuple[float, float]:
    """Return (statistic, pvalue) for a two-sided comparison.

    Supported tests:
      - brunnermunzel (default)
      - mannwhitney
      - welch  (Welch's t-test; unequal variances). For positive-valued metrics,
               we recommend running Welch on log10(metric) to reduce skew.
    """
    test = test.lower().strip()
    if stats is None:
        raise RuntimeError("scipy is required for statistical tests (install scipy)")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if test in {"brunnermunzel", "bm", "brunner-munzel"}:
        res = stats.brunnermunzel(x, y, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)

    if test in {"welch", "ttest", "t", "welch-t", "welch_t"}:
        xx = x
        yy = y
        if log_transform_for_t:
            # Only defined for positive values; caller should already filter >0.
            xx = np.log10(xx)
            yy = np.log10(yy)
        res = stats.ttest_ind(xx, yy, equal_var=False, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)

    # Mann-Whitney U (two-sided)
    res = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def load_isolate_to_clade_map(path: Path) -> dict:
    # Try to read with header first; if expected columns aren't present, fallback to no header
    try:
        df_try = pd.read_csv(path, sep='\t')
    except Exception:
        df_try = pd.DataFrame()

    if 'tile_id' in df_try.columns and 'Clade' in df_try.columns:
        df = df_try
    else:
        df = pd.read_csv(path, sep='\t', header=None, names=['tile_id', 'Clade'])
    # extract isolate suffix
    def iso_from_tile(tile_id: str) -> str:
        return str(tile_id).split('_')[-1]
    df['isolate'] = df['tile_id'].map(iso_from_tile)
    # normalize/merge subtypes
    def to_super(c: str) -> str:
        c = str(c).strip()
        # direct map first
        if c in SUBTYPE_MAP:
            return SUBTYPE_MAP[c]
        # if already a top-level clade we keep it
        if c in CLADE_ORDER:
            return c
        # CRF01_AE / AE-like strings
        if 'AE' in c.upper():
            return 'AE'
        # Anything else (CRFs, composites, unknowns) -> Other
        return 'Other'
    df['SuperClade'] = df['Clade'].map(to_super)
    # keep last occurrence per isolate
    mp = df.set_index('isolate')['SuperClade'].to_dict()
    return mp


def draw_grammar_rect(ax, row_idx: int, pres_row: pd.Series, x0: float = 0.0, width: float = 1.0, height: float = 0.8):
    """Draw the 4-slot rectangle for one grammar at y=row_idx on ax.
    x0/width are in axis data units (we'll use a separate axes with unit scale).
    """
    seg_w = width / 4.0
    y = row_idx - height/2

    def add_rect(x, w, color):
        ax.add_patch(Rectangle((x, y), w, height, facecolor=color, edgecolor=EDGE, linewidth=0.5))

    for k in range(4):
        x = x0 + k * seg_w
        site_names = FOUR_SLOTS[k]
        # Determine presence and family to color by
        present_fams = []
        for sname in site_names:
            if bool(pres_row.get(sname, False)):
                fam = sname.split("_site")[0]
                present_fams.append(fam)
        if not present_fams:
            add_rect(x, seg_w, COL_ABS)
        else:
            # choose by SLOT_PRIORITY
            pri = SLOT_PRIORITY.get(k+1, ())
            chosen_fam = None
            for fam in pri:
                if fam in present_fams:
                    chosen_fam = fam
                    break
            if chosen_fam is None:
                # fallback to first present family
                chosen_fam = present_fams[0]
            color = FAMILY_COLORS.get(chosen_fam, "#666666")
            add_rect(x, seg_w, color)

    # set limits for this small canvas
    ax.set_xlim(x0, x0 + width)
    ax.set_ylim(0.5, len(ax.get_yticks()) + 0.5)


def draw_pie(ax, center_x: float, center_y: float, frac_by_clade: dict, radius: float = 0.35):
    total = sum(frac_by_clade.values())
    if total <= 0:
        return
    start_angle = 90.0
    for clade in CLADE_ORDER:
        val = frac_by_clade.get(clade, 0)
        if val <= 0:
            continue
        theta = 360.0 * (val / total)
        wedge = Wedge((center_x, center_y), radius, start_angle, start_angle + theta,
                      facecolor=CLADE_COLORS.get(clade, '#cccccc'), edgecolor='#333333', linewidth=0.6)
        ax.add_patch(wedge)
        start_angle += theta


def plot_grammars_with_activity(
    presence_path: Path,
    counts_path: Path,
    base_path: Path,
    ifng_path: Path,
    clades_path: Path,
    variation_path: Path | None,
    min_n: int,
    outprefix: Path,
    order_by: str = "baseline",
    baseline_only: bool = False,
    baseline_and_variation: bool = False,
    variation_metric: str = "sd_ftrend",
    pair_sp_only: bool = False,
    pair_test: str = "brunnermunzel",
):
    pres, cnts, order = load_presence_counts(presence_path, counts_path, min_n)
    act_base = load_activity(base_path)
    # Only load IFNg when we actually plan to plot it
    if not baseline_only and not baseline_and_variation:
        act_ifng = load_activity(ifng_path)
    else:
        act_ifng = None

    # Load variation metrics if provided
    if variation_path is not None:
        var_df = load_variation(variation_path, metric=variation_metric)
    else:
        var_df = None

    iso2clade = load_isolate_to_clade_map(clades_path)

    # Collect activity rows using initial order
    rows_base_o = collect_activity_by_signature(pres, order, act_base)
    if act_ifng is not None:
        rows_ifng_o = collect_activity_by_signature(pres, order, act_ifng)
    else:
        rows_ifng_o = []

    # Collect variation rows using initial order (if available)
    if var_df is not None:
        rows_var_o = collect_variation_by_signature(pres, order, var_df, metric=variation_metric)
    else:
        rows_var_o = []

    # Choose which condition determines row ordering (by median, desc)
    order_src = order_by.lower().strip()
    if baseline_only:
        order_src = "baseline"
    if order_src not in {"baseline", "ifng"}:
        order_src = "baseline"

    # Optional: restrict to matched SP/KLF on/off pairs (architecture identical except SP/KLF_site1)
    # Optional: group matched SP/KLF on/off pairs (architecture identical except SP/KLF_site1)
    if pair_sp_only:
        pairs = build_sp_pairs(pres, order)
        in_pair = {s for a, b in pairs for s in (a, b)}

        # Baseline median per signature for scoring
        base_by_sig = {sig: vals for sig, _iso, vals in rows_base_o}
        med_base = {
            sig: (float(np.median(base_by_sig.get(sig, np.array([]))))
                if base_by_sig.get(sig, np.array([])).size > 0 else np.nan)
            for sig in order
        }

        # Each pair is one sortable entity (score = max of medians)
        # Each unpaired signature is its own entity (score = its median)
        entities: list[tuple[str, float, tuple[str, ...]]] = []

        for sig_no, sig_yes in pairs:
            m1 = med_base.get(sig_no, np.nan)
            m2 = med_base.get(sig_yes, np.nan)
            # Pair score: max baseline median across the two (ignoring NaNs)
            if np.isnan(m1) and np.isnan(m2):
                score = np.nan
            elif np.isnan(m1):
                score = float(m2)
            elif np.isnan(m2):
                score = float(m1)
            else:
                score = float(max(m1, m2))
            entities.append((f"PAIR::{sig_no}::{sig_yes}", score, (sig_no, sig_yes)))

        for sig in order:
            if sig in in_pair:
                continue
            entities.append((f"SIG::{sig}", med_base.get(sig, np.nan), (sig,)))

        # Sort entities by score desc; NaNs at bottom
        entities.sort(key=lambda x: (np.isnan(x[1]), -x[1] if not np.isnan(x[1]) else 0))

        # Expand entities into final signature order; keep pair members adjacent (SP+ then SP-); avoid duplicates
        ordered_sigs = []
        seen = set()
        for _eid, _score, members in entities:
            if len(members) == 2:
                # members are (sig_no_sp, sig_with_sp)
                sig_no, sig_yes = members
                # We want SP+ first, then SP-
                for s in (sig_yes, sig_no):
                    if s not in seen:
                        ordered_sigs.append(s)
                        seen.add(s)
            else:
                s = members[0]
                if s not in seen:
                    ordered_sigs.append(s)
                    seen.add(s)


    else:
        rows_src = {"baseline": rows_base_o, "ifng": rows_ifng_o}[order_src]
        med_list = []
        for sig, _iso, vals in rows_src:
            median_val = float(np.median(vals)) if vals.size > 0 else np.nan
            med_list.append((sig, median_val))
        ordered_sigs = [
            sig
            for sig, _ in sorted(
                med_list,
                key=lambda x: (np.isnan(x[1]), -x[1] if not np.isnan(x[1]) else 0),
            )
        ]

    # Build rows for all conditions in the chosen order
    rows_base = collect_activity_by_signature(pres, ordered_sigs, act_base)
    if not baseline_only and not baseline_and_variation and act_ifng is not None:
        rows_ifng = collect_activity_by_signature(pres, ordered_sigs, act_ifng)
    else:
        rows_ifng = []

    if var_df is not None:
        rows_var = collect_variation_by_signature(pres, ordered_sigs, var_df, metric=variation_metric)
    else:
        rows_var = []

    # Use baseline rows to determine number of grammars
    n = len(rows_base)
    if n == 0:
        print("No grammar groups meet the minimum isolate threshold.")
        return

    # Build figure layout depending on mode
    if baseline_and_variation:
        # pies, grammars, baseline activity, variation (sd_ftrend)
        fig = plt.figure(figsize=(16, max(3.5, 0.6 * n)))
        gs = fig.add_gridspec(
            nrows=1,
            ncols=4,
            width_ratios=[0.95, 0.80, 3.0, 3.0],
        )
        axP = fig.add_subplot(gs[0, 0])  # pies axis (leftmost)
        axG = fig.add_subplot(gs[0, 1])  # grammar glyphs + labels
        axB = fig.add_subplot(gs[0, 2])  # baseline violins
        axV = fig.add_subplot(gs[0, 3])  # variation violins (per grammar)
        axT = None
        axC = None
        gs.update(wspace=0.08)
    elif baseline_only:
        fig = plt.figure(figsize=(11, max(3.5, 0.6 * n)))
        gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[0.95, 0.80, 3.0])
        axP = fig.add_subplot(gs[0, 0])  # pies axis (leftmost)
        axG = fig.add_subplot(gs[0, 1])  # grammar glyphs + labels
        axB = fig.add_subplot(gs[0, 2])
        axV = None
        axT = None
        axC = None
    else:
        # pies, grammars, baseline, IFNg
        fig = plt.figure(figsize=(14, max(3.5, 0.6 * n)))
        gs = fig.add_gridspec(nrows=1, ncols=4, width_ratios=[0.95, 0.80, 3.0, 3.0])
        axP = fig.add_subplot(gs[0, 0])  # pies axis (leftmost)
        axG = fig.add_subplot(gs[0, 1])  # grammar glyphs + labels
        axB = fig.add_subplot(gs[0, 2])
        axT = fig.add_subplot(gs[0, 3])
        axV = None
        axC = None
        gs.update(wspace=0.08)

    # Prepare legend handles (figure-level legend placed in a clear area)
    legend_handles = [
        Rectangle((0,0), 1, 1, facecolor=FAMILY_COLORS["IRF_x2"], edgecolor=EDGE, linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=FAMILY_COLORS["IRF_x3"], edgecolor=EDGE, linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=FAMILY_COLORS["E2F"],    edgecolor=EDGE, linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=FAMILY_COLORS["SP/KLF"], edgecolor=EDGE, linewidth=0.5),
    ]

    # Prepare y positions (top-to-bottom)
    y_positions = np.arange(1, n + 1)
    axP.set_ylim(0.5, n + 0.5)
    axG.set_ylim(0.5, n + 0.5)
    axB.set_ylim(0.5, n + 0.5)
    if axT is not None:
        axT.set_ylim(0.5, n + 0.5)
    if 'axV' in locals() and axV is not None:
        axV.set_ylim(0.5, n + 0.5)
    # Match row ordering with other panels so pies move with row reordering
    axP.invert_yaxis()

    # pies axis aesthetics
    axP.set_xlim(0.0, 1.0)
    axP.set_xticks([])
    axP.set_yticks([])
    for spine in ["top", "right", "bottom", "left"]:
        axP.spines[spine].set_visible(False)
    # Ensure pies are circular
    axP.set_aspect('equal', adjustable='box')

    # Left: grammar rectangles and labels
    axG.set_xticks([])
    axG.set_yticks(y_positions)
    labels = []
    for i, (sig, iso_list, vals) in enumerate(rows_base, start=1):
        # representative row from presence for drawing the sites
        pres_row = pres[pres["signature"] == sig].iloc[0]
        draw_grammar_rect(axG, i, pres_row, x0=0.0, width=4.0, height=0.8)
        count = len(iso_list)
        sp_flag = "SP+" if bool(pres_row.get("SP/KLF_site1", False)) else "SP-"
        if pair_sp_only:
            labels.append(f"{sp_flag}  n={count}")
        else:
            labels.append(f"n={count}")
        # pie: fraction of clades among isolates in this grammar (drawn on axP)
        clade_counts = {c: 0 for c in CLADE_ORDER}
        for iso in iso_list:
            cl = iso2clade.get(str(iso), 'Other')
            if cl not in CLADE_ORDER:
                cl = 'Other'
            clade_counts[cl] = clade_counts.get(cl, 0) + 1
        draw_pie(axP, 0.5, i, clade_counts, radius=0.45)
    axG.set_yticklabels(labels, fontsize=11)
    axG.set_title("Motif grammars (Tile 13 — 4 slots)", fontsize=12)
    axG.set_xlim(0, 4.0)
    axG.invert_yaxis()  # top grammar at top
    axG.tick_params(axis='y', pad=1)

    # Clean aesthetics for left panel
    for spine in ["top", "right", "bottom", "left"]:
        axG.spines[spine].set_visible(False)

    def draw_simple_violin(ax, i, vals):
        if vals.size == 0:
            return
        parts = ax.violinplot([vals], positions=[i], vert=False, showextrema=False, widths=0.6)
        for b in parts['bodies']:
            b.set_facecolor('#bbbbbb')
            b.set_alpha(0.25)
            b.set_edgecolor('#666666')
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        ax.add_patch(Rectangle((q25, i - 0.12), q75 - q25, 0.24, facecolor='#444444', alpha=0.25, edgecolor='none'))
        ax.vlines(q50, i - 0.3, i + 0.3, colors='#222222', linewidth=2.5)

    def draw_vertical_violin(ax, pos, vals, color="#bbbbbb"):
        """Vertical violin with a dark IQR box and median line."""
        if vals.size == 0:
            return
        parts = ax.violinplot(
            [vals],
            positions=[pos],
            vert=True,
            showextrema=False,
            widths=0.6,
        )
        for b in parts["bodies"]:
            b.set_facecolor(color)
            b.set_alpha(0.25)
            b.set_edgecolor("#444444")
            b.set_linewidth(1.2)
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        # IQR box
        ax.add_patch(
            Rectangle(
                (pos - 0.12, q25),
                0.24,
                q75 - q25,
                facecolor="#444444",
                alpha=0.25,
                edgecolor="none",
            )
        )
        # median line (horizontal across the violin)
        ax.hlines(q50, pos - 0.18, pos + 0.18, colors="#222222", linewidth=2.5)

    # Baseline violins
    axB.set_yticks(y_positions)
    axB.set_yticklabels([])
    for i, (_sig, _iso, vals_base) in enumerate(rows_base, start=1):
        draw_simple_violin(axB, i, vals_base)
    axB.tick_params(axis='both', labelsize=11)
    for spine in ["top", "right"]:
        axB.spines[spine].set_visible(False)
    axB.set_xlabel('Baseline Activity (log2FC)', fontsize=12)
    axB.invert_yaxis()

    # Variation violins when requested (log10(metric), metric > 0)
    if baseline_and_variation and axV is not None and rows_var:
        axV.set_yticks(y_positions)
        axV.set_yticklabels([])

        metric_label = "sd_ftrend" if variation_metric.lower().strip() in {"sd", "sd_ftrend", "ftrend"} else "F-stat"

        for i, (_sig, _iso, vals_var) in enumerate(rows_var, start=1):
            vals_var = np.asarray(vals_var, dtype=float)
            vals_pos = vals_var[vals_var > 0]
            if vals_pos.size == 0:
                continue
            vals_log = np.log10(vals_pos)
            draw_simple_violin(axV, i, vals_log)

        axV.tick_params(axis="both", labelsize=11)
        for spine in ["top", "right"]:
            axV.spines[spine].set_visible(False)
        axV.set_xlabel(f"log10 patient variability ({metric_label})", fontsize=12)
        axV.invert_yaxis()

    # Per-pair statistical tests: compare SP- vs SP+ within each matched architecture
    if baseline_and_variation and pair_sp_only and axV is not None and var_df is not None and rows_var:
        # signature -> plotted row index (1-based)
        sig_to_row = {sig: i for i, (sig, _iso, _vals) in enumerate(rows_var, start=1)}
        # signature -> raw metric values (not log-transformed)
        var_lookup = {sig: np.asarray(vals, dtype=float) for sig, _iso, vals in rows_var}

        # Determine the matched SP- vs SP+ pairs from the underlying presence table
        pairs = build_sp_pairs(pres, ordered_sigs)

        # Place the text near the right side of the variation panel.
        x0, x1 = axV.get_xlim()
        x_text = x1 - 0.02 * (x1 - x0)
        x_br = x1 - 0.06 * (x1 - x0)
        tick = 0.01 * (x1 - x0)

        for sig_no, sig_yes in pairs:
            if sig_no not in sig_to_row or sig_yes not in sig_to_row:
                continue
            r1 = sig_to_row[sig_no]
            r2 = sig_to_row[sig_yes]
            # ensure r1 is the top row (smaller index after invert_yaxis)
            y_top = min(r1, r2)
            y_bot = max(r1, r2)
            y_center = (y_top + y_bot) / 2.0

            x = var_lookup.get(sig_no, np.array([]))
            y = var_lookup.get(sig_yes, np.array([]))
            x = x[np.isfinite(x) & (x > 0)]
            y = y[np.isfinite(y) & (y > 0)]
            if x.size == 0 or y.size == 0:
                continue

            # For Welch t-test, use log10(metric) (caller filters metric>0).
            stat, pval = compare_two_distributions(
                x,
                y,
                test=pair_test,
                log_transform_for_t=True,
            )
            delta = cliffs_delta(y, x)  # positive => SP+ tends higher


            # p-value formatting: always show numeric value
            # Use scientific notation for very small p-values.
            if pval < 1e-3:
                ptxt = f"p={pval:.2e}"
            else:
                ptxt = f"p={pval:.4f}".rstrip("0").rstrip(".")

            axV.text(
                x_text,
                y_center,
                f"{ptxt}  Δ={delta:.2f}",
                ha="right",
                va="center",
                fontsize=9,
                color="#222222",
            )

    # (No clade-wise variation violins panel in main figure for baseline_and_variation mode)

    if not baseline_only and not baseline_and_variation and axT is not None and rows_ifng:
        # INFg violins
        axT.set_yticks(y_positions)
        axT.set_yticklabels([])
        for i, (_sig, _iso, vals_ifng) in enumerate(rows_ifng, start=1):
            draw_simple_violin(axT, i, vals_ifng)
        axT.tick_params(axis='both', labelsize=11)
        for spine in ["top", "right"]:
            axT.spines[spine].set_visible(False)
        axT.set_xlabel('INFg Delta Activity (log2FC)', fontsize=12)
        axT.invert_yaxis()

    # Clade legend (top-center)
    clade_handles = [Rectangle((0,0), 1, 1, facecolor=CLADE_COLORS[c], edgecolor=EDGE, linewidth=0.4)
                     for c in CLADE_ORDER]
    clade_labels = CLADE_ORDER
    fig.legend(clade_handles, clade_labels, loc='upper center', ncol=len(CLADE_ORDER),
               frameon=False, fontsize=9, bbox_to_anchor=(0.7, 0.99))

    # Site-type legend: place on the right-most panel using automatic positioning
    # to minimize overlap with plotted data.
    site_labels = ["IRF_x2", "IRF_x3", "E2F", "SP/KLF"]

    # Choose the most appropriate axis to host the legend
    if baseline_and_variation and axV is not None:
        legend_ax = axV
    elif (not baseline_only) and (axT is not None):
        legend_ax = axT
    else:
        legend_ax = axB

    legend_ax.legend(
        legend_handles,
        site_labels,
        loc="best",
        frameon=False,
        fontsize=10,
    )

    fig.subplots_adjust(left=0.04, right=0.995, top=0.93, bottom=0.08)

    out_pdf = Path(str(outprefix) + '.pdf')
    out_png = Path(str(outprefix) + '.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {out_pdf}\nSaved figure: {out_png}")

    # --- Global SP/KLF pooled comparison (2 violins) ---
    if baseline_and_variation and (var_df is not None):
        vmetric = "sd_ftrend" if variation_metric.lower().strip() in {"sd", "sd_ftrend", "ftrend"} else "fstat"
        sp_minus, sp_plus = pooled_variation_by_sp(pres, var_df, metric=variation_metric)

        # Filter to positive values for log10 plotting (and for the test; monotone transform doesn't change ranks)
        sp_minus = sp_minus[np.isfinite(sp_minus) & (sp_minus > 0)]
        sp_plus = sp_plus[np.isfinite(sp_plus) & (sp_plus > 0)]

        # Save log10(metric) values to match the plotted scale
        out3_prefix = Path(str(outprefix) + f"_SPglobal_{vmetric}")
        out3_vals_csv = Path(str(out3_prefix) + "_values.csv")
        sp_plus_log_csv = np.log10(sp_plus)
        sp_minus_log_csv = np.log10(sp_minus)
        max_len = int(max(sp_plus_log_csv.size, sp_minus_log_csv.size))
        df_sp = pd.DataFrame({
            "SP_plus": pd.Series(sp_plus_log_csv).reindex(range(max_len)),
            "SP_minus": pd.Series(sp_minus_log_csv).reindex(range(max_len)),
        })
        df_sp.to_csv(out3_vals_csv, index=False)
        print(f"Saved global SP pooled values CSV (log10 scale): {out3_vals_csv}")

        if (sp_minus.size > 0) and (sp_plus.size > 0):
            # Brunner–Munzel (two-sided)
            stat, pval = compare_two_distributions(sp_minus, sp_plus, test="brunnermunzel")
            delta = cliffs_delta(sp_plus, sp_minus)  # positive => SP+ tends higher

            # Plot on log10 scale for readability
            sp_minus_log = np.log10(sp_minus)
            sp_plus_log = np.log10(sp_plus)

            fig3, ax3 = plt.subplots(figsize=(2.8, 4.8))
            draw_vertical_violin(ax3, 1, sp_plus_log, color=FAMILY_COLORS.get("SP/KLF", "#bbbbbb"))
            draw_vertical_violin(ax3, 2, sp_minus_log, color="#f2f2f2")

            ax3.set_xticks([1, 2])
            ax3.set_xticklabels([
                f"SP+ (n={sp_plus.size})",
                f"SP- (n={sp_minus.size})",
            ], fontsize=10)

            # Tighten x-limits and margins to emphasize thin width.
            ax3.set_xlim(0.5, 2.5)
            fig3.subplots_adjust(left=0.28, right=0.98, top=0.88, bottom=0.18)

            # Exact p-value formatting
            if pval < 1e-3:
                ptxt = f"p={pval:.2e}"
            else:
                ptxt = f"p={pval:.4f}".rstrip("0").rstrip(".")

            ax3.set_ylabel(f"log10 patient variability ({vmetric})", fontsize=12)
            ax3.set_title(f"Global SP/KLF effect (Brunner–Munzel)\n{ptxt}  Δ={delta:.2f}", fontsize=11)

            for spine in ["top", "right"]:
                ax3.spines[spine].set_visible(False)

            out3_pdf = Path(str(out3_prefix) + ".pdf")
            out3_png = Path(str(out3_prefix) + ".png")
            fig3.savefig(out3_pdf, bbox_inches="tight")
            fig3.savefig(out3_png, dpi=300, bbox_inches="tight")
            plt.close(fig3)
            print(f"Saved global SP pooled figure: {out3_pdf}\nSaved global SP pooled figure: {out3_png}")

    # --- Standalone clade-only violin plot (for baseline_and_variation mode) ---
    # Also export a TSV/CSV with raw sd_ftrend/fstat values per clade and an "all" column.
    if baseline_and_variation and (var_df is not None):
        # Build clade column
        var_clade = var_df.copy()
        vmetric = "sd_ftrend" if variation_metric.lower().strip() in {"sd", "sd_ftrend", "ftrend"} else "fstat"
        var_clade["SuperClade"] = var_clade["isolate"].map(iso2clade).fillna("Other")
        clades_present = [c for c in CLADE_ORDER if (var_clade["SuperClade"] == c).any()]
        if clades_present:
            # --- CSV with raw vmetric per clade and an "all" column ---
            clade_series = {}
            for cl in clades_present:
                vals_raw = (
                    var_clade.loc[var_clade["SuperClade"] == cl, vmetric]
                    .astype(float)
                    .dropna()
                    .reset_index(drop=True)
                )
                clade_series[cl] = vals_raw

            all_vals = (
                var_clade[vmetric]
                .astype(float)
                .dropna()
                .reset_index(drop=True)
            )
            clade_series["all"] = all_vals

            max_len = max(len(s) for s in clade_series.values())
            df_clades = pd.DataFrame(
                {name: s.reindex(range(max_len)) for name, s in clade_series.items()}
            )
            clade_outprefix = Path(str(outprefix) + f"_cladeVariationOnly_{vmetric}")
            out2_csv = Path(str(clade_outprefix) + f"_{vmetric}_values.tsv")
            df_clades.to_csv(out2_csv, sep="\t", index=False)

            # --- TSV that MATCHES the violin plot: log10(vmetric), vmetric > 0 ---
            clade_series_log = {}

            for cl in clades_present:
                vals_raw = (
                    var_clade.loc[var_clade["SuperClade"] == cl, vmetric]
                    .astype(float)
                    .dropna()
                )
                vals_pos = vals_raw[vals_raw > 0].reset_index(drop=True)
                clade_series_log[cl] = np.log10(vals_pos)

            all_vals_raw = var_clade[vmetric].astype(float).dropna()
            all_vals_pos = all_vals_raw[all_vals_raw > 0].reset_index(drop=True)
            clade_series_log["all"] = np.log10(all_vals_pos)

            max_len_log = max(len(v) for v in clade_series_log.values())
            df_clades_log = pd.DataFrame(
                {k: pd.Series(v).reindex(range(max_len_log)) for k, v in clade_series_log.items()}
            )

            out2_csv_log = Path(str(clade_outprefix) + f"_log10{vmetric}_values.tsv")
            df_clades_log.to_csv(out2_csv_log, sep="\t", index=False)

            print(f"Saved RAW {vmetric} TSV: {out2_csv}")
            print(f"Saved PLOT-MATCHING log10({vmetric}) TSV: {out2_csv_log}")

            # --- Log10 vmetric clade-only violin plot (as before) ---
            positions = np.arange(1, len(clades_present) + 1)
            fig2, ax2 = plt.subplots(figsize=(max(4.0, 0.8 * len(clades_present)), 4.5))
            for pos, cl in zip(positions, clades_present):
                vals = (
                    var_clade.loc[var_clade["SuperClade"] == cl, vmetric]
                    .astype(float)
                    .dropna()
                    .values
                )
                vals = vals[vals > 0]
                if vals.size == 0:
                    continue
                vals_log = np.log10(vals)
                color = CLADE_COLORS.get(cl, "#bbbbbb")
                draw_vertical_violin(ax2, pos, vals_log, color=color)
            ax2.set_xticks(positions)
            ax2.set_xticklabels(clades_present, rotation=45, ha="right", fontsize=9)
            ax2.set_ylabel(f"log10 patient variability ({vmetric})", fontsize=12)
            ax2.set_title(f"Variation by clade (Tile 13 — {vmetric})", fontsize=12)
            for spine in ["top", "right"]:
                ax2.spines[spine].set_visible(False)
            out2_pdf = Path(str(clade_outprefix) + ".pdf")
            out2_png = Path(str(clade_outprefix) + ".png")
            fig2.savefig(out2_pdf, bbox_inches="tight")
            fig2.savefig(out2_png, dpi=300, bbox_inches="tight")
            plt.close(fig2)
            print(f"Saved clade-variation-only figure: {out2_pdf}\nSaved clade-variation-only figure: {out2_png}")
            print(f"Saved clade-variation {vmetric} table: {out2_csv}")


def main():
    p = argparse.ArgumentParser(description='Plot activity violins per motif grammar with 4-slot grammar glyphs (Tile 13)')
    p.add_argument('--presence', default='../results/motif_counts/tile13_site_presence_fixedbins.tsv')
    p.add_argument('--counts',   default='../results/motif_counts/tile13_site_combination_counts_fixedbins.tsv')
    p.add_argument('--baseline', default='../data/activity/OL53_run_Jurkat_berkay_activity.tsv')
    p.add_argument('--ifng',     default='../data/activity/comparison_INFg_vs_Ctrl.tsv')
    p.add_argument('--clades',   default='../data/clades.tsv')
    p.add_argument('--variation', default=None,
                   help='TSV with per-tile variation metrics (must include ID and sd_ftrend).')
    p.add_argument('--min-n', type=int, default=10)
    p.add_argument('--order-by', choices=['baseline','ifng'], default='baseline', help='Order grammars by median activity of this condition')
    p.add_argument('--outfig', default='../results/figures/tile13_grammar_activity')
    p.add_argument('--baseline-only', action='store_true',
                   help='Plot only baseline violins (no IFNg panel).')
    p.add_argument('--baseline-and-variation', action='store_true',
                   help='Plot baseline violins and a patient-variation (sd_ftrend) panel (no IFNg panel).')
    p.add_argument(
        "--variation-metric",
        choices=["sd_ftrend", "fstat"],
        default="sd_ftrend",
        help="Which variation metric to plot/test (sd_ftrend or fstat).",
    )
    p.add_argument(
        "--pair-sp-only",
        action="store_true",
        help="Group matched grammar pairs that differ only by SP/KLF_site1 (SP- vs SP+) so they appear adjacent; keep unpaired grammars as well.",
    )
    p.add_argument(
        "--pair-test",
        choices=["brunnermunzel", "mannwhitney", "welch"],
        default="brunnermunzel",
        help="Two-sided test for SP- vs SP+ within each matched pair (brunnermunzel, mannwhitney, welch). Welch is run on log10(metric) after filtering metric>0.",
    )
    args = p.parse_args()

    outprefix = Path(str(args.outfig) + f'_{args.order_by}Ordered')
    variation_path = Path(args.variation) if args.variation is not None else None

    plot_grammars_with_activity(
        Path(args.presence),
        Path(args.counts),
        Path(args.baseline),
        Path(args.ifng),
        Path(args.clades),
        variation_path,
        args.min_n,
        outprefix,
        args.order_by,
        baseline_only=args.baseline_only,
        baseline_and_variation=args.baseline_and_variation,
        variation_metric=args.variation_metric,
        pair_sp_only=args.pair_sp_only,
        pair_test=args.pair_test,
    )


if __name__ == '__main__':
    main()
