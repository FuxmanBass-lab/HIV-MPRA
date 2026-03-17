#!/usr/bin/env python3
"""
Plot activity distributions (log2FoldChange) per motif grammar for Tile 6.

Inputs expected (produced by 06_tile6_motif_grammar.py):
  - ../results/motif_counts/tile6_site_presence_fixedbins.tsv
  - ../results/motif_counts/tile6_site_combination_counts_fixedbins.tsv
And activity metrics:
  - ../data/activity/OL53_run_Jurkat_berkay_activity.tsv
  - ../data/activity/comparison_StimJurkat_vs_Jurkat_berkay.tsv
  - ../data/activity/comparison_TNF_vs_Ctrl.tsv

Behavior
--------
- Keeps only motif grammars (signatures) with ≥ 10 isolates.
- Y-axis lists grammars; for each y-row we draw a 7-slot rectangle representing
  sites: [N1, N2, N3, (N4|S1), S2, S3, S4].
    * Fill color for present sites:
        - NFKB/REL: teal (#008080)
        - SP/KLF  : purple (#6A0DAD)
      Absent sites are light gray (#eeeeee) with a thin edge.
      For the shared bin (N4|S1) if both are present, we split the segment into
      two halves (left teal, right purple).
- X-axis is activity (log2FoldChange) as horizontal violins for isolates within
  each grammar.
- Shows three violin panels: baseline (Jurkat), Stim (Jurkat), and TNF.
- Saves PDF and PNG.

Usage
-----
python 07_tile6_plot_grammar_activity.py \
  --presence ../results/motif_grammar/tile6_site_presence_fixedbins.tsv \
  --counts   ../results/motif_grammar/tile6_site_combination_counts_fixedbins.tsv \
  --baseline ../data/activity/OL53_run_Jurkat_berkay_activity.tsv \
  --stim     ../data/activity/comparison_StimJurkat_vs_Jurkat_berkay.tsv \
  --tnf      ../data/activity/comparison_TNF_vs_Ctrl.tsv \
  --min-n 3 \
  --order-by stim \
  --outfig ../results/figures/tile6_grammar_activity_3iso

python 10_plot_patients_activity_variation.py \
  --presence ../results/motif_grammar/tile6_site_presence_fixedbins.tsv \
  --counts   ../results/motif_grammar/tile6_site_combination_counts_fixedbins.tsv \
  --baseline ../data/activity/OL53_T_primaryT_activity.tsv \
  --variation ../results/patients_variation/activity_vs_variance_activity_variation_metrics.tsv \
  --min-n 3 \
  --order-by baseline \
  --baseline-and-variation\
  --outfig ../results/figures/variation/tile6_grammar_activity_CD4_3iso
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge

FAM_NFKB = "NFKB/REL"
FAM_SPKLF = "SP/KLF"

# Colors
COL_NFKB = "#66a3a3"   # muted teal
COL_SP   = "#9370db"   # muted purple (medium purple)
COL_ABS  = "#eeeeee"
EDGE     = "#555555"

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

SITES = [
    f"{FAM_NFKB}_site1",
    f"{FAM_NFKB}_site2",
    f"{FAM_NFKB}_site3",
    # shared slot: (NFKB_site4 | SP_site1)
    (f"{FAM_NFKB}_site4", f"{FAM_SPKLF}_site1"),
    f"{FAM_SPKLF}_site2",
    f"{FAM_SPKLF}_site3",
    f"{FAM_SPKLF}_site4",
]



# --- Helper to count NFKB/REL sites in a grammar signature ---
def count_nfkB_sites_in_signature(pres_row: pd.Series) -> int:
    """Count how many NFKB/REL sites (site1..site4) are present for a grammar."""
    n = 0
    for k in range(1, 5):
        col = f"{FAM_NFKB}_site{k}"
        if bool(pres_row.get(col, False)):
            n += 1
    return n

# --- Helper to count SP/KLF sites in a grammar signature ---

def count_sp_sites_in_signature(pres_row: pd.Series) -> int:
    """Count how many SP/KLF sites (site1..site4) are present for a grammar."""
    n = 0
    for k in range(1, 5):
        col = f"{FAM_SPKLF}_site{k}"
        if bool(pres_row.get(col, False)):
            n += 1
    return n


# --- Helper to count total motif sites (NFKB/REL 1..4 and SP/KLF 1..4) in a grammar signature ---
def count_total_sites_in_signature(pres_row: pd.Series) -> int:
    """Count total present sites across NFKB/REL (1..4) and SP/KLF (1..4)."""
    n = 0
    for k in range(1, 5):
        if bool(pres_row.get(f"{FAM_NFKB}_site{k}", False)):
            n += 1
        if bool(pres_row.get(f"{FAM_SPKLF}_site{k}", False)):
            n += 1
    return n


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
    # Filter to tile 6 only using the pattern :6:
    act = act[act["ID"].astype(str).str.contains(r":6:")].copy()
    act["isolate"] = act["ID"].map(extract_isolate_from_id)
    return act[["ID", "isolate", "log2FoldChange"]]



def collect_activity_by_signature(pres: pd.DataFrame, order: list[str], act: pd.DataFrame):
    # pres has one row per isolate with boolean columns and signature
    data = []
    for sig in order:
        iso = pres.loc[pres["signature"] == sig, "isolate"].astype(str).tolist()
        vals = act.loc[act["isolate"].isin(iso), "log2FoldChange"].astype(float).dropna().values
        data.append((sig, iso, vals))
    return data


# --- New helper for loading variation metrics ---
def load_variation(variation_path: Path) -> pd.DataFrame:
    """Load per-tile variation metrics and extract sd_Ftrend for Tile 6, keyed by isolate."""
    var = pd.read_csv(variation_path, sep="\t")
    if "ID" not in var.columns:
        raise ValueError(f"Variation file {variation_path} must contain an 'ID' column.")
    # Try to find the sd_Ftrend-like column, preferring sd_Ftrend first
    sd_col = None
    for candidate in ["sd_Ftrend", "sd_Ftrend_ctrl", "sd_ftrend", "sd_ftrend_ctrl"]:
        if candidate in var.columns:
            sd_col = candidate
            break
    if sd_col is None:
        raise ValueError(
            f"Variation file {variation_path} must contain one of sd_Ftrend/sd_Ftrend_ctrl columns."
        )
    # Restrict to Tile 6 entries
    var = var[var["ID"].astype(str).str.contains(r":6:")].copy()
    var["isolate"] = var["ID"].map(extract_isolate_from_id)
    var = var[["ID", "isolate", sd_col]].rename(columns={sd_col: "sd_Ftrend"})
    return var


# --- New helper to collect sd_Ftrend values by motif grammar ---
def collect_variation_by_signature(pres: pd.DataFrame, order: list[str], var_df: pd.DataFrame):
    """For each signature in order, collect the sd_Ftrend values across isolates."""
    data = []
    for sig in order:
        iso = pres.loc[pres["signature"] == sig, "isolate"].astype(str).tolist()
        vals = var_df.loc[var_df["isolate"].isin(iso), "sd_Ftrend"].astype(float).dropna().values
        data.append((sig, iso, vals))
    return data


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
    """Draw the 7-slot rectangle for one grammar at y=row_idx on ax.
    x0/width are in axis data units (we'll use a separate axes with unit scale).
    """
    seg_w = width / 7.0
    y = row_idx - height/2

    def add_rect(x, w, color):
        ax.add_patch(Rectangle((x, y), w, height, facecolor=color, edgecolor=EDGE, linewidth=0.5))

    for k in range(7):
        slot = SITES[k]
        x = x0 + k * seg_w
        if isinstance(slot, tuple):
            n4, s1 = slot
            has_n4 = bool(pres_row.get(n4, False))
            has_s1 = bool(pres_row.get(s1, False))
            if has_n4 and has_s1:
                # split half/half
                add_rect(x, seg_w/2, COL_NFKB)
                add_rect(x + seg_w/2, seg_w/2, COL_SP)
            elif has_n4:
                add_rect(x, seg_w, COL_NFKB)
            elif has_s1:
                add_rect(x, seg_w, COL_SP)
            else:
                add_rect(x, seg_w, COL_ABS)
        else:
            present = bool(pres_row.get(slot, False))
            if not present:
                add_rect(x, seg_w, COL_ABS)
            else:
                color = COL_NFKB if "NFKB/REL" in slot else COL_SP
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
    stim_path: Path,
    tnf_path: Path,
    clades_path: Path,
    min_n: int,
    outprefix: Path,
    order_by: str = "stim",
    variation_path: Path | None = None,
    baseline_only: bool = False,
    baseline_and_variation: bool = False,
):
    pres, cnts, order = load_presence_counts(presence_path, counts_path, min_n)
    act_base = load_activity(base_path)
    var_df = None
    if baseline_and_variation:
        if variation_path is None:
            raise ValueError("baseline_and_variation requested but no variation_path was provided.")
        var_df = load_variation(variation_path)
    if not baseline_only and not baseline_and_variation:
        act_stim = load_activity(stim_path)
        act_tnf = load_activity(tnf_path)
    else:
        act_stim = None
        act_tnf = None

    iso2clade = load_isolate_to_clade_map(clades_path)

    # Collect activity rows using initial order
    rows_base_o = collect_activity_by_signature(pres, order, act_base)
    if not baseline_only and not baseline_and_variation:
        rows_stim_o = collect_activity_by_signature(pres, order, act_stim)
        rows_tnf_o  = collect_activity_by_signature(pres, order, act_tnf)
    else:
        # When in baseline_only or baseline_and_variation mode, we don’t have stim/TNF
        rows_stim_o = []
        rows_tnf_o  = []

    # Collect variation rows in the initial (unsorted) order if needed
    rows_var_o = []
    if baseline_and_variation and var_df is not None:
        rows_var_o = collect_variation_by_signature(pres, order, var_df)

    # Choose which condition determines row ordering (by median, desc)
    order_src = order_by.lower().strip()

    # Decide ordering source
    if order_src == "variation":
        if not baseline_and_variation or var_df is None:
            raise ValueError("order-by variation requires --baseline-and-variation and a variation file.")
        rows_src = rows_var_o
        # use median log10(sd_Ftrend) for ordering
        med_list = []
        for sig, _iso, vals in rows_src:
            vals = np.asarray(vals, dtype=float)
            vals = vals[vals > 0]
            if vals.size == 0:
                median_val = np.nan
            else:
                median_val = float(np.median(np.log10(vals)))
            med_list.append((sig, median_val))
    else:
        if baseline_only or baseline_and_variation:
            order_src = "baseline"
        if order_src not in {"stim", "baseline", "tnf"}:
            order_src = "stim"
        rows_src_map = {"stim": rows_stim_o, "baseline": rows_base_o, "tnf": rows_tnf_o}
        rows_src = rows_src_map[order_src]
        med_list = []
        for sig, _iso, vals in rows_src:
            median_val = float(np.median(vals)) if vals.size > 0 else np.nan
            med_list.append((sig, median_val))

    # Final ordering: NaNs last, descending
    ordered_sigs = [
        sig for sig, _ in sorted(
            med_list,
            key=lambda x: (np.isnan(x[1]), -x[1] if not np.isnan(x[1]) else 0)
        )
    ]

    # Build rows in the chosen order
    rows_base = collect_activity_by_signature(pres, ordered_sigs, act_base)
    if not baseline_only and not baseline_and_variation:
        rows_stim = collect_activity_by_signature(pres, ordered_sigs, act_stim)
        rows_tnf  = collect_activity_by_signature(pres, ordered_sigs, act_tnf)
    else:
        rows_stim = rows_base  # reuse for layout (same signatures/order)
        rows_tnf  = []

    rows_var = []
    if baseline_and_variation and var_df is not None:
        rows_var = collect_variation_by_signature(pres, ordered_sigs, var_df)

    # Use rows_stim (or rows_base in baseline-only mode) to count grammars
    if baseline_only or baseline_and_variation:
        grammar_rows = rows_base
    else:
        grammar_rows = rows_stim
    n = len(grammar_rows)
    if n == 0:
        print("No grammar groups meet the minimum isolate threshold.")
        return

    if baseline_only:
        # 3-column layout: pies, grammars, baseline violins
        fig = plt.figure(figsize=(12, max(3.5, 0.6 * n)))
        gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[0.95, 0.95, 3.0])
        axP = fig.add_subplot(gs[0, 0])  # pies axis (leftmost)
        axG = fig.add_subplot(gs[0, 1])  # grammar glyphs + labels
        axB = fig.add_subplot(gs[0, 2])
        axS = None
        axT = None
        axV = None
        axC = None
    elif baseline_and_variation:
        # 4-column layout: pies, grammars, baseline violins, SD_Ftrend violins
        fig = plt.figure(figsize=(16, max(3.5, 0.6 * n)))
        gs = fig.add_gridspec(
            nrows=1,
            ncols=4,
            width_ratios=[0.95, 0.95, 3.0, 3.0],
        )
        axP = fig.add_subplot(gs[0, 0])  # pies axis (leftmost)
        axG = fig.add_subplot(gs[0, 1])  # grammar glyphs + labels
        axB = fig.add_subplot(gs[0, 2])
        axS = None
        axT = None
        axV = fig.add_subplot(gs[0, 3])  # per-grammar variation
        axC = None
        gs.update(wspace=0.08)
    else:
        # 5-column layout: pies, grammars, baseline, Stim, TNF
        fig = plt.figure(figsize=(18, max(3.5, 0.6 * n)))
        gs = fig.add_gridspec(nrows=1, ncols=5, width_ratios=[0.95, 0.95, 3.0, 3.0, 3.0])
        axP = fig.add_subplot(gs[0, 0])  # pies axis (leftmost)
        axG = fig.add_subplot(gs[0, 1])  # grammar glyphs + labels
        axB = fig.add_subplot(gs[0, 2])
        axS = fig.add_subplot(gs[0, 3])
        axT = fig.add_subplot(gs[0, 4])
        axV = None
        axC = None
        gs.update(wspace=0.08)  # Bring glyphs a touch closer to pies

    # Prepare legend handles (figure-level legend placed in a clear area)
    legend_handles = [
        Rectangle((0,0), 1, 1, facecolor=COL_NFKB, edgecolor=EDGE, linewidth=0.5),
        Rectangle((0,0), 1, 1, facecolor=COL_SP, edgecolor=EDGE, linewidth=0.5),
    ]

    # Prepare y positions (top-to-bottom)
    y_positions = np.arange(1, n + 1)
    axP.set_ylim(0.5, n + 0.5)
    axG.set_ylim(0.5, n + 0.5)
    axB.set_ylim(0.5, n + 0.5)
    if axS is not None:
        axS.set_ylim(0.5, n + 0.5)
    if axT is not None:
        axT.set_ylim(0.5, n + 0.5)
    if "axV" in locals() and axV is not None:
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
    for i, (sig, iso_list, vals) in enumerate(grammar_rows, start=1):
        # representative row from presence for drawing the sites
        pres_row = pres[pres["signature"] == sig].iloc[0]
        draw_grammar_rect(axG, i, pres_row, x0=0.0, width=7.0, height=0.8)
        count = len(iso_list)
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
    axG.set_title("Motif grammars (Tile 6)", fontsize=12)
    axG.set_xlim(0, 7.0)
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
            b.set_edgecolor("#666666")
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
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
        ax.vlines(pos, q50, q50, colors="#222222", linewidth=2.5)

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

    # Variation violins (log10 sd_Ftrend) when requested
    if baseline_and_variation and axV is not None and rows_var:
        axV.set_yticks(y_positions)
        axV.set_yticklabels([])
        for i, (_sig, _iso, vals_var) in enumerate(rows_var, start=1):
            vals_var = np.asarray(vals_var, dtype=float)
            vals_pos = vals_var[vals_var > 0]
            if vals_pos.size == 0:
                continue
            vals_log = np.log10(vals_pos)
            draw_simple_violin(axV, i, vals_log)
        axV.tick_params(axis='both', labelsize=11)
        for spine in ["top", "right"]:
            axV.spines[spine].set_visible(False)
        axV.set_xlabel('log10 patient variability (sd_Ftrend)', fontsize=12)
        axV.invert_yaxis()
        # Cut the variation x-axis at -0.3
        xmin, xmax = axV.get_xlim()
        axV.set_xlim(left=-0.3, right=xmax)

    # (Clade-wise variation violins removed from main composite figure.)

    # Separate clade-wise-only variation figure (log10 sd_Ftrend by clade)
    # and CSV export of raw sd_Ftrend per clade plus an "all" column.
    if baseline_and_variation and (var_df is not None):
        var_clade2 = var_df.copy()
        var_clade2["SuperClade"] = var_clade2["isolate"].map(iso2clade).fillna("Other")

        clades_present2 = [c for c in CLADE_ORDER if (var_clade2["SuperClade"] == c).any()]
        if clades_present2:
            # --- CSV with raw sd_Ftrend per clade and an "all" column ---
            clade_series = {}
            for cl in clades_present2:
                vals_raw = (
                    var_clade2.loc[var_clade2["SuperClade"] == cl, "sd_Ftrend"]
                    .astype(float)
                    .dropna()
                    .reset_index(drop=True)
                )
                clade_series[cl] = vals_raw

            all_vals = (
                var_clade2["sd_Ftrend"]
                .astype(float)
                .dropna()
                .reset_index(drop=True)
            )
            clade_series["all"] = all_vals

            max_len = max(len(s) for s in clade_series.values())
            df_clades = pd.DataFrame(
                {name: s.reindex(range(max_len)) for name, s in clade_series.items()}
            )
            outprefix_clade = Path(str(outprefix) + "_cladeVariationOnly")
            out_csv_c = Path(str(outprefix_clade) + "_sdFtrend_values.tsv")
            df_clades.to_csv(out_csv_c, sep="\t", index=False)

            # --- TSV that MATCHES the violin plot: log10(sd_Ftrend), sd_Ftrend > 0 ---
            clade_series_log = {}

            for cl in clades_present2:
                vals_raw = (
                    var_clade2.loc[var_clade2["SuperClade"] == cl, "sd_Ftrend"]
                    .astype(float)
                    .dropna()
                )
                vals_pos = vals_raw[vals_raw > 0].reset_index(drop=True)
                clade_series_log[cl] = np.log10(vals_pos)

            all_vals_raw = var_clade2["sd_Ftrend"].astype(float).dropna()
            all_vals_pos = all_vals_raw[all_vals_raw > 0].reset_index(drop=True)
            clade_series_log["all"] = np.log10(all_vals_pos)

            max_len_log = max(len(v) for v in clade_series_log.values())
            df_clades_log = pd.DataFrame(
                {k: pd.Series(v).reindex(range(max_len_log)) for k, v in clade_series_log.items()}
            )

            out_csv_log = Path(str(outprefix_clade) + "_log10sdFtrend_values.tsv")
            df_clades_log.to_csv(out_csv_log, sep="\t", index=False)

            print(f"Saved RAW sd_Ftrend TSV: {out_csv_c}")
            print(f"Saved PLOT-MATCHING log10(sd_Ftrend) TSV: {out_csv_log}")

            # --- Log-scale clade-only violin plot (unchanged behavior) ---
            positions2 = np.arange(1, len(clades_present2) + 1)
            figC, axC_only = plt.subplots(
                figsize=(max(4, 0.6 * len(clades_present2)), 4)
            )
            for pos, cl in zip(positions2, clades_present2):
                vals = (
                    var_clade2.loc[var_clade2["SuperClade"] == cl, "sd_Ftrend"]
                    .astype(float)
                    .dropna()
                    .values
                )
                vals = vals[vals > 0]
                if vals.size == 0:
                    continue
                vals_log = np.log10(vals)
                color = CLADE_COLORS.get(cl, "#bbbbbb")
                draw_vertical_violin(axC_only, pos, vals_log, color=color)

            axC_only.set_xticks(positions2)
            axC_only.set_xticklabels(clades_present2, rotation=45, ha="right", fontsize=9)
            axC_only.tick_params(axis="both", labelsize=10)
            for spine in ["top", "right"]:
                axC_only.spines[spine].set_visible(False)
            axC_only.set_ylabel("log10 patient variability (sd_Ftrend)", fontsize=12)
            axC_only.set_title("Variation by clade", fontsize=12)

            out_pdf_c = Path(str(outprefix_clade) + ".pdf")
            out_png_c = Path(str(outprefix_clade) + ".png")
            figC.savefig(out_pdf_c, bbox_inches="tight")
            figC.savefig(out_png_c, dpi=300, bbox_inches="tight")
            plt.close(figC)

    if not baseline_only and axS is not None:
        # Stim violins
        axS.set_yticks(y_positions)
        axS.set_yticklabels([])
        for i, (_sig, _iso, vals_stim) in enumerate(rows_stim, start=1):
            draw_simple_violin(axS, i, vals_stim)
        axS.tick_params(axis='both', labelsize=11)
        for spine in ["top", "right"]:
            axS.spines[spine].set_visible(False)
        axS.set_xlabel('PMA+αCD3 Delta Activity (log2FC)', fontsize=12)
        axS.invert_yaxis()

    if not baseline_only and axT is not None:
        # TNF violins
        axT.set_yticks(y_positions)
        axT.set_yticklabels([])
        for i, (_sig, _iso, vals_tnf) in enumerate(rows_tnf, start=1):
            draw_simple_violin(axT, i, vals_tnf)
        axT.tick_params(axis='both', labelsize=11)
        for spine in ["top", "right"]:
            axT.spines[spine].set_visible(False)
        axT.set_xlabel('TNF Delta Activity (log2FC)', fontsize=12)
        axT.invert_yaxis()

    # Clade legend (top-center)
    clade_handles = [Rectangle((0,0), 1, 1, facecolor=CLADE_COLORS[c], edgecolor=EDGE, linewidth=0.4)
                     for c in CLADE_ORDER]
    clade_labels = CLADE_ORDER
    fig.legend(clade_handles, clade_labels, loc='upper center', ncol=len(CLADE_ORDER),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.99))

    # Site-type legend (position depends on layout)
    if baseline_only:
        # In 3-column layout, tuck it near the right edge of the baseline panel
        fig.legend(
            legend_handles,
            ["NFKB/REL", "SP/KLF"],
            loc="lower right",
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(0.98, 0.15),
        )
    else:
        # In 5-column layout, keep it slightly inset from the right
        fig.legend(
            legend_handles,
            ["NFKB/REL", "SP/KLF"],
            loc="lower right",
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(0.98, 0.15),
        )

    fig.subplots_adjust(left=0.04, right=0.995, top=0.93, bottom=0.08)

    out_pdf = Path(str(outprefix) + '.pdf')
    out_png = Path(str(outprefix) + '.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {out_pdf}\nSaved figure: {out_png}")



    # --- Auxiliary plot: number of NFKB sites vs patient variation (per-grammar) ---
    if baseline_and_variation and rows_var:
        x_counts = []
        y_meds = []
        sigs = []

        for (sig, _iso, vals_var) in rows_var:
            # Count NFKB sites from the grammar definition
            pres_row = pres[pres["signature"] == sig].iloc[0]
            n_nfkB = count_nfkB_sites_in_signature(pres_row)

            vals_var = np.asarray(vals_var, dtype=float)
            vals_pos = vals_var[vals_var > 0]
            if vals_pos.size == 0:
                continue
            med_log = float(np.median(np.log10(vals_pos)))

            sigs.append(sig)
            x_counts.append(n_nfkB)
            y_meds.append(med_log)

        if len(x_counts) > 0:
            figNV, axNV = plt.subplots(figsize=(5, 4))
            axNV.scatter(x_counts, y_meds, s=22, alpha=0.8, edgecolors="none")
            axNV.set_xlabel("Number of NFKB/REL sites in grammar", fontsize=11)
            axNV.set_ylabel("Median log10 patient variability (sd_Ftrend)", fontsize=11)
            axNV.set_xticks([0, 1, 2, 3, 4])
            for spine in ["top", "right"]:
                axNV.spines[spine].set_visible(False)

            out_pdf_nv = Path(str(outprefix) + "_nfkbCount_vs_variation.pdf")
            out_png_nv = Path(str(outprefix) + "_nfkbCount_vs_variation.png")
            figNV.savefig(out_pdf_nv, bbox_inches="tight")
            figNV.savefig(out_png_nv, dpi=300, bbox_inches="tight")
            plt.close(figNV)
            print(f"Saved NFKB-count vs variation plot: {out_pdf_nv}\nSaved NFKB-count vs variation plot: {out_png_nv}")



    # --- Auxiliary plot: number of SP/KLF sites vs patient variation (per-grammar) ---
    if baseline_and_variation and rows_var:
        x_counts = []
        y_meds = []

        for (sig, _iso, vals_var) in rows_var:
            pres_row = pres[pres["signature"] == sig].iloc[0]
            n_sp = count_sp_sites_in_signature(pres_row)

            vals_var = np.asarray(vals_var, dtype=float)
            vals_pos = vals_var[vals_var > 0]
            if vals_pos.size == 0:
                continue
            med_log = float(np.median(np.log10(vals_pos)))

            x_counts.append(n_sp)
            y_meds.append(med_log)

        if len(x_counts) > 0:
            figSV, axSV = plt.subplots(figsize=(5, 4))
            axSV.scatter(x_counts, y_meds, s=22, alpha=0.8, edgecolors="none")
            axSV.set_xlabel("Number of SP/KLF sites in grammar", fontsize=11)
            axSV.set_ylabel("Median log10 patient variability (sd_Ftrend)", fontsize=11)
            axSV.set_xticks([0, 1, 2, 3, 4])
            for spine in ["top", "right"]:
                axSV.spines[spine].set_visible(False)

            out_pdf_sv = Path(str(outprefix) + "_spCount_vs_variation.pdf")
            out_png_sv = Path(str(outprefix) + "_spCount_vs_variation.png")
            figSV.savefig(out_pdf_sv, bbox_inches="tight")
            figSV.savefig(out_png_sv, dpi=300, bbox_inches="tight")
            plt.close(figSV)
            print(f"Saved SP-count vs variation plot: {out_pdf_sv}\nSaved SP-count vs variation plot: {out_png_sv}")

    # --- Auxiliary plot: total number of motif sites vs patient variation (per-grammar) ---
    if baseline_and_variation and rows_var:
        x_counts = []
        y_meds = []

        for (sig, _iso, vals_var) in rows_var:
            pres_row = pres[pres["signature"] == sig].iloc[0]
            n_tot = count_total_sites_in_signature(pres_row)

            vals_var = np.asarray(vals_var, dtype=float)
            vals_pos = vals_var[vals_var > 0]
            if vals_pos.size == 0:
                continue
            med_log = float(np.median(np.log10(vals_pos)))

            x_counts.append(n_tot)
            y_meds.append(med_log)

        if len(x_counts) > 0:
            figTV, axTV = plt.subplots(figsize=(5, 4))
            axTV.scatter(x_counts, y_meds, s=22, alpha=0.8, edgecolors="none")
            axTV.set_xlabel("Total number of motif sites in grammar", fontsize=11)
            axTV.set_ylabel("Median log10 patient variability (sd_Ftrend)", fontsize=11)
            axTV.set_xticks(list(range(0, 9)))
            for spine in ["top", "right"]:
                axTV.spines[spine].set_visible(False)

            out_pdf_tv = Path(str(outprefix) + "_totalSiteCount_vs_variation.pdf")
            out_png_tv = Path(str(outprefix) + "_totalSiteCount_vs_variation.png")
            figTV.savefig(out_pdf_tv, bbox_inches="tight")
            figTV.savefig(out_png_tv, dpi=300, bbox_inches="tight")
            plt.close(figTV)
            print(f"Saved total-site-count vs variation plot: {out_pdf_tv}\nSaved total-site-count vs variation plot: {out_png_tv}")


def main():
    p = argparse.ArgumentParser(description='Plot activity violins per motif grammar with 7-slot grammar glyphs (Tile 6)')
    p.add_argument('--presence', default='../results/motif_counts/tile6_site_presence_fixedbins.tsv')
    p.add_argument('--counts',   default='../results/motif_counts/tile6_site_combination_counts_fixedbins.tsv')
    p.add_argument('--baseline', default='../data/activity/OL53_run_Jurkat_berkay_activity.tsv')
    p.add_argument('--stim',     default='../data/activity/comparison_StimJurkat_vs_Jurkat_berkay.tsv')
    p.add_argument('--tnf',      default='../data/activity/comparison_TNF_vs_Ctrl.tsv')
    p.add_argument('--clades',   default='../data/clades.tsv')
    p.add_argument('--min-n', type=int, default=10)
    p.add_argument('--order-by', choices=['stim','baseline','tnf','variation'],
                  default='stim',
                  help='Order grammars by median activity (stim/baseline/tnf) or by median log10(sd_Ftrend)')
    p.add_argument('--outfig', default='../results/figures/tile6_grammar_activity')
    p.add_argument('--baseline-only', action='store_true',
                  help='Plot only baseline violins (no Stim/TNF panels).')
    p.add_argument('--variation', default=None,
                   help='TSV with per-tile variation metrics (including sd_Ftrend[_ctrl])')
    p.add_argument('--baseline-and-variation', action='store_true',
                  help='Plot baseline activity and SD_Ftrend (no Stim/TNF panels).')
    args = p.parse_args()

    variation_path = Path(args.variation) if args.variation is not None else None
    outprefix = Path(str(args.outfig) + f'_{args.order_by}Ordered')
    plot_grammars_with_activity(
        Path(args.presence),
        Path(args.counts),
        Path(args.baseline),
        Path(args.stim),
        Path(args.tnf),
        Path(args.clades),
        args.min_n,
        outprefix,
        args.order_by,
        variation_path=variation_path,
        baseline_only=args.baseline_only,
        baseline_and_variation=args.baseline_and_variation,
    )


if __name__ == '__main__':
    main()
