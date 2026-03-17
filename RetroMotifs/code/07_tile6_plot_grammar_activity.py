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

python 07_tile6_plot_grammar_activity.py \
  --presence ../results/motif_grammar/tile6_site_presence_fixedbins.tsv \
  --counts   ../results/motif_grammar/tile6_site_combination_counts_fixedbins.tsv \
  --baseline ../data/activity/OL53_T_primaryT_activity.tsv \
  --min-n 3 \
  --order-by baseline \
  --baseline-only \
  --outfig ../results/figures/tile6_grammar_activity_CD4_3iso
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
    baseline_only: bool = False,
):
    pres, cnts, order = load_presence_counts(presence_path, counts_path, min_n)
    act_base = load_activity(base_path)
    if not baseline_only:
        act_stim = load_activity(stim_path)
        act_tnf = load_activity(tnf_path)
    else:
        act_stim = None
        act_tnf = None

    iso2clade = load_isolate_to_clade_map(clades_path)

    # Collect activity rows using initial order
    rows_base_o = collect_activity_by_signature(pres, order, act_base)
    if not baseline_only:
        rows_stim_o = collect_activity_by_signature(pres, order, act_stim)
        rows_tnf_o  = collect_activity_by_signature(pres, order, act_tnf)
    else:
        rows_stim_o = []
        rows_tnf_o  = []

    # Choose which condition determines row ordering (by median, desc)
    order_src = order_by.lower().strip()
    if baseline_only:
        order_src = "baseline"
    if order_src not in {"stim", "baseline", "tnf"}:
        order_src = "stim"
    rows_src_map = {"stim": rows_stim_o, "baseline": rows_base_o, "tnf": rows_tnf_o}
    rows_src = rows_src_map[order_src]
    med_list = []
    for sig, _iso, vals in rows_src:
        median_val = float(np.median(vals)) if vals.size > 0 else np.nan
        med_list.append((sig, median_val))
    ordered_sigs = [sig for sig, _ in sorted(med_list, key=lambda x: (np.isnan(x[1]), -x[1] if not np.isnan(x[1]) else 0))]

    # Build rows in the chosen order
    rows_base = collect_activity_by_signature(pres, ordered_sigs, act_base)
    if not baseline_only:
        rows_stim = collect_activity_by_signature(pres, ordered_sigs, act_stim)
        rows_tnf  = collect_activity_by_signature(pres, ordered_sigs, act_tnf)
    else:
        rows_stim = rows_base  # reuse for layout (same signatures/order)
        rows_tnf  = []

    # Use rows_stim (or rows_base in baseline-only mode) to count grammars
    grammar_rows = rows_stim if not baseline_only else rows_base
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
    else:
        # 5-column layout: pies, grammars, baseline, Stim, TNF
        fig = plt.figure(figsize=(18, max(3.5, 0.6 * n)))
        gs = fig.add_gridspec(nrows=1, ncols=5, width_ratios=[0.95, 0.95, 3.0, 3.0, 3.0])
        axP = fig.add_subplot(gs[0, 0])  # pies axis (leftmost)
        axG = fig.add_subplot(gs[0, 1])  # grammar glyphs + labels
        axB = fig.add_subplot(gs[0, 2])
        axS = fig.add_subplot(gs[0, 3])
        axT = fig.add_subplot(gs[0, 4])
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


def main():
    p = argparse.ArgumentParser(description='Plot activity violins per motif grammar with 7-slot grammar glyphs (Tile 6)')
    p.add_argument('--presence', default='../results/motif_counts/tile6_site_presence_fixedbins.tsv')
    p.add_argument('--counts',   default='../results/motif_counts/tile6_site_combination_counts_fixedbins.tsv')
    p.add_argument('--baseline', default='../data/activity/OL53_run_Jurkat_berkay_activity.tsv')
    p.add_argument('--stim',     default='../data/activity/comparison_StimJurkat_vs_Jurkat_berkay.tsv')
    p.add_argument('--tnf',      default='../data/activity/comparison_TNF_vs_Ctrl.tsv')
    p.add_argument('--clades',   default='../data/clades.tsv')
    p.add_argument('--min-n', type=int, default=10)
    p.add_argument('--order-by', choices=['stim','baseline','tnf'], default='stim', help='Order grammars by median activity of this condition')
    p.add_argument('--outfig', default='../results/figures/tile6_grammar_activity')
    p.add_argument('--baseline-only', action='store_true',
                  help='Plot only baseline violins (no Stim/TNF panels).')
    args = p.parse_args()

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
        baseline_only=args.baseline_only,
    )


if __name__ == '__main__':
    main()
