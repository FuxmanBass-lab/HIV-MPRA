#!/usr/bin/env python3
"""
Pie-chart matrix per tile to summarize TF motif count distributions by HIV clade.

For each requested tile, we load the motif count table (rows = isolates, columns = TF
families, values = integer counts per isolate). We map each isolate to a clade using
`../data/clades.tsv`, collapse subtypes via SUBTYPE_MAP (e.g., A1->A), and include
AE explicitly. Then we build a matrix where columns are TFs of interest (per-tile
selection), rows are clade groups present in the data, and each cell is a pie chart
showing the distribution of motif counts for that TF within that clade (slices for
0 motifs, 1 motif, 2 motifs, ... up to the global maximum observed for that tile).

One figure is produced per tile, saved as both PDF and PNG (300 dpi) under:
    ../results/figures/pies/tile_<tile>_tf_pies.(pdf|png)

Usage examples:
    python 04_piechart.py --tiles 6 9 11 13
    python code/04_piechart.py --tiles 6 --outdir ../results/figures/pies

Notes:
- The script tolerates either `../results/motif_counts/tile_<n>.counts.tsv` (plural)
  or `../results/motif_count/tile_<n>.counts.tsv` (singular) on disk.
- The `clades.tsv` file must have columns: tile_id, Clade. The isolate accession is
  parsed from `tile_id` as the suffix after the last underscore, e.g.,
  `HIV-1:REJO:250:-_OK532808.1` -> `OK532808.1`.
- Requested TF names such as "IRFx2" are normalized to match column names like
  "IRF_x2" present in the counts tables.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------------------- Config --------------------------------------- #
SUBTYPE_MAP: Dict[str, str] = {
    "A1": "A", "A6": "A", "A4": "A", "A2": "A", "A": "A",
    "01_AE": "AE",
    "B": "B", "C": "C", "D": "D",
    "F1": "F", "F2": "F", "F1F2": "F",
    "G": "G", "O": "O",
    "H": "H"
}

# TFs of interest per tile (user-specified)
TILE_TF_WHITELIST: Dict[int, List[str]] = {
    6:  ["NFKB/REL", "SP/KLF", "ETS", "USF", "ATF1"],
    13: ["IRFx2", "IRFx3", "SP/KLF", "E2F"],
    11: ["NFKB/REL", "IRFx2", "IRFx3", "SP/KLF", "E2F"],
    9:  ["NFKB/REL", "SP/KLF"],
}

# Reference max number of TF columns across tiles (controls right-side whitespace)
MAX_TF_COLS = max(len(v) for v in TILE_TF_WHITELIST.values())

# Preferred row order when present
CLADE_ORDER = ["A", "B", "C", "D", "AE", "F", "G", "H", "O"]

# --------------------------- Utilities -------------------------------------- #

def normalize_tf_name(tf: str) -> str:
    """Normalize requested TF names to match table columns.
    E.g., "IRFx2" -> "IRF_x2", "IRFx3" -> "IRF_x3".
    Other names are returned as-is.
    """
    if tf.upper().startswith("IRFX") and len(tf) >= 5 and tf[-1].isdigit():
        return tf.upper().replace("IRFX", "IRF_x")
    return tf


def find_counts_path(tile: int) -> Path:
    """Return an existing path for the tile counts file, trying plural then singular."""
    candidates = [
        Path(f"../results/motif_counts/tile_{tile}.counts.tsv"),
        Path(f"../results/motif_count/tile_{tile}.counts.tsv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Counts file not found for tile {tile}. Tried: " + ", ".join(str(c) for c in candidates)
    )


def load_counts(tile: int) -> pd.DataFrame:
    path = find_counts_path(tile)
    df = pd.read_csv(path, sep="\t")
    # Expect first column named 'isolate' (exact per provided head). If not, coerce.
    if df.columns[0].lower() != "isolate":
        df = df.rename(columns={df.columns[0]: "isolate"})
    return df


def load_clade_map() -> pd.DataFrame:
    """Load clade map and extract isolate accession as `isolate` column.
    Input columns: tile_id, Clade
    Output columns: isolate, Clade (original), Group (collapsed)
    """
    p = Path("../data/clades.tsv")
    if not p.exists():
        raise FileNotFoundError("Expected clade map at ../data/clades.tsv")
    m = pd.read_csv(p, sep="\t")
    if not {"tile_id", "Clade"}.issubset(m.columns):
        raise ValueError("clades.tsv must have columns: tile_id, Clade")

    # isolate accession is the suffix after the last underscore
    isolates = m["tile_id"].astype(str).str.split("_").str[-1]
    m = m.assign(isolate=isolates)

    # Collapse subtypes per SUBTYPE_MAP; keep AE as AE; otherwise keep original
    def group_clade(c: str) -> str:
        c = str(c).strip()
        if c == "AE":
            return "AE"
        return SUBTYPE_MAP.get(c, c)

    m["Group"] = m["Clade"].map(group_clade)
    return m[["isolate", "Clade", "Group"]]


def compute_global_max_k(df: pd.DataFrame, tf_cols: List[str]) -> int:
    """Maximum motif count observed across selected TF columns."""
    if not tf_cols:
        return 0
    return int(df[tf_cols].max().max())


def prepare_distributions(
    counts: pd.DataFrame,
    clades: pd.DataFrame,
    tf_cols: List[str],
) -> Tuple[List[str], Dict[Tuple[str, str], List[int]], List[int]]:
    """Join counts with clade groups and compute per-(clade, TF) histograms.

    Returns:
        row_labels: ordered clade groups actually present in the joined data
        hist_map: dict keyed by (clade_group, tf) -> list[int] counts for k = 0..K
        k_values: list of k values [0, 1, ..., K]
    """
    # Join on isolate
    merged = counts.merge(clades, on="isolate", how="left")

    # Drop isolates without clade mapping
    merged = merged.dropna(subset=["Group"]).copy()

    # Keep only the specified clades and drop everything else
    merged = merged[merged["Group"].isin(CLADE_ORDER)].copy()

    # Determine global max K across selected TF columns
    K = compute_global_max_k(merged, tf_cols)
    # Cap at 4, group everything >=4
    k_values = [0, 1, 2, 3, 4]

    # Determine clade row order using CLADE_ORDER but only those present
    present = (
        merged[["isolate", "Group"]].drop_duplicates()["Group"].dropna().astype(str).unique().tolist()
    )
    # Only include clades from the fixed order list, nothing else
    row_labels = [c for c in CLADE_ORDER if c in present]

    hist_map: Dict[Tuple[str, str], List[int]] = {}
    for clade in row_labels:
        sub = merged.loc[merged["Group"] == clade]
        for tf in tf_cols:
            vals = sub[tf].dropna().astype(int)
            # histogram for 0..4 with grouping >=4
            counts_k = [
                int((vals == 0).sum()),
                int((vals == 1).sum()),
                int((vals == 2).sum()),
                int((vals == 3).sum()),
                int((vals >= 4).sum()),
            ]
            hist_map[(clade, tf)] = counts_k

    return row_labels, hist_map, k_values


# ----------------------------- Plotting ------------------------------------- #

def plot_pie_matrix(
    tile: int,
    row_labels: List[str],
    tf_cols: List[str],
    hist_map: Dict[Tuple[str, str], List[int]],
    k_values: List[int],
    outdir: Path,
):
    n_rows = len(row_labels)
    n_cols = len(tf_cols)
    if n_rows == 0 or n_cols == 0:
        print(f"[tile {tile}] Nothing to plot (rows={n_rows}, cols={n_cols}). Skipping.")
        return

    grid_cols = max(n_cols, MAX_TF_COLS)
    fig_w = max(6, 1.2 * grid_cols)
    fig_h = max(4, 0.9 * n_rows)
    fig, axes = plt.subplots(n_rows, grid_cols, figsize=(fig_w, fig_h), squeeze=False)
    # Tight packing between pies
    fig.subplots_adjust(wspace=0.0, hspace=0.0)

    # Hide unused right-side axes to create whitespace
    if grid_cols > n_cols:
        for i in range(n_rows):
            for j in range(n_cols, grid_cols):
                ax = axes[i, j]
                ax.axis('off')
    

    # Manual layout with constant row spacing across tiles.
    # - Row center-to-center spacing is fixed using the full clade list length
    #   so tiles with fewer rows do not look cramped.
    # - Cell height is capped to fit within that spacing; cells remain square.
    L, R = fig.subplotpars.left, fig.subplotpars.right
    B, T = fig.subplotpars.bottom, fig.subplotpars.top
    avail_w = R - L
    avail_h = T - B

    # Constant row spacing target based on full clade order length
    BASE_ROWS = len(CLADE_ORDER)
    row_step = avail_h / max(BASE_ROWS, 1)

    # Base square cell size from limiting dimension, but do not exceed row spacing
    base_cell = min(avail_w / max(n_cols, 1), avail_h / max(n_rows, 1))
    cell_h = min(base_cell, row_step * 0.88)  # small headroom under labels
    cell_w = cell_h

    # Horizontal step (column centers). Keep slightly tighter than vertical but 
    # ensure the last column fits within the right margin; extra width goes right.
    if n_cols > 1:
        max_col_step = max(1e-9, (avail_w - cell_w) / (n_cols - 1))
    else:
        max_col_step = avail_w
    desired_col_step = 1.5 * cell_h
    col_step = min(desired_col_step, max_col_step)

    # Position axes using fixed row_step (independent of n_rows)
    for i in range(n_rows):
        for j in range(n_cols):
            x0 = L + j * col_step
            y0 = T - cell_h - i * row_step
            axes[i, j].set_position([x0, y0, cell_w, cell_h])
            axes[i, j].margins(0)



    # Colors: white→red scale by number of motifs (0=white, 4=red, >4 clamped to 4)
    cmap = plt.get_cmap("Reds")
    norm = mpl.colors.Normalize(vmin=0, vmax=4)

    def k_to_color(k: int):
        if k <= 0:
            return (0.95, 0.95, 0.95, 1.0)  # faint gray instead of pure white
        return cmap(norm(min(k, 4)))

    colors = [k_to_color(k) for k in k_values]

    for i, clade in enumerate(row_labels):
        for j, tf in enumerate(tf_cols):
            ax = axes[i, j]
            data = hist_map[(clade, tf)]
            total = sum(data)
            if total == 0:
                # Draw an empty circle to indicate no data
                ax.add_artist(plt.Circle((0, 0), 0.6, fill=False, lw=1.0))
                ax.set_aspect('equal')
                ax.set_xticks([]); ax.set_yticks([])
            else:
                # Avoid zero-only pies (matplotlib warns). Replace with tiny eps.
                frac = np.array(data, dtype=float)
                if frac.sum() == 0:
                    frac = np.array([1.0] + [0.0] * (len(data) - 1))
                wedges, _ = ax.pie(
                    frac,
                    startangle=90,
                    counterclock=False,
                    colors=colors,
                    radius=1.8,
                    wedgeprops=dict(linewidth=0.4, edgecolor="white"),
                )
                # No labels on slices for cleanliness
            if i == 0:
                ax.set_title(tf, fontsize=10, pad=16)  # add extra space under column label
            if j == 0:
                ax.set_ylabel(clade, rotation=0, ha='right', va='center', fontsize=9, labelpad=20)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])

    # Build a shared legend for k values
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=k_to_color(k), edgecolor='white', label=str(k)) for k in k_values]
    legend = fig.legend(
        handles=legend_handles,
        title="# motifs",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(k_values), 10),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    fig.suptitle(f"Tile {tile}: TF motif count distributions by HIV clade", y=1.10, x=0.42, fontsize=12)

    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / f"tile_{tile}_tf_pies.pdf"
    png_path = outdir / f"tile_{tile}_tf_pies.png"
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[tile {tile}] Saved: {pdf_path} and {png_path}")


# ------------------------------ CLI ----------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Pie-chart matrix per tile summarizing TF motif counts by clade.")
    parser.add_argument(
        "--tiles",
        type=int,
        nargs="+",
        required=True,
        help="Tile numbers to plot (e.g., 6 9 11 13)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="../results/figures/pies",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    clade_map = load_clade_map()
    outdir = Path(args.outdir)

    for tile in args.tiles:
        if tile not in TILE_TF_WHITELIST:
            print(f"[tile {tile}] No TF whitelist provided. Skipping.")
            continue
        requested_tfs = [normalize_tf_name(t) for t in TILE_TF_WHITELIST[tile]]

        counts = load_counts(tile)

        # Keep only TF columns that exist in the table
        available_cols = set(counts.columns)
        tf_cols = [c for c in requested_tfs if c in available_cols]
        missing = [c for c in requested_tfs if c not in available_cols]
        if missing:
            print(f"[tile {tile}] Warning: missing TF columns not found in counts file: {', '.join(missing)}")
        if not tf_cols:
            print(f"[tile {tile}] No requested TF columns are present. Skipping.")
            continue

        rows, hist_map, k_values = prepare_distributions(counts, clade_map, tf_cols)
        plot_pie_matrix(tile, rows, tf_cols, hist_map, k_values, outdir)


if __name__ == "__main__":
    main()
