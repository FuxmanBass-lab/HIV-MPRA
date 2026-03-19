#!/usr/bin/env python3
"""
Scatter plot: IRF_x3 motif strength (FIMO score) vs tile activity (log2FC)

Inputs
------
1) Activity TSV with columns including at least:
   - ID (tile identifier string that contains ":<tile_num>:" and isolate suffix)
   - log2FoldChange (activity metric to plot on Y)
2) Hits TSV (aligned or raw) with columns including at least:
   - tile (e.g., "tile_13")
   - isolate
   - assigned_family (e.g., "IRF_x3")
   - score (FIMO score)
   - dedup_kept (boolean-like) [optional, defaults to True if present]

Behavior
--------
- Filters the activity table to rows whose ID contains the requested tile pattern (e.g., ":13:").
- Extracts the isolate from the activity ID as the last underscore-delimited token.
- From hits, keeps rows for the requested tile and family (default: tile_13 & IRF_x3).
- Uses only deduplicated hits by default (dedup_kept == True if the column exists); pass
  --all-hits to include all.
- Aggregates per-isolate motif strength as the **max** FIMO score across hits for that isolate
  (configurable via --agg {max,mean,median}).
- Joins activity and motif strength by isolate and plots a scatter with a regression fit line,
  reporting Pearson and Spearman correlations in the title.

Usage
-----
python 08_scatter_IRF_IFNg.py \
  --activity ../data/activity/comparison_INFg_vs_Ctrl.tsv \
  --hits     ../results/motif_counts/tile_13.hits.aligned.tsv \
  --tile 13 \
  --family IRF_x3 \
  --out    ../results/figures/tile13_IRFx3_vs_IFNg

Notes
-----
- If your activity file is Baseline instead of IFNγ, just point --activity to that file.
- The script only assumes the activity file has an "ID" column and a numeric "log2FoldChange".
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def _to_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "t", "1", "yes", "y"])


def load_activity(path: Path, tile_num: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    if "ID" not in df.columns or "log2FoldChange" not in df.columns:
        raise ValueError("Activity file must contain columns: ID, log2FoldChange")
    # subset to this tile
    pat = f":{tile_num}:"
    df = df[df["ID"].astype(str).str.contains(pat, na=False)].copy()
    # extract isolate (last underscore-delimited token)
    def iso_from_id(x: str) -> str:
        xs = str(x).split("_")
        return xs[-1] if xs else str(x)
    df["isolate"] = df["ID"].astype(str).map(iso_from_id)
    df["log2FoldChange"] = pd.to_numeric(df["log2FoldChange"], errors="coerce")
    df = df.dropna(subset=["log2FoldChange", "isolate"])  # keep valid
    return df[["isolate", "log2FoldChange"]]


def load_hits(path: Path, tile_num: int, family: str, all_hits: bool) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    need = {"tile", "isolate", "assigned_family", "score"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Hits file missing required columns: {missing}")
    df = df[df["tile"].astype(str) == f"tile_{tile_num}"]
    df = df[df["assigned_family"].astype(str) == family]
    df = df[~df["assigned_family"].astype(str).str.contains("IRF_x2")]
    if not all_hits and "dedup_kept" in df.columns:
        df = df[_to_bool_series(df["dedup_kept"])].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])  # keep valid scores
    return df[["isolate", "score"]]


def aggregate_scores(hits_df: pd.DataFrame, agg: str) -> pd.DataFrame:
    if hits_df.empty:
        return pd.DataFrame(columns=["isolate", "motif_score"])  # empty
    agg = agg.lower()
    if agg == "max":
        ser = hits_df.groupby("isolate")["score"].max()
    elif agg == "mean":
        ser = hits_df.groupby("isolate")["score"].mean()
    elif agg == "median":
        ser = hits_df.groupby("isolate")["score"].median()
    else:
        raise ValueError("agg must be one of: max, mean, median")
    out = ser.reset_index().rename(columns={"score": "motif_score"})
    return out


def scatter_plot(df: pd.DataFrame, title: str, outprefix: Path):
    if df.empty:
        print("No data after join — nothing to plot.")
        return
    x = df["motif_score"].values
    y = df["log2FoldChange"].values

    # correlations
    pear_r, pear_p = (np.nan, np.nan)
    spear_r, spear_p = (np.nan, np.nan)
    if len(df) >= 3:
        pear_r, pear_p = stats.pearsonr(x, y)
        spear_r, spear_p = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(x, y, s=18, alpha=0.75, edgecolors="none")

    # simple least-squares fit line
    if len(df) >= 2 and np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xx, m*xx + b, linewidth=1.6, alpha=0.9)

    ax.set_xlabel("IRF_x3 motif strength (FIMO score)")
    ax.set_ylabel("Activity (log2FC)")
    ax.set_title(f"IRF_x3 vs Activity  |  Pearson r={pear_r:.2f} (p={pear_p:.1e}), Spearman ρ={spear_r:.2f} (p={spear_p:.1e})")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    outprefix = Path(outprefix)
    outprefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{outprefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{outprefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outprefix}.pdf\nSaved: {outprefix}.png")


def main():
    ap = argparse.ArgumentParser(description="Scatter: IRF_x3 motif strength vs activity (log2FC)")
    ap.add_argument("--activity", type=str, required=True, help="Activity TSV with ID and log2FoldChange")
    ap.add_argument("--hits",     type=str, required=True, help="Hits TSV (aligned or raw)")
    ap.add_argument("--tile",     type=int, default=13, help="Tile number to match in activity IDs and hits (default: 13)")
    ap.add_argument("--family",   type=str, default="IRF_x3", help="Motif family to measure (default: IRF_x3)")
    ap.add_argument("--agg",      type=str, default="max", choices=["max","mean","median"], help="Aggregate score per isolate")
    ap.add_argument("--all-hits", action="store_true", help="Use all hits (ignore dedup_kept if present)")
    ap.add_argument("--out",      type=str, default="../results/figures/tile13_IRFx3_vs_activity", help="Output prefix for figure")
    args = ap.parse_args()

    act = load_activity(Path(args.activity), args.tile)
    hits = load_hits(Path(args.hits), args.tile, args.family, args.all_hits)
    scores = aggregate_scores(hits, args.agg)

    df = pd.merge(act, scores, on="isolate", how="inner")
    n0 = len(df)
    print(f"Joined isolates: {n0}")

    title = f"Tile {args.tile}: {args.family} motif strength vs activity"
    scatter_plot(df, title, Path(args.out))


if __name__ == "__main__":
    main()
