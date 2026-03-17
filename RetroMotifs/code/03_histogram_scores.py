#!/usr/bin/env python3
"""
Histogram per TF (motif/gene/family) from a TSV file.

Example:
  python 03_histogram_scores.py \
    --tsv ../results/motif_counts/tile_6.hits.tsv \
    --outdir ../figures/hist_by_tf \
    --value-column score \
    --group-by gene \
    --bins 30 \
    --min-n 5

Input file must be a tab-separated table with columns like:
  tile,isolate,start,end,strand,motif,motif_len,rel_score,score,kseq,gene,p_value,q_value,cluster_id,dedup_kept,dedup_reason,families_here,assigned_family

Typical TF identifiers:
  - motif (e.g., "IRF3")
  - gene  (e.g., "IRF3")
  - assigned_family (e.g., "IRF_x2")

Outputs one histogram per group to OUTDIR, named:
  <group>__<sanitized_name>__<value-column>_hist.<ext>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------

def sanitize_filename(name: str) -> str:
    # Keep alnum, dash, underscore; replace others with '_'
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._") or "NA"


def auto_bins(series: pd.Series) -> int:
    # Fallback if user passes bins==0; use Freedman–Diaconis rule
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 10
    iqr = s.quantile(0.75) - s.quantile(0.25)
    if iqr == 0:
        return 10
    bin_width = 2 * iqr * (len(s) ** (-1 / 3))
    if bin_width <= 0:
        return 10
    bins = max(5, int((s.max() - s.min()) / bin_width))
    return bins


# -----------------------------
# Core plotting
# -----------------------------

def plot_histograms(
    df: pd.DataFrame,
    outdir: Path,
    group_by: str,
    value_col: str,
    bins: int,
    min_n: int,
    ext: str,
    one_pdf: bool,
    tight: bool,
    dpi: int,
    logx: bool,
    density: bool,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure numeric
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col, group_by])

    # Group and plot
    groups = df.groupby(group_by, dropna=True)
    plotted = 0
    skipped = 0

    pdf_ctx: Optional[PdfPages] = None
    if one_pdf:
        pdf_path = outdir / f"hist_by_{group_by}__{value_col}.pdf"
        pdf_ctx = PdfPages(pdf_path)

    for name, g in groups:
        n = len(g)
        if n < min_n:
            skipped += 1
            continue

        vals = g[value_col].astype(float)
        nbins = auto_bins(vals) if bins == 0 else bins

        fig = plt.figure(figsize=(6, 4))
        plt.hist(vals, bins=nbins, edgecolor="black", linewidth=0.5, density=density)
        title = f"{group_by} = {name} (n={n})\n{value_col} distribution"
        plt.title(title)
        plt.xlabel(value_col)
        plt.ylabel("Density" if density else "Count")
        if logx:
            plt.xscale("log")
        if tight:
            plt.tight_layout()

        if pdf_ctx is not None:
            pdf_ctx.savefig(fig, dpi=dpi)
            plt.close(fig)
        else:
            fname = f"{group_by}__{sanitize_filename(name)}__{value_col}_hist.{ext}"
            out_path = outdir / fname
            fig.savefig(out_path, dpi=dpi)
            plt.close(fig)
        plotted += 1

    if pdf_ctx is not None:
        pdf_ctx.close()

    print(f"Plotted {plotted} histograms. Skipped {skipped} groups with n < {min_n}.")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make per-TF histograms from motif table (TSV)")
    p.add_argument("--tsv", required=True, help="Input TSV file path")
    p.add_argument("--outdir", required=True, help="Output directory for histograms")
    p.add_argument(
        "--group-by",
        default="gene",
        choices=["motif", "gene", "assigned_family", "families_here"],
        help="Column to group by (TF identifier)",
    )
    p.add_argument(
        "--value-column",
        default="rel_score",
        help="Numeric column to histogram (e.g., rel_score, score, motif_len)",
    )
    p.add_argument("--bins", type=int, default=30, help="Number of histogram bins; 0 = auto")
    p.add_argument("--min-n", type=int, default=5, help="Minimum rows per group to plot")
    p.add_argument("--ext", default="png", choices=["png", "pdf", "svg"], help="Image format")
    p.add_argument("--one-pdf", action="store_true", help="Write all plots into a single PDF")
    p.add_argument("--tight", action="store_true", help="Use tight_layout for figures")
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI for raster outputs")
    p.add_argument("--logx", action="store_true", help="Log-scale the X axis")
    p.add_argument("--density", action="store_true", help="Normalize histogram to density")
    p.add_argument(
        "--filter",
        default=None,
        help=(
            "Optional pandas query string to prefilter rows, e.g. "
            "`dedup_kept == True and q_value < 0.05`"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tsv_path = Path(args.tsv)
    outdir = Path(args.outdir)

    if not tsv_path.exists():
        print(f"ERROR: TSV not found: {tsv_path}", file=sys.stderr)
        sys.exit(2)

    # Read TSV
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)

    # Optional filter
    if args.filter:
        try:
            # Convert some common numeric/bool columns to proper dtypes for filtering
            for c in (args.value_column, "q_value", "p_value", "motif_len", "score", "rel_score"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="ignore")
            if "dedup_kept" in df.columns:
                df["dedup_kept"] = df["dedup_kept"].map({"True": True, "False": False}).fillna(df["dedup_kept"])
            df = df.query(args.filter)
        except Exception as e:
            print(f"WARNING: Failed to apply --filter due to: {e}", file=sys.stderr)

    # Plot
    plot_histograms(
        df=df,
        outdir=outdir,
        group_by=args.group_by,
        value_col=args.value_column,
        bins=args.bins,
        min_n=args.min_n,
        ext=args.ext,
        one_pdf=args.one_pdf,
        tight=args.tight,
        dpi=args.dpi,
        logx=args.logx,
        density=args.density,
    )


if __name__ == "__main__":
    main()