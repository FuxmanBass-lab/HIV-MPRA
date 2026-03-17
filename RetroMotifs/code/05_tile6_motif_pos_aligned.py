#!/usr/bin/env python3
"""
Plot position distributions of NFKB/REL and SP/KLF motif hits for Tile 6
on a 0–200 bp axis using the deduplicated hits table.

Input (default): ../results/motif_counts/tile6.hits.tsv

Behavior:
- Filters to `tile_6` rows only.
- Uses only deduplicated sites (`dedup_kept == True`).
- Computes motif midpoint = (start + end) / 2 in bp coordinates.
- Two panels (stacked):
    1) NFKB/REL position distribution
    2) SP/KLF position distribution
- Each panel shows a normalized histogram (density=True) over 0..200 with
  thin binning and a smoothed line (simple moving average on the histogram).
- Saves PDF and 300-dpi PNG.

Usage:
    python 05_tile6_motif_pos_aligned.py \
        --hits ../results/motif_counts/tile_6.hits.tsv \
        --outdir ../results/figures/tile_6_positions \
        --bin 2

    python 05_tile6_motif_pos_aligned.py \
        --hits ../results/motif_counts_patients/tile_6.hits.tsv \
        --outdir ../results/figures_patients/tile6_positions \
        --bin 2

Notes:
- If you prefer raw counts instead of density, pass `--counts`.
- If you want to include *all* hits (not only deduplicated), pass `--all-hits`.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Families we care about
FAM_NFKB = "NFKB/REL"
FAM_SPKLF = "SP/KLF"

# Fixed axis for tile length
AX_MIN = 0.0  # AX_MAX will be determined from the reference alignment length


def load_hits(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    # Normalize expected columns
    expected = {"tile", "isolate", "start", "end", "assigned_family", "dedup_kept"}
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    # Ensure types
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    # midpoint
    df["pos"] = (df["start"].astype(float) + df["end"].astype(float)) / 2.0
    return df


def filter_tile6(df: pd.DataFrame, keep_all_hits: bool = False) -> pd.DataFrame:
    d = df[df["tile"].astype(str) == "tile_6"].copy()
    if not keep_all_hits and "dedup_kept" in d.columns:
        # Keep only deduplicated sites (True). Handle values like 'True'/'False' strings too.
        d = d[d["dedup_kept"].astype(str).str.lower() == "true"].copy()
    return d


def family_positions(d: pd.DataFrame, fam: str) -> np.ndarray:
    return d.loc[d["assigned_family"] == fam, "pos"].dropna().to_numpy()


def family_ranges(d: pd.DataFrame, fam: str, ax_max: float) -> pd.DataFrame:
    sub = d.loc[d["assigned_family"] == fam, ["start", "end"]].dropna().copy()
    # clamp to axis bounds (dynamic)
    sub["start"] = sub["start"].clip(AX_MIN, ax_max)
    sub["end"] = sub["end"].clip(AX_MIN, ax_max)
    # ensure start <= end
    sub[["start", "end"]] = np.sort(sub[["start", "end"]].values, axis=1)
    return sub


def moving_average(y: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return y
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / float(window)
    return np.convolve(ypad, kernel, mode="valid")


def needleman_wunsch(a: str, b: str, match: int = 2, mismatch: int = -1, gap: int = -2) -> tuple[str, str]:
    """Simple global alignment (Needleman-Wunsch) returning aligned strings.
    Suitable for ~200–400 bp tiles.
    """
    n, m = len(a), len(b)
    # score and traceback matrices
    score = np.zeros((n + 1, m + 1), dtype=int)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0=diag, 1=up, 2=left
    for i in range(1, n + 1):
        score[i, 0] = i * gap
        trace[i, 0] = 1
    for j in range(1, m + 1):
        score[0, j] = j * gap
        trace[0, j] = 2
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            bj = b[j - 1]
            s_diag = score[i - 1, j - 1] + (match if ai == bj else mismatch)
            s_up = score[i - 1, j] + gap
            s_left = score[i, j - 1] + gap
            best = s_diag
            t = 0
            if s_up > best:
                best, t = s_up, 1
            if s_left > best:
                best, t = s_left, 2
            score[i, j] = best
            trace[i, j] = t
    # Traceback
    i, j = n, m
    a_aln, b_aln = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i, j] == 0:
            a_aln.append(a[i - 1]); b_aln.append(b[j - 1]); i -= 1; j -= 1
        elif i > 0 and (j == 0 or trace[i, j] == 1):
            a_aln.append(a[i - 1]); b_aln.append('-'); i -= 1
        else:
            a_aln.append('-'); b_aln.append(b[j - 1]); j -= 1
    return ''.join(reversed(a_aln)), ''.join(reversed(b_aln))


def build_q2r_map(ref_aln: str, qry_aln: str) -> tuple[list[int | None], int]:
    """Return mapping from query ungapped index -> reference ungapped index (or None).
    Also return the ungapped reference length.
    """
    q2r: list[int | None] = []
    q_idx = r_idx = 0
    for ra, qa in zip(ref_aln, qry_aln):
        ref_gap = (ra == '-')
        qry_gap = (qa == '-')
        if not qry_gap:
            if not ref_gap:
                q2r.append(r_idx)
            else:
                q2r.append(None)
            q_idx += 1
        if not ref_gap:
            r_idx += 1
    return q2r, r_idx


def parse_tile_number(tile_id: str) -> int | None:
    m = re.search(r":(\d+):", str(tile_id))
    return int(m.group(1)) if m else None


def extract_isolate(tile_id: str) -> str:
    return str(tile_id).split('_')[-1]


def load_ref_seqs_for_tile(path: Path, tile_num: int) -> pd.DataFrame:
    ref = pd.read_csv(path, sep='\t')
    required = {"tile_id", "Target_sequence", "Query_sequence"}
    if not required.issubset(ref.columns):
        raise ValueError(f"{path} must have columns: {sorted(required)}")
    ref = ref.assign(tile_num=ref["tile_id"].map(parse_tile_number),
                     isolate=ref["tile_id"].map(extract_isolate))
    ref = ref[ref["tile_num"] == tile_num].copy()
    return ref[["isolate", "Target_sequence", "Query_sequence"]]


def map_hits_to_reference(d_hits: pd.DataFrame, ref_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Map hit start/end (query coords) to reference coords via per-isolate NW alignment.
    Returns (updated_df, ref_axis_max).
    - If a motif spans only query insertions (no ref mapping), that row is dropped.
    - Original start/end/pos are preserved as start_orig/end_orig/pos_orig.
    """
    if d_hits.empty:
        return d_hits.assign(start_orig=d_hits.get('start'), end_orig=d_hits.get('end'), pos_orig=d_hits.get('pos')), 0

    # Build per-isolate mappings
    maps: dict[str, list[int | None]] = {}
    ref_len_by_iso: dict[str, int] = {}
    for iso, grp in d_hits.groupby('isolate'):
        r = ref_df.loc[ref_df['isolate'] == iso]
        if r.empty:
            continue
        ref_seq = str(r['Target_sequence'].iloc[0])
        qry_seq = str(r['Query_sequence'].iloc[0])
        ref_aln, qry_aln = needleman_wunsch(ref_seq, qry_seq)
        q2r, ref_len = build_q2r_map(ref_aln, qry_aln)
        maps[iso] = q2r
        ref_len_by_iso[iso] = ref_len

    keep_rows = []
    new_start = []
    new_end = []
    for idx, row in d_hits.iterrows():
        iso = row['isolate']
        q2r = maps.get(iso)
        if q2r is None:
            continue  # skip rows without mapping
        qs = int(row['start']); qe = int(row['end'])
        # end assumed exclusive; map each query position to ref index
        mapped = [q2r[k] for k in range(qs, min(qe, len(q2r))) if q2r[k] is not None]
        if not mapped:
            continue
        rs = int(min(mapped)); re_ = int(max(mapped)) + 1  # exclusive end
        new_start.append(rs); new_end.append(re_); keep_rows.append(idx)

    if not keep_rows:
        # No mappable rows; return empty
        return d_hits.iloc[0:0].copy(), max(ref_len_by_iso.values()) if ref_len_by_iso else 0

    out = d_hits.loc[keep_rows].copy()
    out['start_orig'] = out['start']
    out['end_orig'] = out['end']
    out['pos_orig'] = out.get('pos', np.nan)
    out['start'] = new_start
    out['end'] = new_end
    out['pos'] = (out['start'].astype(float) + out['end'].astype(float)) / 2.0
    out['is_aligned'] = True

    ref_axis_max = max(ref_len_by_iso.values()) if ref_len_by_iso else 0
    return out, ref_axis_max


def plot_two_panels(
    nfkb_pos: np.ndarray,
    spklf_pos: np.ndarray,
    bin_width: float,
    use_density: bool,
    outdir: Path,
    ax_min: float,
    ax_max: float,
):
    outdir.mkdir(parents=True, exist_ok=True)

    # Build common bins on 0..200
    bins = np.arange(ax_min, ax_max + bin_width, bin_width)

    fig_h = 5.2
    fig_w = 10
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)

    def _panel(ax, data: np.ndarray, title: str, color: str):
        if data.size == 0:
            ax.text(0.5, 0.5, "No hits", ha="center", va="center")
            ax.set_xlim(ax_min, ax_max)
            ax.set_ylim(0, 1)
            ax.set_title(title)
            ax.grid(False)
            return
        # Histogram
        h, edges, _ = ax.hist(
            data,
            bins=bins,
            density=use_density,
            histtype="stepfilled",
            alpha=0.35,
            edgecolor=color,
            linewidth=0.8,
            facecolor=color,
            label="hist",
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        # Annotate more bins on the x-axis for clarity
        step = max(10, int(bin_width))  # every 10 bp or at least one bin width
        ax.set_xticks(np.arange(ax_min, ax_max + 1, step))
        # Smoothed line over the histogram for visual story
        h_smooth = moving_average(h, window=max(3, int(5 * (2.0 / bin_width))))  # adapt window a bit
        ax.plot(centers, h_smooth, color=color, linewidth=2.0)
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylabel("Density" if use_density else "Count")
        ax.set_title(title)
        # Light grid to guide the eye
        ax.grid(True, linewidth=0.2, alpha=0.4)

    _panel(
        axes[0],
        nfkb_pos,
        f"Tile 6 — {FAM_NFKB} positions (n={nfkb_pos.size})",
        color="#b22222",  # firebrick-ish
    )
    _panel(
        axes[1],
        spklf_pos,
        f"Tile 6 — {FAM_SPKLF} positions (n={spklf_pos.size})",
        color="#4444aa",  # cool blue for contrast
    )

    axes[1].set_xlabel("Position on tile (bp)")

    fig.tight_layout()

    pdf = outdir / "tile6_nfkb_sp_positions.pdf"
    png = outdir / "tile6_nfkb_sp_positions.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf}\nSaved: {png}")


def plot_range_coverage(
    nfkb_ranges: pd.DataFrame,
    spklf_ranges: pd.DataFrame,
    outdir: Path,
    ax_min: float,
    ax_max: float,
):
    outdir.mkdir(parents=True, exist_ok=True)
    # Build 0..200 integer bp grid
    x = np.arange(int(ax_min), int(ax_max) + 1)

    def coverage_from_ranges(ranges: pd.DataFrame) -> np.ndarray:
        cov = np.zeros_like(x, dtype=float)
        for _, row in ranges.iterrows():
            s = int(np.floor(row["start"]))
            e = int(np.ceil(row["end"]))
            s = max(int(ax_min), s)
            e = min(int(ax_max), e)
            if e >= s:
                cov[s:e+1] += 1.0
        return cov

    cov_nfkb = coverage_from_ranges(nfkb_ranges)
    cov_sp   = coverage_from_ranges(spklf_ranges)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)

    def _panel(ax, xvals, cov, title, color):
        ax.fill_between(xvals, cov, step="mid", alpha=0.35, color=color)
        ax.plot(xvals, cov, linewidth=1.8, color=color)
        ax.set_xlim(ax_min, ax_max)
        # Add more ticks along the 0–200 axis for clarity
        step = 10  # every 10 bp
        ax.set_xticks(np.arange(ax_min, ax_max + 1, step))
        ax.set_ylabel("Coverage (bp)")
        ax.set_title(title)
        ax.grid(True, linewidth=0.2, alpha=0.4)

    _panel(axes[0], x, cov_nfkb, f"Tile 6 — {FAM_NFKB} motif range coverage", color="#b22222")
    _panel(axes[1], x, cov_sp,   f"Tile 6 — {FAM_SPKLF} motif range coverage", color="#4444aa")

    axes[1].set_xlabel("Position on tile (bp)")
    fig.tight_layout()

    pdf = outdir / "tile6_nfkb_sp_coverage.pdf"
    png = outdir / "tile6_nfkb_sp_coverage.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf}\nSaved: {png}")




def main():
    ap = argparse.ArgumentParser(description="Plot NFKB/REL and SP/KLF position distributions for tile 6")
    ap.add_argument("--hits", type=str, default="../results/motif_counts/tile6.hits.tsv",
                    help="Path to tile6 hits tsv")
    ap.add_argument("--outdir", type=str, default="../results/figures/tile6_positions",
                    help="Output directory for figures")
    ap.add_argument("--bin", type=float, default=2.0, help="Histogram bin width in bp")
    ap.add_argument("--counts", action="store_true", help="Plot counts instead of density")
    ap.add_argument("--all-hits", action="store_true", help="Include non-deduplicated hits as well")
    args = ap.parse_args()

    hits_path = Path(args.hits)
    outdir = Path(args.outdir)
    density = not args.counts

    df = load_hits(hits_path)
    df = filter_tile6(df, keep_all_hits=args.all_hits)

    # Load per-isolate reference/query sequences for tile 6 and align
    ref_path = Path("../data/ref_seqs_all.tsv")
    ref_df = load_ref_seqs_for_tile(ref_path, tile_num=6)
    df_aln, axis_max = map_hits_to_reference(df, ref_df)

    if df_aln.empty or axis_max <= 0:
        print("No mappable hits after alignment or invalid reference length. Exiting.")
        return


    # Write aligned hits (preserving original columns with *_orig)
    aligned_out = hits_path.with_name(hits_path.stem + ".aligned.tsv")
    df_aln.to_csv(aligned_out, sep='\t', index=False)
    print(f"Aligned hits written to: {aligned_out}")

    # Position histograms (centers)
    nfkb = family_positions(df_aln, FAM_NFKB)
    spklf = family_positions(df_aln, FAM_SPKLF)
    plot_two_panels(nfkb, spklf, args.bin, density, outdir, AX_MIN, float(axis_max))

    # Range-based coverage plot
    nfkb_rng = family_ranges(df_aln, FAM_NFKB, float(axis_max))
    spklf_rng = family_ranges(df_aln, FAM_SPKLF, float(axis_max))
    plot_range_coverage(nfkb_rng, spklf_rng, outdir, AX_MIN, float(axis_max))


if __name__ == "__main__":
    main()
