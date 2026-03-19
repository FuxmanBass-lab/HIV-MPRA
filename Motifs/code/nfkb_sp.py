#!/usr/bin/env python3
"""
Check whether SP/KLF family overlaps with NFKB/REL family within the SAME tile
using the ORIGINAL (unaligned) hits file for Tile 6.

It runs two passes:
  1) Considering ONLY deduplicated hits (dedup_kept == True)
  2) Considering ALL hits (no dedup filter)

An "overlap" is defined on the original query coordinates as any two intervals
(start, end) with start < other_end and end > other_start. Overlaps are checked
per isolate within tile_6, then aggregated across isolates.

Usage:
    python code/nfkb_sp.py \
        --hits ../results/motif_counts/tile6.hits.tsv

Outputs a text summary to stdout including:
- Whether any overlaps were found in each pass
- Number of isolates with ≥1 overlap
- Total overlapping pairs across all isolates
- Example rows for a few overlaps
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd

FAM_SPKLF = "SP/KLF"
FAM_NFKB = "NFKB/REL"


def load_hits(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    # Basic column checks
    required = {"tile", "isolate", "start", "end", "assigned_family"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    # Coerce numeric
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    return df


def interval_overlaps(sp: List[Tuple[int, int]], nf: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Return list of overlapping (sp_interval, nf_interval) pairs using a two-pointer sweep.
    Intervals are treated as half-open [start, end).
    """
    sp_sorted = sorted(sp, key=lambda x: (x[0], x[1]))
    nf_sorted = sorted(nf, key=lambda x: (x[0], x[1]))
    i = j = 0
    overlaps = []
    while i < len(sp_sorted) and j < len(nf_sorted):
        s1, e1 = sp_sorted[i]
        s2, e2 = nf_sorted[j]
        # Check overlap
        if s1 < e2 and e1 > s2:
            overlaps.append(((s1, e1), (s2, e2)))
        # Advance the interval that ends first
        if e1 <= e2:
            i += 1
        else:
            j += 1
    return overlaps


def interval_overlaps_with_idx(sp_intervals, nf_intervals, min_frac: float = 0.5):
    """Return list of overlapping pairs with indices and overlap metrics.
    Each item in sp_intervals / nf_intervals is (start, end, idx). Intervals are
    half-open [start, end). A pair is kept only if the overlap length is at least
    `min_frac` of **both** intervals (reciprocal overlap ≥ min_frac).
    Returns a list of tuples: (sp_idx, nf_idx, ov_len, sp_len, nf_len, frac_sp, frac_nf).
    """
    sp_sorted = sorted(sp_intervals, key=lambda x: (x[0], x[1]))
    nf_sorted = sorted(nf_intervals, key=lambda x: (x[0], x[1]))
    i = j = 0
    results = []
    while i < len(sp_sorted) and j < len(nf_sorted):
        s1, e1, idx1 = sp_sorted[i]
        s2, e2, idx2 = nf_sorted[j]
        # compute overlap
        ov_start = max(s1, s2)
        ov_end = min(e1, e2)
        ov_len = max(0, ov_end - ov_start)
        if ov_len > 0:
            sp_len = max(1, e1 - s1)  # guard against zero-length
            nf_len = max(1, e2 - s2)
            frac_sp = ov_len / sp_len
            frac_nf = ov_len / nf_len
            if frac_sp >= min_frac and frac_nf >= min_frac:
                results.append((idx1, idx2, ov_len, sp_len, nf_len, frac_sp, frac_nf))
        # advance pointer that ends first
        if e1 <= e2:
            i += 1
        else:
            j += 1
    return results


def run_check(df: pd.DataFrame, dedup_only: bool, write_path: Path | None = None) -> pd.DataFrame:
    df_t6 = df[df["tile"].astype(str) == "tile_6"].copy()
    if dedup_only and "dedup_kept" in df_t6.columns:
        df_t6 = df_t6[df_t6["dedup_kept"].astype(str).str.lower() == "true"].copy()

    # Collect overlaps per isolate
    rows = []
    total_pairs = 0
    isolates_with_overlap = 0

    for iso, g in df_t6.groupby("isolate"):
        # Build lists with indices for two-pointer sweep
        sp_rows = g.loc[g["assigned_family"] == FAM_SPKLF, ["start", "end"]].dropna().astype(int)
        nf_rows = g.loc[g["assigned_family"] == FAM_NFKB, ["start", "end"]].dropna().astype(int)
        if sp_rows.empty or nf_rows.empty:
            continue
        sp_list = [(int(s), int(e), idx) for idx, (s, e) in sp_rows.iterrows()]
        nf_list = [(int(s), int(e), idx) for idx, (s, e) in nf_rows.iterrows()]

        pairs = interval_overlaps_with_idx(sp_list, nf_list, min_frac=0.5)
        if not pairs:
            continue
        isolates_with_overlap += 1
        total_pairs += len(pairs)

        # Build detailed rows for output (include overlap metrics)
        for sp_idx, nf_idx, ov_len, sp_len, nf_len, frac_sp, frac_nf in pairs:
            sp_rec = g.loc[sp_idx]
            nf_rec = g.loc[nf_idx]
            rows.append({
                "tile": sp_rec.get("tile", "tile_6"),
                "isolate": iso,
                "dedup_kept_sp": sp_rec.get("dedup_kept", None),
                "dedup_kept_nf": nf_rec.get("dedup_kept", None),
                "sp_start": int(sp_rec["start"]),
                "sp_end": int(sp_rec["end"]),
                "sp_motif": sp_rec.get("motif", None),
                "sp_assigned_family": sp_rec.get("assigned_family", None),
                "nf_start": int(nf_rec["start"]),
                "nf_end": int(nf_rec["end"]),
                "nf_motif": nf_rec.get("motif", None),
                "nf_assigned_family": nf_rec.get("assigned_family", None),
                "overlap_bp": int(ov_len),
                "sp_len_bp": int(sp_len),
                "nf_len_bp": int(nf_len),
                "overlap_frac_sp": round(frac_sp, 3),
                "overlap_frac_nf": round(frac_nf, 3),
            })

    title = "[DE-DUPLICATED ONLY]" if dedup_only else "[ALL HITS]"
    print("\n" + "=" * 80)
    print(f"Overlap check {title}")
    print("=" * 80)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("No overlaps detected between SP/KLF and NFKB/REL within tile_6.")
    else:
        print(f"Isolates with ≥1 overlap: {isolates_with_overlap}")
        print(f"Total overlapping pairs:  {total_pairs}")
        print(f"Overlaps dataframe rows:  {len(out_df)}")
        print("(Pairs require ≥50% reciprocal overlap in both intervals)")
        # Show a few examples
        print(out_df.head(10).to_string(index=False))

    # Optional write
    if write_path is not None:
        if out_df.empty:
            # still write an empty file with headers for reproducibility
            out_df.to_csv(write_path, sep='\t', index=False)
        else:
            out_df.to_csv(write_path, sep='\t', index=False)
        print(f"Saved all overlaps to: {write_path}")

    return out_df


def main():
    ap = argparse.ArgumentParser(description="Check SP/KLF vs NFKB/REL overlaps on original Tile 6 hits")
    ap.add_argument("--hits", type=str, default="../results/motif_counts/tile6.hits.tsv",
                    help="Path to ORIGINAL (unaligned) tile6 hits TSV")
    args = ap.parse_args()

    path = Path(args.hits)
    df = load_hits(path)

    out_dir = path.parent
    out1 = out_dir / "tile6_overlaps_dedup.tsv"
    out2 = out_dir / "tile6_overlaps_all.tsv"

    # Pass 1: deduplicated only
    run_check(df, dedup_only=True, write_path=out1)
    # Pass 2: all hits
    run_check(df, dedup_only=False, write_path=out2)


if __name__ == "__main__":
    main()
