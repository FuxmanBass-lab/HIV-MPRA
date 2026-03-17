#!/usr/bin/env python3
"""
Count per‑isolate combinations of canonical NFKB/REL and SP/KLF sites for Tile 6
using the **aligned** hits (query->reference lifted) and **fixed bins**.

Bins (bp on the reference axis):
  NFKB 1: [75, 95)
  NFKB 2: [95, 110)
  NFKB 3: [110, 120)
  NFKB 4  (same region as SP 1): [120, 135)
  SP   1  (same region as NFKB 4): [120, 135)
  SP   2: [135, 145)
  SP   3: [145, 155)
  SP   4: [155, 170)

Logic
-----
- Reads the aligned hits file (default: ../results/motif_counts/tile6.hits.aligned.tsv)
- Filters to tile_6 and (by default) deduplicated hits (dedup_kept==True). Use
  --all-hits to include everything.
- Assigns each hit to a site **by midpoint** ("pos") falling inside a bin.
  (If desired, we can switch to span-overlap assignment later.)
- For each isolate, records presence (>=1 hit) at each of the 4 NFKB sites and
  4 SP sites.
- Builds bitstrings NFKB[1..4] + SP[1..4] and counts the number of isolates per
  combination.
- Writes two TSVs:
    1) tile6_site_presence_fixedbins.tsv — per isolate presence matrix + bits
    2) tile6_site_combination_counts_fixedbins.tsv — counts per combination

python3 06_tile6_motif_grammar.py \
  --aligned ../results/motif_counts/tile_6.hits.aligned.tsv \
  --outdir ../results/motif_grammar

python3 06_tile6_motif_grammar.py \
  --aligned ../results/motif_counts_patients/tile_6.hits.aligned.tsv \
  --outdir ../results/motif_grammar_patients
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

FAM_NFKB = "NFKB/REL"
FAM_SPKLF = "SP/KLF"

# Fixed bins (half-open intervals [start, end))
NFKB_BINS = [(75, 95), (95, 110), (110, 120), (120, 135)]
SP_BINS   = [(120, 135), (135, 145), (145, 155), (155, 170)]


def load_aligned_hits(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    need = {"tile", "isolate", "pos", "assigned_family"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
    return df


def in_bin(x: float, lo: float, hi: float) -> bool:
    return (x >= lo) and (x < hi)


def family_presence_by_isolate(df: pd.DataFrame, fam: str, bins: list[tuple[float, float]]) -> pd.DataFrame:
    sub = df[df["assigned_family"] == fam].copy()
    if sub.empty:
        # no rows for this family — return all False for any isolates in df
        isolates = sorted(df["isolate"].unique())
        out = pd.DataFrame({"isolate": isolates})
        for i in range(4):
            out[f"{fam}_site{i+1}"] = False
        return out
    # compute presence by midpoint membership
    pres = (
        sub.groupby("isolate")["pos"].apply(list).reset_index()
    )
    # initialize
    for i in range(4):
        pres[f"{fam}_site{i+1}"] = False
    # fill
    for idx, row in pres.iterrows():
        hits = row["pos"]
        for i, (lo, hi) in enumerate(bins):
            if any(in_bin(float(p), lo, hi) for p in hits if pd.notnull(p)):
                pres.at[idx, f"{fam}_site{i+1}"] = True
    # keep columns
    keep = ["isolate"] + [f"{fam}_site{i+1}" for i in range(4)]
    return pres[keep]


def main():
    ap = argparse.ArgumentParser(description="Tile 6: count NFKB/REL and SP/KLF site combinations (fixed bins, aligned hits)")
    ap.add_argument("--aligned", type=str, default="../results/motif_counts/tile6.hits.aligned.tsv",
                    help="Aligned hits TSV (with reference coordinates)")
    ap.add_argument("--outdir", type=str, default="../results/motif_counts",
                    help="Where to write output TSVs")
    ap.add_argument("--all-hits", action="store_true", help="Use all hits (default: dedup_kept==True only)")
    args = ap.parse_args()

    path = Path(args.aligned)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_aligned_hits(path)
    df = df[df["tile"].astype(str) == "tile_6"].copy()
    if not args.all_hits and "dedup_kept" in df.columns:
        df = df[df["dedup_kept"].astype(str).str.lower() == "true"].copy()

    # Presence matrices per family
    pres_nfkb = family_presence_by_isolate(df, FAM_NFKB, NFKB_BINS)
    pres_sp   = family_presence_by_isolate(df, FAM_SPKLF, SP_BINS)

    # Merge (outer) so isolates present in only one family are kept
    pres = pd.merge(pres_nfkb, pres_sp, on="isolate", how="outer").fillna(False)

    nf_cols = [f"{FAM_NFKB}_site{i}" for i in range(1, 5)]
    sp_cols = [f"{FAM_SPKLF}_site{i}" for i in range(1, 5)]

    def pack_bits(row, cols):
        return ''.join('1' if bool(row[c]) else '0' for c in cols)

    pres["nfkb_bits"] = pres.apply(lambda r: pack_bits(r, nf_cols), axis=1)
    pres["sp_bits"] = pres.apply(lambda r: pack_bits(r, sp_cols), axis=1)
    pres["signature"] = pres["nfkb_bits"] + "|" + pres["sp_bits"]

    # Counts of combinations across isolates
    counts = (
        pres.groupby(["nfkb_bits", "sp_bits", "signature"])\
            .size().rename("n_isolates").reset_index()\
            .sort_values("n_isolates", ascending=False)
    )

    # Write outputs
    pres_path = outdir / "tile6_site_presence_fixedbins.tsv"
    counts_path = outdir / "tile6_site_combination_counts_fixedbins.tsv"
    pres.to_csv(pres_path, sep='\t', index=False)
    counts.to_csv(counts_path, sep='\t', index=False)

    print(f"Wrote per-isolate presence: {pres_path}")
    print(f"Wrote combination counts:  {counts_path}")


if __name__ == "__main__":
    main()
