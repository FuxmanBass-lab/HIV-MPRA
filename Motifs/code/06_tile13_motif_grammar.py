#!/usr/bin/env python3
"""
Count per‑isolate combinations of canonical IRF_x2, IRF_x3, E2F, and SP/KLF sites for **Tile 13**
using the **aligned** hits (query->reference lifted) and **fixed bins**.

Bins (bp on the reference axis; half‑open [start, end))
  IRF_x2 1: [10, 30)
  IRF_x2 2: [40, 55)
  E2F 1:   [40, 55)
  IRF_x2 3: [60, 80)
  IRF_x3 1: [60, 80)
  E2F 2:   [120, 135)
  SP/KLF 1:[135, 150)

Logic
-----
- Reads the aligned hits file (default: ../results/motif_counts/tile13.hits.aligned.tsv)
- Filters to tile_13 and (by default) deduplicated hits (dedup_kept==True). Use
  --all-hits to include everything.
- Assigns each hit to a site **by midpoint** ("pos") falling inside the family‑specific bins.
- For each isolate, records presence (>=1 hit) at each site for each family.
- Builds bitstrings (per family) in this order and concatenates with '|':
    IRF_x2 (3 bits) | IRF_x3 (1 bit) | E2F (2 bits) | SP/KLF (1 bit)
- Writes two TSVs:
    1) tile13_site_presence_fixedbins.tsv — per‑isolate presence matrix + per‑family bits + signature
    2) tile13_site_combination_counts_fixedbins.tsv — counts per signature

python3 06_tile13_motif_grammar.py \
  --aligned ../results/motif_counts/tile_13.hits.aligned.tsv \
  --outdir ../results/motif_grammar
    
python3 06_tile13_motif_grammar.py \
  --aligned ../results/motif_counts_patients/tile_13.hits.aligned.tsv \
  --outdir ../results/motif_grammar_patients
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Families and bins (half‑open intervals [start, end))
FAMILY_BINS: dict[str, list[tuple[float, float]]] = {
    "IRF_x2": [(10, 30), (40, 55), (60, 80)],
    "IRF_x3": [(60, 80)],
    "E2F":    [(40, 55), (120, 135)],
    "SP/KLF": [(135, 150)],
}
FAMILY_ORDER = ["IRF_x2", "IRF_x3", "E2F", "SP/KLF"]


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
    isolates = sorted(df["isolate"].unique())
    out = pd.DataFrame({"isolate": isolates})
    # initialize columns
    for i in range(len(bins)):
        out[f"{fam}_site{i+1}"] = False
    if sub.empty:
        return out
    # collect hits per isolate
    pos_by_iso = sub.groupby("isolate")["pos"].apply(list).to_dict()
    for iso, hits in pos_by_iso.items():
        for i, (lo, hi) in enumerate(bins):
            present = any(in_bin(float(p), lo, hi) for p in hits if pd.notnull(p))
            if present:
                out.loc[out["isolate"] == iso, f"{fam}_site{i+1}"] = True
    return out


def main():
    ap = argparse.ArgumentParser(description="Tile 13: count IRF_x2, IRF_x3, E2F, SP/KLF site combinations (fixed bins, aligned hits)")
    ap.add_argument("--aligned", type=str, default="../results/motif_counts/tile13.hits.aligned.tsv",
                    help="Aligned hits TSV (with reference coordinates)")
    ap.add_argument("--outdir", type=str, default="../results/motif_counts",
                    help="Where to write output TSVs")
    ap.add_argument("--all-hits", action="store_true", help="Use all hits (default: dedup_kept==True only)")
    args = ap.parse_args()

    path = Path(args.aligned)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_aligned_hits(path)
    df = df[df["tile"].astype(str) == "tile_13"].copy()
    if not args.all_hits and "dedup_kept" in df.columns:
        df = df[df["dedup_kept"].astype(str).str.lower() == "true"].copy()

    # Build presence matrices per family and merge
    merged = None
    family_bits_cols: list[str] = []
    for fam in FAMILY_ORDER:
        pres = family_presence_by_isolate(df, fam, FAMILY_BINS[fam])
        if merged is None:
            merged = pres
        else:
            merged = pd.merge(merged, pres, on="isolate", how="outer").fillna(False)
        # track bit cols for this family
        n_sites = len(FAMILY_BINS[fam])
        fam_cols = [f"{fam}_site{i+1}" for i in range(n_sites)]
        family_bits_cols.append(";".join(fam_cols))  # temporary store as string list

    # Build per‑family bitstrings and full signature in FAMILY_ORDER
    def pack_bits(row, cols):
        return ''.join('1' if bool(row[c]) else '0' for c in cols)

    per_family_bits = []
    for fam in FAMILY_ORDER:
        n_sites = len(FAMILY_BINS[fam])
        cols = [f"{fam}_site{i+1}" for i in range(n_sites)]
        bits = merged.apply(lambda r: pack_bits(r, cols), axis=1)
        merged[f"{fam}_bits"] = bits
        per_family_bits.append(f"{fam}_bits")

    merged["signature"] = merged.apply(lambda r: '|'.join(str(r[col]) for col in per_family_bits), axis=1)

    # Counts of combinations across isolates
    group_cols = [f"{fam}_bits" for fam in FAMILY_ORDER] + ["signature"]
    counts = (
        merged.groupby(group_cols).size().rename("n_isolates").reset_index().sort_values("n_isolates", ascending=False)
    )

    # Write outputs
    pres_path = outdir / "tile13_site_presence_fixedbins.tsv"
    counts_path = outdir / "tile13_site_combination_counts_fixedbins.tsv"
    merged.to_csv(pres_path, sep='\t', index=False)
    counts.to_csv(counts_path, sep='\t', index=False)

    print(f"Wrote per-isolate presence: {pres_path}")
    print(f"Wrote combination counts:  {counts_path}")


if __name__ == "__main__":
    main()
