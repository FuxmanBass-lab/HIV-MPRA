#!/usr/bin/env python3
"""
Generate FASTA files for HIV-1 isolates based on tiled sequences.

Inputs
------
- TSV at data/sequences.tsv with columns:
  family, strain, genome, tile_id, tile_sequence, tile_type

What it does
------------
- Writes one FASTA **per tile id** (e.g., tile_6.fasta, tile_7.fasta, ...),
  where each entry is that tile sequence for a different isolate.

Notes
-----
- No genome downloads are performed. This script only writes per-tile FASTAs
  directly from the sequences.tsv file.

Usage
-----
python code/01_generate_fastas.py

"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict, namedtuple
from pathlib import Path

Tile = namedtuple("Tile", ["tile_num", "strand", "sequence", "tile_id"])  # strand from tile_id (e.g., '+')

# -------------------------
# Utility helpers
# -------------------------

def log(msg: str):
    sys.stderr.write(f"[generate_fastas] {msg}\n")


def parse_tile_id(tile_id: str):
    """Parse a tile_id like 'HIV_1:REJO:11:+_AF004885.1' -> (tile_num=11, strand='+')."""
    # Capture :<num>:<strand> before the underscore
    m = re.search(r":(\d+):([+-])(?:_|$)", tile_id)
    if not m:
        raise ValueError(f"Could not parse tile_id: {tile_id}")
    return int(m.group(1)), m.group(2)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_sequences_tsv(tsv_path: Path):
    """Read TSV and return structures:
    - per_isolate: {accession: {tile_num: Tile}}
    - per_tile: {tile_num: [(accession, sequence)]}
    """
    per_isolate: dict[str, dict[int, Tile]] = defaultdict(dict)
    per_tile: dict[int, list[tuple[str, str]]] = defaultdict(list)

    with tsv_path.open() as f:
        reader = csv.DictReader(f, delimiter='\t')
        required = {"genome", "tile_id", "tile_sequence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        for row in reader:
            accession = row["genome"].strip()
            tile_id = row["tile_id"].strip()
            seq = row["tile_sequence"].strip().upper()
            if not accession or not tile_id or not seq:
                continue
            try:
                tile_num, strand = parse_tile_id(tile_id)
            except ValueError:
                # Skip rows we cannot parse
                log(f"WARN: skipping row with unparseable tile_id: {tile_id}")
                continue
            per_isolate[accession][tile_num] = Tile(tile_num, strand, seq, tile_id)
            per_tile[tile_num].append((accession, seq))

    return per_isolate, per_tile


def write_fasta(path: Path, records: list[tuple[str, str]]):
    ensure_dir(path.parent)
    with path.open("w") as out:
        for header, seq in records:
            out.write(f">{header}\n")
            # wrap to 80 columns
            for i in range(0, len(seq), 80):
                out.write(seq[i:i+80] + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate per-tile FASTAs from tiled sequences")
    parser.add_argument("--sequences", default="data/sequences.tsv", help="Path to sequences TSV")
    parser.add_argument("--outdir", default="results", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    tsv_path = Path(args.sequences)
    if not tsv_path.exists():
        log(f"ERROR: TSV not found: {tsv_path}")
        sys.exit(2)

    per_isolate, per_tile = read_sequences_tsv(tsv_path)
    if not per_isolate:
        log("ERROR: No isolates parsed from TSV.")
        sys.exit(2)

    tiles_out_dir = outdir / "tiles"
    for tile_num, entries in sorted(per_tile.items()):
        records = [(f"{accession}|tile_{tile_num}", seq) for accession, seq in entries]
        write_fasta(tiles_out_dir / f"tile_{tile_num}.fasta", records)

    log("Done.")


if __name__ == "__main__":
    main()
