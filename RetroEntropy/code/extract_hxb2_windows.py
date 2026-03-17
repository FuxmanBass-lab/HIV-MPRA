#!/usr/bin/env python3
"""
Extract aligned subregions from an aligned FASTA using HXB2 genomic coordinates.

Assumptions:
- Input FASTA is an alignment (all sequences same length).
- The alignment contains an HXB2 reference record (header contains 'HXB2' and/or 'K03455').
- The alignment is for a specific gene (e.g., env, gag), so we need the HXB2 coordinate of the *first* ungapped base
  in the HXB2 sequence in this alignment. For gp160 (env), this is commonly 6225 in HXB2.

Optionally, the script can expand windows to start/end on full codons relative to the gene start codon at --codon-start.

Usage:

  python code/extract_hxb2_windows.py \
  --aln data/HIV1_ALL_2022_env_DNA.fasta \
  --ref-start 6225 \
  --window env1:7432-7632 \
  --window env2:7762-7862 \
  --outdir results/windows_env


  python code/extract_hxb2_windows.py \
  --aln data/HIV1_ALL_2022_env_DNA.fasta \
  --ref-start 6225 \
  --codonize \
  --window env1:7432-7632 \
  --window env2:7762-7862 \
  --outdir results/windows_env_codon


  python extract_hxb2_windows.py \
    --aln ../data/HIV1_ALL_2022_gag_DNA.fasta \
    --ref-start 790 \
    --window gag_mid:1500-1600 \
    --outdir ../results/windows

 
 python code/extract_hxb2_windows.py \
  --aln data/HIV1_ALL_2022_gag_DNA.fasta \
  --ref-start 790 \
  --codonize \
  --window gag_mid:1436-1554 \
  --outdir results/windows_gag_codon

   python code/extract_hxb2_windows.py \
  --aln data/HIV1_ALL_2022_gag_DNA.fasta \
  --ref-start 790 \
  --codonize \
  --window gag_mid:790-2080 \
  --outdir results/windows_gag_all_codon

Outputs:
  results/windows/gag_mid_HXB2_1500_1600.aln.fasta
"""

import argparse
import os
import re
from typing import Dict, List, Tuple

def read_fasta(path: str) -> List[Tuple[str, str]]:
    records = []
    header = None
    seq_chunks = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if header is not None:
            records.append((header, "".join(seq_chunks)))
    return records

def write_fasta(records: List[Tuple[str, str]], path: str, wrap: int = 80) -> None:
    with open(path, "w") as out:
        for h, s in records:
            out.write(f">{h}\n")
            for i in range(0, len(s), wrap):
                out.write(s[i:i+wrap] + "\n")

def find_hxb2(records: List[Tuple[str, str]]) -> Tuple[str, str]:
    # Prefer the record that contains both HXB2 and K03455 if present.
    best = None
    for h, s in records:
        score = 0
        if re.search(r"\bHXB2\b", h, re.IGNORECASE): score += 2
        if re.search(r"\bK03455\b", h, re.IGNORECASE): score += 2
        if re.search(r"\bIIIB\b", h, re.IGNORECASE): score += 1
        if score > 0:
            if best is None or score > best[0]:
                best = (score, h, s)
    if best is None:
        raise RuntimeError("Could not find an HXB2 record in the alignment (looked for 'HXB2'/'K03455').")
    return best[1], best[2]

def build_coord_to_col(hxb2_aln_seq: str, ref_start_coord: int) -> Dict[int, int]:
    """Map HXB2 genomic coordinate -> alignment column index (0-based).

    ref_start_coord is the HXB2 coordinate of the first non-gap base in `hxb2_aln_seq`
    for *this alignment* (e.g., env often starts at 6225 for gp160; gag often starts at 790
    if the alignment begins at the gag start codon).
    """
    coord_to_col: Dict[int, int] = {}
    coord = ref_start_coord
    for col, base in enumerate(hxb2_aln_seq):
        if base != "-":
            coord_to_col[coord] = col
            coord += 1
    return coord_to_col

def parse_window(w: str) -> Tuple[str, int, int]:
    """
    window format: name:start-end  (inclusive coords)
    """
    m = re.match(r"^([^:]+):(\d+)-(\d+)$", w.strip())
    if not m:
        raise ValueError(f"Bad --window '{w}'. Expected format name:start-end (e.g., env1:7432-7632).")
    name, a, b = m.group(1), int(m.group(2)), int(m.group(3))
    if a > b:
        raise ValueError(f"Window start > end in '{w}'.")
    return name, a, b


def adjust_to_codon_boundaries(start: int, end: int, codon_start: int) -> Tuple[int, int]:
    """Adjust an inclusive [start, end] HXB2 window to full codons relative to codon_start.

    codon_start is the HXB2 coordinate of the first base of the gene start codon (ATG).
    The adjusted window is expanded (never shrunk) so that:
      - (start - codon_start) % 3 == 0
      - (end - codon_start + 1) % 3 == 0
    """
    if start > end:
        start, end = end, start

    # Shift start down to codon boundary
    start_offset = (start - codon_start) % 3
    adj_start = start - start_offset

    # Shift end up so length from adj_start is multiple of 3
    length = end - adj_start + 1
    rem = length % 3
    adj_end = end if rem == 0 else end + (3 - rem)

    return adj_start, adj_end


def translate_dna(seq: str) -> str:
    """Translate an ungapped DNA sequence (A/C/G/T/N) into amino acids.

    - Translates in frame starting at position 0.
    - Any codon containing non-ACGT becomes 'X'.
    - Uses standard genetic code.
    """
    codon_table = {
        # Phenylalanine / Leucine
        "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
        "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
        # Isoleucine / Methionine
        "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
        # Valine
        "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        # Serine / Proline / Threonine / Alanine
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
        "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        # Tyrosine / Histidine / Glutamine / Asparagine / Lysine / Aspartate / Glutamate
        "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
        "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        # Cysteine / Tryptophan / Arginine / Glycine
        "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    }

    seq = seq.upper()
    aa = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if set(codon) <= set("ACGT"):
            aa.append(codon_table.get(codon, "X"))
        else:
            aa.append("X")
    return "".join(aa)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aln", required=True, help="Aligned FASTA (gene alignment).")
    ap.add_argument(
        "--ref-start",
        type=int,
        default=6225,
        help=(
            "HXB2 coordinate of the first ungapped base in the HXB2 reference record for THIS alignment. "
            "(env often 6225; gag often 790 if the alignment begins at gag start codon)."
        ),
    )
    # Backward-compatible alias
    ap.add_argument(
        "--env-start",
        dest="ref_start",
        type=int,
        help="DEPRECATED alias for --ref-start.",
    )
    ap.add_argument(
        "--codon-start",
        type=int,
        default=None,
        help=(
            "HXB2 coordinate of the first base of the gene start codon used for --codonize. "
            "Defaults to --ref-start if not provided."
        ),
    )
    ap.add_argument("--window", action="append", required=True,
                    help="Window in HXB2 coords: name:start-end (inclusive). Can be repeated.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--codonize", action="store_true",
                    help="Expand each requested window to start/end on full codons relative to --codon-start or --ref-start.")
    ap.add_argument("--write-hxb2-aa", action="store_true",
                    help="Write the HXB2 amino-acid sequence (translated from ungapped DNA) for each extracted window.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    records = read_fasta(args.aln)
    if not records:
        raise RuntimeError("No records found in alignment FASTA.")

    aln_lens = {len(s) for _, s in records}
    if len(aln_lens) != 1:
        raise RuntimeError(f"Input FASTA is not a strict alignment; found multiple lengths: {sorted(aln_lens)[:10]} ...")

    hxb2_header, hxb2_seq = find_hxb2(records)
    coord_to_col = build_coord_to_col(hxb2_seq, args.ref_start)

    windows_raw = [parse_window(w) for w in args.window]

    windows = []
    for name, start, end in windows_raw:
        if args.codonize:
            codon_start = args.ref_start if args.codon_start is None else args.codon_start
            adj_start, adj_end = adjust_to_codon_boundaries(start, end, codon_start)
            windows.append((name, adj_start, adj_end))
            if (adj_start, adj_end) != (start, end):
                print(f"Codonized {name}: {start}-{end} -> {adj_start}-{adj_end} (len={adj_end-adj_start+1})")
        else:
            windows.append((name, start, end))

    # sanity: ensure requested coords exist in map
    for name, start, end in windows:
        missing = [c for c in (start, end) if c not in coord_to_col]
        if missing:
            raise RuntimeError(
                f"Window {name}:{start}-{end} references coords not covered by HXB2 in this alignment: {missing}\n"
                f"Tip: your --ref-start might be wrong for this alignment, or the alignment doesn't span that region."
            )

    for name, start, end in windows:
        c0 = coord_to_col[start]
        c1 = coord_to_col[end]
        if c0 > c1:
            # should not happen, but be safe
            c0, c1 = c1, c0

        sliced = []
        for h, s in records:
            sub = s[c0:c1+1]  # inclusive
            sliced.append((h, sub))

        out_path = os.path.join(args.outdir, f"{name}_HXB2_{start}_{end}.aln.fasta")
        write_fasta(sliced, out_path)

        # optional: quick report
        print(f"Wrote {out_path}  (columns {c0}-{c1}, length={c1-c0+1})")
        # confirm HXB2 slice has the right number of ungapped bases
        hxb2_slice = hxb2_seq[c0:c1+1]
        ungapped = len(hxb2_slice.replace("-", ""))
        expected = end - start + 1
        if ungapped != expected:
            print(f"WARNING: HXB2 ungapped bases in {name} slice = {ungapped}, expected {expected}. "
                  f"This can happen if HXB2 has gaps here in the alignment; double-check.")

        if args.write_hxb2_aa:
            hxb2_ungapped = hxb2_slice.replace("-", "")
            aa = translate_dna(hxb2_ungapped)
            aa_path = os.path.join(args.outdir, f"{name}_HXB2_{start}_{end}.hxb2.aa.fasta")
            with open(aa_path, "w") as f:
                f.write(f">HXB2|window={name}|HXB2:{start}-{end}\n")
                for i in range(0, len(aa), 80):
                    f.write(aa[i:i+80] + "\n")
            print(f"Wrote {aa_path} (AA length={len(aa)})")

if __name__ == "__main__":
    main()