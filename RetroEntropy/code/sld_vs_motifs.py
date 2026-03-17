

#!/usr/bin/env python3
"""synonymous_leaning_diversity.py

Compute codon-level "synonymous-leaning diversity" (SLD) from a LANL-style
nucleotide multiple sequence alignment (MSA) anchored to HXB2 coordinates.

Motivation
----------
We want a metric for nucleotide diversity that is *preferentially synonymous*
(i.e., does not change the encoded amino acid), while properly accounting for:
  - the genetic code (codon->AA mapping)
  - unequal codon degeneracy across amino acids (e.g., Met has 1 codon, Leu has 6)
  - alignment gaps / ambiguous bases

Core idea
---------
For each ENV-frame codon (relative to --env-start), we consider the random
variable X = codon (64 states) among non-reference isolates, and Y = amino acid
induced by the genetic code (deterministic function Y=f(X)).

We compute:
  H_codon = H(X)
  H_aa    = H(Y)
  H_syn   = H(X|Y) = H(X) - H(Y)   (valid because Y is deterministic of X)

To account for unequal degeneracy, we normalize H_syn by its *maximum possible*
value given the observed amino-acid distribution at that codon:

  H_syn_max = sum_y p(y) * log2( deg(y) )

where deg(y) is the number of standard codons encoding amino acid y.

Then:
  SLD = H_syn / H_syn_max     (in [0,1] when H_syn_max>0)

Interpretation
--------------
- High H_aa: many amino-acid changes (nonsynonymous diversity)
- High H_syn: many codon differences *within* amino-acid classes (synonymous-leaning)
- High SLD (~1): codon usage within each AA is near-maximally diverse given the
  AA composition (strong synonymous variation / wobble diversity)
- Low SLD (~0): little synonymous diversity beyond AA changes or AA has low degeneracy

Outputs
-------
Writes a TSV of per-codon metrics and produces plots:
  - codon_entropy
  - aa_entropy
  - synonymous_entropy (H_syn)
  - synonymous_leaning_diversity (SLD)
  - combined plots (optional overlays)

If `--motifs-json` is provided, the script will also write additional SLD plots
with motif tracks (as shaded regions) for each requested motif type.

Notes
-----
- Entropy is computed *reference-less*: the HXB2 sequence is excluded from all
  entropy distributions.
- Codons with gaps/ambiguous bases (anything outside A/C/G/T) are excluded.
- HXB2 coordinate mapping is provided via --hxb2-start (window start coordinate)
  and derived by walking ungapped HXB2 bases through the alignment.

Example
-------
python code/synonymous_leaning_diversity.py \
  --aln results/windows_codon/env1_HXB2_7431_7634.aln.fasta \
  --hxb2-start 7431 \
  --env-start 6225 \
  --out results/syn/env1.sld.tsv \
  --plot-prefix results/syn/env1.sld

python code/synonymous_leaning_diversity.py \
  --aln results/windows_codon/env2_HXB2_7761_7862.aln.fasta \
  --hxb2-start 7761 \
  --env-start 6225 \
  --out results/syn/env2.sld.tsv \
  --plot-prefix results/syn/env2.sld

python code/sld_vs_motifs.py \
  --aln results/windows_codon/env1_HXB2_7431_7634.aln.fasta \
  --hxb2-start 7431 \
  --env-start 6225 \
  --out results/syn/env1.sld.tsv \
  --plot-prefix results/syn/env1.sld  --motifs-json data/env1_motifs.json \
  --motif-types SP,ETS

python code/sld_vs_motifs.py \
  --aln results/windows_codon/env2_HXB2_7761_7862.aln.fasta \
  --hxb2-start 7761 \
  --env-start 6225 \
  --out results/syn/env2.sld.tsv \
  --plot-prefix results/syn/env2.sld --motifs-json data/env2_motifs.json \
  --motif-types SP,ETS

# Example with ROI annotation for a motif in gag:
python code/sld_vs_motifs.py \
  --aln results/windows_gag_codon/gag_mid_HXB2_1435_1554.aln.fasta \
  --hxb2-start 1435 \
  --env-start 790 \
  --out results/syn/gag_mid.sld.tsv \
  --plot-prefix results/syn_gag/gag_mid.sld \
  --roi motif:1486-1504

 python code/sld_vs_motifs.py   --aln results/windows_gag_all_codon/gag_mid_HXB2_790_2082.aln.fasta   --hxb2-start 790   --env-start 790   --out results/syn_all/gag_all.sld.tsv   --plot-prefix results/syn_gag_all/gag_all.sld   --window-test motifRegion:1486-1504   --window-test-nperm 10000 

 
# command used to run circular permutation test
python code/sld_vs_motifs.py   --aln results/windows_gag_all_codon/gag_mid_HXB2_790_2082.aln.fasta   --hxb2-start 790   --env-start 790   --out results/syn_all/gag_all.sld.tsv   --plot-prefix results/syn_gag_all/gag_all.sld   --window-test motifRegion:1486-1506   --window-test-nperm 10000  
"""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import random
import statistics

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VALID_BASES = set("ACGT-")
VALID_UNGAPPED = set("ACGT")


# --- Genetic code helpers -------------------------------------------------

CODON_TABLE: Dict[str, str] = {
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

AA_DEGENERACY: Dict[str, int] = defaultdict(int)
for cod, aa in CODON_TABLE.items():
    AA_DEGENERACY[aa] += 1


def translate_codon(codon: str) -> Optional[str]:
    """Translate a codon to a single-letter AA.

    Returns None if codon contains non-ACGT bases.
    """
    codon = codon.upper()
    if len(codon) != 3:
        return None
    if set(codon) <= VALID_UNGAPPED:
        return CODON_TABLE.get(codon)
    return None


# --- FASTA + HXB2 coordinate mapping --------------------------------------

def read_fasta(path: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    seq_parts: List[str] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if header is not None:
            records.append((header, "".join(seq_parts)))
    if not records:
        raise RuntimeError(f"No FASTA records found in {path}")
    # sanity: all same length
    L = len(records[0][1])
    for h, s in records:
        if len(s) != L:
            raise RuntimeError("FASTA is not an alignment: sequences have different lengths")
    return records


def find_hxb2(records: List[Tuple[str, str]]) -> Tuple[int, str, str]:
    """Return (index, header, seq) for HXB2 record."""
    for i, (h, s) in enumerate(records):
        hh = h.upper()
        if "HXB2" in hh or "K03455" in hh:
            return i, h, s
    raise RuntimeError("Could not find HXB2 record (expected header containing 'HXB2' or 'K03455').")



def build_coord_maps(hxb2_aln: str, hxb2_start: int) -> Tuple[List[Optional[int]], Dict[int, int]]:
    """Map alignment columns -> HXB2 coordinate, and coordinate -> column.

    HXB2 coordinate increments on non-gap bases in HXB2.
    """
    col_to_coord: List[Optional[int]] = [None] * len(hxb2_aln)
    coord_to_col: Dict[int, int] = {}

    coord = hxb2_start
    for col, ch in enumerate(hxb2_aln):
        if ch == "-":
            col_to_coord[col] = None
        else:
            col_to_coord[col] = coord
            coord_to_col[coord] = col
            coord += 1
    return col_to_coord, coord_to_col


# --- Motif JSON helpers ----------------------------------------------------

def load_motif_intervals(
    json_path: str,
    window_hxb2_start: int,
    beg_is_1based: bool = False,
    end_is_inclusive: bool = False,
    major_clade: Optional[str] = None,
) -> Dict[str, List[Tuple[int, int]]]:
    """Load motif intervals from a LANL-tile-style JSON.

    Expected schema: list of dicts with fields:
      - motif: list[str]
      - beg:   list[int]
      - end:   list[int]

    Coordinates in the JSON are assumed to be offsets (bp) within the window,
    where the window starts at `window_hxb2_start`.

    Returns a dict: motif_name -> list of (start_hxb2, end_hxb2) half-open intervals.
    """
    with open(json_path) as f:
        data = json.load(f)

    out: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    for row in data:
        if major_clade is not None:
            st = (row.get("subtype", "") or "").upper()
            # Treat A1/A2/AE/etc as major clade A; similarly for B/C.
            st_major = None
            if st.startswith("A"):
                st_major = "A"
            elif st.startswith("B"):
                st_major = "B"
            elif st.startswith("C"):
                st_major = "C"
            if st_major != major_clade.upper():
                continue

        motifs = row.get("motif", []) or []
        begs = row.get("beg", []) or []
        ends = row.get("end", []) or []
        if not motifs:
            continue
        if not (len(motifs) == len(begs) == len(ends)):
            # Skip malformed rows gracefully
            continue

        for m, b, e in zip(motifs, begs, ends):
            if m is None:
                continue
            try:
                b_i = int(b)
                e_i = int(e)
            except Exception:
                continue

            if beg_is_1based:
                b_i -= 1
                e_i -= 1

            # Convert to half-open interval in absolute HXB2 coordinates.
            start = window_hxb2_start + b_i
            end = window_hxb2_start + e_i
            if end_is_inclusive:
                end += 1

            if end <= start:
                continue
            out[str(m)].append((start, end))

    return out


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping/adjacent half-open intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


# --- Motif enrichment / depletion tests -----------------------------------

def intervals_to_codon_mask(
    codon_starts: List[int],
    intervals: List[Tuple[int, int]],
    codon_len: int = 3,
) -> List[bool]:
    """Return a boolean mask over codon_starts indicating overlap with any interval.

    A codon at coordinate c is treated as spanning [c, c+codon_len) in HXB2 coords.
    Intervals are half-open [start, end).
    """
    if not intervals:
        return [False] * len(codon_starts)
    intervals = sorted(intervals)
    mask = [False] * len(codon_starts)

    j = 0
    for i, c in enumerate(codon_starts):
        cod_s = c
        cod_e = c + codon_len
        # advance intervals whose end is before this codon
        while j < len(intervals) and intervals[j][1] <= cod_s:
            j += 1
        k = j
        while k < len(intervals) and intervals[k][0] < cod_e:
            s, e = intervals[k]
            if not (e <= cod_s or s >= cod_e):
                mask[i] = True
                break
            k += 1
    return mask


def circular_shift_mask(mask: List[bool], shift: int) -> List[bool]:
    n = len(mask)
    if n == 0:
        return mask
    shift = shift % n
    if shift == 0:
        return mask[:]
    return mask[-shift:] + mask[:-shift]


def nanmedian(vals: List[float]) -> float:
    vv = [v for v in vals if v == v]
    if not vv:
        return float("nan")
    return float(statistics.median(vv))


def motif_delta_median(sld: List[float], mask: List[bool]) -> Tuple[float, int, int]:
    """Compute delta = median(in motif) - median(outside motif) ignoring NaNs."""
    in_vals = [v for v, m in zip(sld, mask) if m and v == v]
    out_vals = [v for v, m in zip(sld, mask) if (not m) and v == v]
    if not in_vals or not out_vals:
        return float("nan"), len(in_vals), len(out_vals)
    d = float(statistics.median(in_vals) - statistics.median(out_vals))
    return d, len(in_vals), len(out_vals)


def metric_delta_median(values: List[float], mask: List[bool]) -> Tuple[float, int, int]:
    """Compute delta = median(in mask) - median(outside mask) ignoring NaNs."""
    in_vals = [v for v, m in zip(values, mask) if m and v == v]
    out_vals = [v for v, m in zip(values, mask) if (not m) and v == v]
    if not in_vals or not out_vals:
        return float("nan"), len(in_vals), len(out_vals)
    d = float(statistics.median(in_vals) - statistics.median(out_vals))
    return d, len(in_vals), len(out_vals)


def permutation_test_circular_shift_metric(
    values: List[float],
    mask: List[bool],
    n_perm: int = 5000,
    seed: int = 1,
) -> Tuple[float, float, List[float]]:
    """Circular-shift permutation test for depletion on a generic metric.

    Returns (delta_obs, p_one_sided, deltas_perm), where p_one_sided tests whether
    delta_obs is unusually low (<=) relative to the null.
    """
    delta_obs, _, _ = metric_delta_median(values, mask)
    if delta_obs != delta_obs:
        return float("nan"), float("nan"), []

    rng = random.Random(seed)
    n = len(mask)
    deltas: List[float] = []
    for _ in range(n_perm):
        shift = rng.randrange(n) if n > 0 else 0
        m2 = circular_shift_mask(mask, shift)
        d, _, _ = metric_delta_median(values, m2)
        if d == d:
            deltas.append(d)

    if not deltas:
        return delta_obs, float("nan"), []

    # One-sided p-value for depletion (delta more negative than expected)
    p = (sum(1 for d in deltas if d <= delta_obs) + 1) / (len(deltas) + 1)
    return delta_obs, p, deltas

def permutation_test_circular_shift(
    sld: List[float],
    mask: List[bool],
    n_perm: int = 2000,
    seed: int = 1,
) -> Tuple[float, float, List[float]]:
    """Circular-shift permutation test for motif depletion.

    Returns (delta_obs, p_one_sided, deltas_perm), where p_one_sided tests whether
    delta_obs is unusually low (<=) relative to the null.
    """
    delta_obs, _, _ = motif_delta_median(sld, mask)
    if delta_obs != delta_obs:
        return float("nan"), float("nan"), []

    rng = random.Random(seed)
    n = len(mask)
    deltas: List[float] = []
    for _ in range(n_perm):
        shift = rng.randrange(n) if n > 0 else 0
        m2 = circular_shift_mask(mask, shift)
        d, _, _ = motif_delta_median(sld, m2)
        if d == d:
            deltas.append(d)

    if not deltas:
        return delta_obs, float("nan"), []

    # One-sided p-value for depletion (delta more negative than expected)
    p = (sum(1 for d in deltas if d <= delta_obs) + 1) / (len(deltas) + 1)
    return delta_obs, p, deltas


# --- Entropy helpers -------------------------------------------------------

def shannon_entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log2(p)
    return H


@dataclass
class CodonMetrics:
    hxb2_codon_start: int
    env_codon_index: int
    n_total_nonref: int
    n_used: int
    # entropies (bits)
    H_codon: float
    H_aa: float
    H_syn: float
    H_syn_max: float
    SLD: float
    # optional descriptive stats
    n_unique_codons: int
    n_unique_aas: int


def infer_major_clade(header: str) -> Optional[str]:
    """Infer major clade (A/B/C) from a LANL-style header.

    Examples:
      >B.FR.83.... -> B
      >A1.BE.94... -> A

    Returns 'A', 'B', 'C' or None if not recognized.
    """
    if not header:
        return None
    first = header.split(".", 1)[0].upper()
    if first.startswith("A"):
        return "A"
    if first.startswith("B"):
        return "B"
    if first.startswith("C"):
        return "C"
    return None


def main() -> None:
    def compute_bp_entropy_for_seqs(
        seqs_subset: List[str],
        min_used: int,
    ) -> Tuple[List[int], List[float], List[int]]:
        """Compute per-base Shannon entropy across HXB2 coords in this window.

        Returns (coords, H_bits, n_used) where coords are all mapped HXB2 coords in [min..max].
        Bases considered: A/C/G/T only. Gaps/ambiguous are ignored.
        """
        if not available_coords:
            return [], [], []
        win_start = min(available_coords)
        win_end = max(available_coords)

        coords_out: List[int] = []
        H_out: List[float] = []
        n_used_out: List[int] = []

        for coord in range(win_start, win_end + 1):
            col = coord_to_col.get(coord)
            if col is None:
                continue
            counts: Counter = Counter()
            used = 0
            for s in seqs_subset:
                b = s[col]
                if b not in VALID_UNGAPPED:
                    continue
                used += 1
                counts[b] += 1

            if used < min_used:
                H = float("nan")
            else:
                H = shannon_entropy_from_counts(counts)

            coords_out.append(coord)
            H_out.append(H)
            n_used_out.append(used)

        return coords_out, H_out, n_used_out
    ap = argparse.ArgumentParser()
    ap.add_argument("--aln", required=True, help="Aligned nucleotide FASTA (includes HXB2).")
    ap.add_argument("--hxb2-start", type=int, required=True, help="HXB2 coordinate of first ungapped base in this window.")
    ap.add_argument("--env-start", type=int, required=True, help="HXB2 coordinate of ENV start codon first base (ATG).")
    ap.add_argument("--out", required=True, help="Output TSV path.")
    ap.add_argument(
        "--sld-csv",
        default=None,
        help=(
            "Optional path to write a compact CSV of SLD vs codon position. "
            "Columns: hxb2_codon_start,env_codon_index,SLD,n_used,codons_used."
        ),
    )
    ap.add_argument("--plot-prefix", default=None, help="Prefix for plot outputs (no extension).")
    ap.add_argument("--min-used", type=int, default=50, help="Min usable (non-ambiguous) codons to report/plot (default 50).")
    ap.add_argument("--exclude-stop", action="store_true", help="If set, exclude stop-codon translations ('*') from AA distribution.")
    ap.add_argument("--motifs-json", default=None, help="Optional motif JSON file to overlay on SLD plots.")
    ap.add_argument(
        "--motif-types",
        default=None,
        help="Comma-separated motif types to plot (e.g., 'SP,NFKB'). If omitted, all motif types in the JSON are plotted.",
    )
    ap.add_argument("--motif-beg-1based", action="store_true", help="Treat motif beg/end offsets in JSON as 1-based instead of 0-based.")
    ap.add_argument("--motif-end-inclusive", action="store_true", help="Treat motif end offsets in JSON as inclusive (end+1).")
    ap.add_argument("--motif-merge", action="store_true", help="Merge overlapping/adjacent motif intervals before plotting.")
    ap.add_argument("--clade-plots", action="store_true", help="Also write SLD plots stratified by major clade (A/B/C).")
    ap.add_argument("--clades", default="A,B,C", help="Comma-separated major clades to plot (default: A,B,C).")
    ap.add_argument("--min-used-clade", type=int, default=30, help="Min usable codons per site for clade-stratified plots (default 30).")
    ap.add_argument("--motif-tests", action="store_true", help="Run motif depletion tests for SLD (delta median + circular-shift permutation).")
    ap.add_argument("--motif-tests-nperm", type=int, default=2000, help="Number of permutations for motif tests (default 2000).")
    ap.add_argument("--motif-tests-seed", type=int, default=1, help="Random seed for motif tests (default 1).")
    ap.add_argument("--motif-tests-out", default=None, help="Output prefix for motif test summary/plots. Default: <plot-prefix>.motif_tests")
    ap.add_argument("--roi", default=None, help="Optional region-of-interest to annotate on plots. Format: start-end or name:start-end (HXB2 coordinates, inclusive).")
    ap.add_argument("--roi-color-alpha", type=float, default=0.18, help="Alpha value for ROI region shading (default 0.18).")
    ap.add_argument(
        "--print-roi-seq",
        action="store_true",
        help=(
            "If set (and --roi provided), print the HXB2 base sequence for the ROI region "
            "to stdout, inserting spaces between codons using the frame defined by --env-start."
        ),
    )
    ap.add_argument(
        "--print-window-seq",
        action="store_true",
        help=(
            "If set, print the HXB2 base sequence for the entire plotted window (min..max HXB2 coords in this alignment window), "
            "inserting spaces between codons using the frame defined by --env-start."
        ),
    )
    ap.add_argument(
        "--print-seq-wrap",
        type=int,
        default=0,
        help=(
            "Optional wrap width for printed sequences (after codon spacing). 0 means no wrapping."
        ),
    )
    ap.add_argument(
        "--window-test",
        default=None,
        help=(
            "Motif-independent depletion test for a user-specified window. "
            "Format: start-end or name:start-end (HXB2 coordinates, inclusive). "
            "Tests whether the window has unusually low H(codon), H(AA), and H(codon|AA) vs the rest "
            "using a circular-shift permutation test over codon positions."
        ),
    )
    ap.add_argument(
        "--window-test-nperm",
        type=int,
        default=5000,
        help="Number of circular-shift permutations for --window-test (default 5000).",
    )
    ap.add_argument(
        "--window-test-seed",
        type=int,
        default=1,
        help="Random seed for --window-test (default 1).",
    )
    ap.add_argument(
        "--window-test-out",
        default=None,
        help="Output prefix for --window-test summary/plots. Default: <plot-prefix>.window_test",
    )
    args = ap.parse_args()

    records = read_fasta(args.aln)

    hxb2_idx, hxb2_header, hxb2_seq_raw = find_hxb2(records)

    # Clean sequences to A/C/G/T/-/N (convert anything else to N)
    headers: List[str] = []
    seqs: List[str] = []
    for h, s in records:
        s = s.upper()
        s = "".join(ch if ch in VALID_BASES else "N" for ch in s)
        headers.append(h)
        seqs.append(s)

    hxb2 = seqs[hxb2_idx]
    col_to_coord, coord_to_col = build_coord_maps(hxb2, args.hxb2_start)

    # Non-reference isolates for entropy/distributions
    seqs_no_ref = [s for i, s in enumerate(seqs) if i != hxb2_idx]
    n_nonref = len(seqs_no_ref)
    if n_nonref <= 0:
        raise RuntimeError("Need at least one non-reference sequence.")

    # Determine codon starts in this window that align to ENV frame
    available_coords = sorted(coord_to_col.keys())
    codon_starts: List[int] = []
    for c in available_coords:
        if ((c - args.env_start) % 3) != 0:
            continue
        if (c + 1) not in coord_to_col or (c + 2) not in coord_to_col:
            continue
        codon_starts.append(c)

    if not codon_starts:
        raise RuntimeError("No full codons found in this window for the provided --env-start / --hxb2-start.")

    # Helper to extract HXB2-mapped codon (3 coords -> 3 alignment columns)
    def get_codon(seq: str, start_coord: int) -> str:
        i0 = coord_to_col[start_coord]
        i1 = coord_to_col[start_coord + 1]
        i2 = coord_to_col[start_coord + 2]
        return seq[i0] + seq[i1] + seq[i2]

    def compute_metrics_for_seqs(seqs_subset: List[str], min_used: int) -> List[CodonMetrics]:
        """Compute CodonMetrics over `codon_starts` using the provided sequences (non-reference only)."""
        metrics_local: List[CodonMetrics] = []
        n_nonref_local = len(seqs_subset)
        if n_nonref_local <= 0:
            raise RuntimeError("Need at least one non-reference sequence in subset")

        for c in codon_starts:
            codon_counts: Counter = Counter()
            aa_counts: Counter = Counter()

            used = 0
            for s in seqs_subset:
                cod = get_codon(s, c)
                aa = translate_codon(cod)
                if aa is None:
                    continue
                if args.exclude_stop and aa == "*":
                    continue
                used += 1
                codon_counts[cod] += 1
                aa_counts[aa] += 1

            if used < min_used:
                H_codon = float("nan")
                H_aa = float("nan")
                H_syn = float("nan")
                H_syn_max = float("nan")
                sld = float("nan")
                n_uc = 0
                n_ua = 0
            else:
                H_codon = shannon_entropy_from_counts(codon_counts)
                H_aa = shannon_entropy_from_counts(aa_counts)

                # Synonymous entropy H(X|Y)
                H_syn = 0.0
                for aa, cnt in aa_counts.items():
                    p = cnt / used
                    cod_counts_aa = Counter({cod: n for cod, n in codon_counts.items() if translate_codon(cod) == aa})
                    H_syn += p * shannon_entropy_from_counts(cod_counts_aa)

                # Max synonymous capacity under observed AA distribution
                H_syn_max = 0.0
                for aa, cnt in aa_counts.items():
                    p = cnt / used
                    deg = AA_DEGENERACY.get(aa, 0)
                    if deg and deg > 1:
                        H_syn_max += p * math.log2(deg)

                if H_syn_max > 0 and H_syn == H_syn:
                    sld_raw = H_syn / H_syn_max
                    # Clip for stability
                    if sld_raw < 0:
                        sld = 0.0
                    elif sld_raw > 1:
                        sld = 1.0
                    else:
                        sld = sld_raw
                else:
                    sld = float("nan")

                n_uc = len(codon_counts)
                n_ua = len(aa_counts)

            env_codon_index = ((c - args.env_start) // 3) + 1
            metrics_local.append(
                CodonMetrics(
                    hxb2_codon_start=c,
                    env_codon_index=env_codon_index,
                    n_total_nonref=n_nonref_local,
                    n_used=used,
                    H_codon=H_codon,
                    H_aa=H_aa,
                    H_syn=H_syn,
                    H_syn_max=H_syn_max,
                    SLD=sld,
                    n_unique_codons=n_uc,
                    n_unique_aas=n_ua,
                )
            )
        return metrics_local

    # Compute metrics for all non-reference isolates
    metrics = compute_metrics_for_seqs(seqs_no_ref, args.min_used)

    # Write TSV
    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with open(args.out, "w") as f:
        f.write("\t".join([
            "hxb2_codon_start",
            "env_codon_index",
            "n_total_nonref",
            "n_used",
            "H_codon_bits",
            "H_aa_bits",
            "H_codon_given_AA_bits",
            "H_syn_max_bits",
            "SLD",
            "n_unique_codons",
            "n_unique_aas",
        ]) + "\n")
        for m in metrics:
            def fmt(x: float) -> str:
                return f"{x:.6f}" if x == x else "NA"
            f.write("\t".join([
                str(m.hxb2_codon_start),
                str(m.env_codon_index),
                str(m.n_total_nonref),
                str(m.n_used),
                fmt(m.H_codon),
                fmt(m.H_aa),
                fmt(m.H_syn),
                fmt(m.H_syn_max),
                fmt(m.SLD),
                str(m.n_unique_codons),
                str(m.n_unique_aas),
            ]) + "\n")

    print(f"Wrote TSV: {args.out}")

    # Optionally write compact SLD CSV
    if args.sld_csv is not None and str(args.sld_csv).strip():
        csv_path = args.sld_csv
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(csv_path, "w") as fcsv:
            fcsv.write("hxb2_codon_start,env_codon_index,SLD,n_used,codons_used\n")
            for m in metrics:
                # Recompute codon usage at this site for reporting
                codon_counts: Counter = Counter()
                for s in seqs_no_ref:
                    cod = get_codon(s, m.hxb2_codon_start)
                    aa = translate_codon(cod)
                    if aa is None:
                        continue
                    if args.exclude_stop and aa == "*":
                        continue
                    codon_counts[cod] += 1

                # Format as CODON:count|CODON:count ...
                parts = []
                for cod, cnt in sorted(codon_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                    parts.append(f"{cod}:{cnt}")
                codons_used = "|".join(parts)

                sld_str = f"{m.SLD:.6f}" if m.SLD == m.SLD else ""
                fcsv.write(
                    f"{m.hxb2_codon_start},{m.env_codon_index},{sld_str},{m.n_used},{codons_used}\n"
                )
        print(f"Wrote SLD CSV: {csv_path}")

    # Plots
    if args.plot_prefix is None:
        prefix = os.path.splitext(args.out)[0]
    else:
        prefix = args.plot_prefix
        os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    xs = [m.hxb2_codon_start for m in metrics]
    codon_starts_xs = xs[:]  # for motif mask alignment

    def take(attr: str) -> List[float]:
        return [getattr(m, attr) for m in metrics]

    Hc = take("H_codon")
    Ha = take("H_aa")
    Hs = take("H_syn")
    S  = take("SLD")

    # --- ROI parsing helper ---
    roi = None  # (name, start, end_exclusive)
    if args.roi is not None and str(args.roi).strip():
        roi_raw = str(args.roi).strip()
        if ":" in roi_raw:
            roi_name, coord = roi_raw.split(":", 1)
            roi_name = roi_name.strip() or "ROI"
        else:
            roi_name, coord = "ROI", roi_raw
        if "-" not in coord:
            raise RuntimeError("--roi must be of form start-end or name:start-end")
        a, b = coord.split("-", 1)
        roi_start = int(a)
        roi_end = int(b)
        if roi_start > roi_end:
            roi_start, roi_end = roi_end, roi_start
        roi = (roi_name, roi_start, roi_end + 1)
    
    # --- Window-test parsing helper ---
    window_test = None  # (name, start, end_exclusive)
    if args.window_test is not None and str(args.window_test).strip():
        wt_raw = str(args.window_test).strip()
        if ":" in wt_raw:
            wt_name, coord = wt_raw.split(":", 1)
            wt_name = wt_name.strip() or "WINDOW"
        else:
            wt_name, coord = "WINDOW", wt_raw
        if "-" not in coord:
            raise RuntimeError("--window-test must be of form start-end or name:start-end")
        a, b = coord.split("-", 1)
        wt_start = int(a)
        wt_end = int(b)
        if wt_start > wt_end:
            wt_start, wt_end = wt_end, wt_start
        window_test = (wt_name, wt_start, wt_end + 1)

    # --- Codon spacing and wrapping helpers ---
    def format_codon_spaced_sequence(bases_with_coords: List[Tuple[int, str]]) -> str:
        """Given (coord, base) tuples in ascending coord order, insert spaces at codon boundaries."""
        parts: List[str] = []
        for i, (coord, ch) in enumerate(bases_with_coords):
            frame = (coord - args.env_start) % 3
            if i > 0 and frame == 0:
                parts.append(" ")
            parts.append(ch)
        return "".join(parts)

    def maybe_wrap(s: str) -> str:
        w = int(args.print_seq_wrap)
        if w <= 0:
            return s
        return "\n".join(s[i:i+w] for i in range(0, len(s), w))

    # Optionally print the HXB2 window sequence with codon spacing
    if args.print_window_seq:
        if not available_coords:
            raise RuntimeError("No mapped HXB2 coordinates available to print window sequence")
        win_start = min(available_coords)
        win_end = max(available_coords)

        bases_win: List[Tuple[int, str]] = []
        missing = 0
        for coord in range(win_start, win_end + 1):
            col = coord_to_col.get(coord)
            if col is None:
                missing += 1
                continue
            ch = hxb2[col]
            if ch == "-":
                missing += 1
                continue
            bases_win.append((coord, ch))

        if not bases_win:
            raise RuntimeError(f"No HXB2 bases could be extracted for window HXB2:{win_start}-{win_end}")

        seq_str = format_codon_spaced_sequence(bases_win)
        print(f"WINDOW HXB2:{win_start}-{win_end}")
        if missing:
            print(f"(warning: skipped {missing} coords not present in HXB2 mapping for this window)")
        print(maybe_wrap(seq_str))

    # Optionally print the HXB2 ROI sequence with codon spacing
    if args.print_roi_seq:
        if roi is None:
            raise RuntimeError("--print-roi-seq requires --roi")
        roi_name, rs, re = roi  # re is end-exclusive
        bases = []
        missing = 0
        for coord in range(rs, re):
            col = coord_to_col.get(coord)
            if col is None:
                missing += 1
                continue
            ch = hxb2[col]
            # hxb2 at mapped coords should not be '-', but guard anyway
            if ch == "-":
                missing += 1
                continue
            bases.append((coord, ch))

        if not bases:
            raise RuntimeError(f"No HXB2 bases could be extracted for ROI {roi_name}:{rs}-{re-1}")

        seq_str = format_codon_spaced_sequence(bases)

        print(f"ROI {roi_name} HXB2:{rs}-{re-1}")
        if missing:
            print(f"(warning: skipped {missing} coords not present in HXB2 mapping for this window)")
        print(maybe_wrap(seq_str))

    # Individual tracks

    # --- Entropy decomposition plot helper ---
    def plot_entropy_decomposition(
        metrics_list: List[CodonMetrics],
        outpath: str,
        title: str,
    ) -> None:
        xs_local = [m.hxb2_codon_start for m in metrics_list]
        Hc_local = [m.H_codon for m in metrics_list]
        Ha_local = [m.H_aa for m in metrics_list]
        Hs_local = [m.H_syn for m in metrics_list]
        S_local = [m.SLD for m in metrics_list]
        S_local_clip = [min(1.0, max(0.0, v)) if v == v else v for v in S_local]

        fig, ax1 = plt.subplots(figsize=(24, 4))

        x_c = [x for x, v in zip(xs_local, Hc_local) if v == v]
        y_c = [v for v in Hc_local if v == v]
        x_a = [x for x, v in zip(xs_local, Ha_local) if v == v]
        y_a = [v for v in Ha_local if v == v]
        x_s = [x for x, v in zip(xs_local, Hs_local) if v == v]
        y_s = [v for v in Hs_local if v == v]

        if x_c:
            ax1.plot(x_c, y_c, label="H(codon)")
        if x_a:
            ax1.plot(x_a, y_a, label="H(AA)")
        if x_s:
            ax1.plot(x_s, y_s, label="H(codon|AA)")

        ax2 = ax1.twinx()
        ax2.set_ylabel("SLD (0-1)")
        ax2.set_ylim(0, 1)
        ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        x_sld = [x for x, v in zip(xs_local, S_local_clip) if v == v]
        y_sld = [v for v in S_local_clip if v == v]
        if x_sld:
            ax2.plot(x_sld, y_sld, label="SLD", linestyle="--")

        if roi is not None:
            roi_name, rs, re = roi
            ax1.axvspan(rs, re, alpha=args.roi_color_alpha)
            ax1.text(rs, 0.98, roi_name, va='top', transform=ax1.get_xaxis_transform())

        ax1.set_xlabel("HXB2 coordinate (codon start)")
        ax1.set_ylabel("Entropy (bits)")
        ax1.set_title(title)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False)

        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        print(f"Wrote plot: {outpath}")

    def plot_entropy_decomposition_no_sld(
        metrics_list: List[CodonMetrics],
        outpath: str,
        title: str,
    ) -> None:
        # Top panel: codon/AA/syn entropies at codon starts
        xs_local = [m.hxb2_codon_start for m in metrics_list]
        Hc_local = [m.H_codon for m in metrics_list]
        Ha_local = [m.H_aa for m in metrics_list]
        Hs_local = [m.H_syn for m in metrics_list]

        # Bottom panel: single-bp entropy across the whole window
        bp_xs, bp_H, _bp_n = compute_bp_entropy_for_seqs(seqs_no_ref, args.min_used)

        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(24, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
        )

        # --- Top ---
        x_c = [x for x, v in zip(xs_local, Hc_local) if v == v]
        y_c = [v for v in Hc_local if v == v]
        x_a = [x for x, v in zip(xs_local, Ha_local) if v == v]
        y_a = [v for v in Ha_local if v == v]
        x_s = [x for x, v in zip(xs_local, Hs_local) if v == v]
        y_s = [v for v in Hs_local if v == v]

        if x_c:
            ax_top.plot(x_c, y_c, label="H(codon)")
        if x_a:
            ax_top.plot(x_a, y_a, label="H(AA)")
        if x_s:
            ax_top.plot(x_s, y_s, label="H(codon|AA)")

        if roi is not None:
            roi_name, rs, re = roi
            ax_top.axvspan(rs, re, alpha=args.roi_color_alpha)
            ax_top.text(rs, 0.98, roi_name, va='top', transform=ax_top.get_xaxis_transform())

        ax_top.set_ylabel("Entropy (bits)")
        ax_top.set_title(title)
        ax_top.legend(loc="upper right", frameon=False)

        # --- Bottom (bp entropy) ---
        bp_x2 = [x for x, v in zip(bp_xs, bp_H) if v == v]
        bp_y2 = [v for v in bp_H if v == v]
        if bp_x2:
            ax_bot.plot(bp_x2, bp_y2, label="H(bp)")

        if roi is not None:
            roi_name, rs, re = roi
            ax_bot.axvspan(rs, re, alpha=args.roi_color_alpha)

        ax_bot.set_xlabel("HXB2 coordinate")
        ax_bot.set_ylabel("BP entropy (bits)")
        ax_bot.legend(loc="upper right", frameon=False)

        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        print(f"Wrote plot: {outpath}")
        
    def plot_track(y: List[float], ylabel: str, title: str, outpath: str, ylim: Optional[Tuple[float, float]] = None) -> None:
        x2 = [x for x, v in zip(xs, y) if v == v]
        y2 = [v for v in y if v == v]
        plt.figure(figsize=(24, 4))
        if x2:
            plt.plot(x2, y2)
        if ylim is not None:
            plt.ylim(*ylim)
        # ROI shading (if present)
        if roi is not None:
            roi_name, rs, re = roi
            plt.axvspan(rs, re, alpha=args.roi_color_alpha)
            # Determine y max for text
            if ylim is not None:
                ymax = ylim[1]
            else:
                ymin, ymax = plt.gca().get_ylim()
            plt.text(rs, ymax * 0.95, roi_name, va='top')
        plt.xlabel("HXB2 coordinate (codon start)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
        print(f"Wrote plot: {outpath}")

    def plot_sld_with_motifs(
        sld: List[float],
        motif_name: str,
        intervals: List[Tuple[int, int]],
        outpath: str,
    ) -> None:
        # Top: SLD track. Bottom: motif intervals as a separate lane.
        x2 = [x for x, v in zip(xs, sld) if v == v]
        y2 = [v for v in sld if v == v]

        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(24, 4),
            sharex=True,
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
        )

        # SLD line
        if x2:
            ax_top.plot(x2, y2)
        ax_top.set_ylim(0, 1)
        ax_top.set_ylabel("SLD (0-1)")
        ax_top.set_title(f"SLD with motif track: {motif_name}")

        # ROI shading (top axis)
        if roi is not None:
            roi_name, rs, re = roi
            ax_top.axvspan(rs, re, alpha=args.roi_color_alpha)
            ax_top.text(rs, 0.98, roi_name, va='top', transform=ax_top.get_xaxis_transform())

        # Motif lane
        ax_bot.set_ylim(0, 1)
        ax_bot.set_yticks([])
        ax_bot.set_ylabel(motif_name)

        # Draw intervals as filled rectangles in the bottom lane
        for (s, e) in intervals:
            ax_bot.axvspan(s, e, ymin=0.15, ymax=0.85, alpha=0.35)

        # ROI shading (bottom axis)
        if roi is not None:
            roi_name, rs, re = roi
            ax_bot.axvspan(rs, re, alpha=args.roi_color_alpha)

        # Minimal cosmetics
        ax_bot.set_xlabel("HXB2 coordinate (codon start)")

        # Ensure x-limits cover the data consistently
        if x2:
            ax_bot.set_xlim(min(x2), max(x2))

        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        print(f"Wrote plot: {outpath}")

    plot_track(Hc, "Shannon entropy (bits)", "Codon entropy vs HXB2", f"{prefix}.codon_entropy.png")
    plot_track(Ha, "Shannon entropy (bits)", "Amino-acid entropy vs HXB2", f"{prefix}.aa_entropy.png")
    plot_track(Hs, "Conditional entropy (bits)", "Conditional entropy H(codon|AA) vs HXB2", f"{prefix}.synonymous_entropy.png")
    S_clip = [min(1.0, max(0.0, v)) if v == v else v for v in S]
    S_clip_vals = [v for v in S_clip if v == v]
    if S_clip_vals:
        print(f"SLD range (clipped): min={min(S_clip_vals):.6f}, max={max(S_clip_vals):.6f}")
    plot_track(S_clip, "Synonymous-leaning diversity (0-1)", "Synonymous-leaning diversity (degeneracy-normalized)", f"{prefix}.SLD.png", ylim=(0, 1))

    # --- Motif-independent window depletion test (optional) ---
    if window_test is not None:
        wt_name, wt_s, wt_e = window_test

        outpref = args.window_test_out
        if outpref is None:
            outpref = f"{prefix}.window_test"
        os.makedirs(os.path.dirname(outpref) or ".", exist_ok=True)

        # Mask at codon-start positions: codon at c spans [c, c+3)
        mask = intervals_to_codon_mask(codon_starts_xs, [(wt_s, wt_e)])

        summary_path = f"{outpref}.summary.tsv"
        summary_rows_for_plot = []  # (metric_label, d_obs, p, deltas)
        with open(summary_path, "w") as fsum:
            fsum.write("\t".join([
                "window_name",
                "window_start",
                "window_end_inclusive",
                "metric",
                "n_in",
                "n_out",
                "delta_median_in_minus_out",
                "p_one_sided_depletion",
                "n_perm_effective",
            ]) + "\n")

            tests = [
                ("H(codon)", Hc, "Δ median(H(codon)_in) − median(H(codon)_out)"),
                ("H(AA)", Ha, "Δ median(H(AA)_in) − median(H(AA)_out)"),
                ("H(codon|AA)", Hs, "Δ median(H(codon|AA)_in) − median(H(codon|AA)_out)"),
            ]

            for metric_name, values, xlabel in tests:
                d_obs2, p, deltas = permutation_test_circular_shift_metric(
                    values,
                    mask,
                    n_perm=args.window_test_nperm,
                    seed=args.window_test_seed,
                )
                summary_rows_for_plot.append((metric_name, d_obs2, p, deltas))
                _, n_in, n_out = metric_delta_median(values, mask)

                fsum.write("\t".join([
                    wt_name,
                    str(wt_s),
                    str(wt_e - 1),
                    metric_name,
                    str(n_in),
                    str(n_out),
                    f"{d_obs2:.6f}" if d_obs2 == d_obs2 else "NA",
                    f"{p:.6g}" if p == p else "NA",
                    str(len(deltas)),
                ]) + "\n")

                if deltas:
                    plt.figure(figsize=(10, 4))
                    plt.hist(deltas, bins=40)
                    plt.axvline(d_obs2, linestyle="--")
                    plt.xlabel(xlabel)
                    plt.ylabel("Count")
                    # Plot title: use metric_name directly
                    plt.title(
                        f"Window depletion test (circular shift): {wt_name} {wt_s}-{wt_e-1}  {metric_name}  p={p:.3g}"
                    )
                    plt.tight_layout()
                    token = metric_name.replace("(", "").replace(")", "").replace("|", "_").replace(" ", "").replace("/", "_")
                    outpng = f"{outpref}.null.{token}.png"
                    plt.savefig(outpng, dpi=200)
                    plt.close()
                    print(f"Wrote plot: {outpng}")

        print(f"Wrote window test summary: {summary_path}")

        # Compact summary plot: violin of null Δ distributions + observed Δ
        if summary_rows_for_plot:
            labels = [r[0] for r in summary_rows_for_plot]
            d_obs_list = [r[1] for r in summary_rows_for_plot]
            p_list = [r[2] for r in summary_rows_for_plot]
            deltas_list = [r[3] for r in summary_rows_for_plot]

            fig, ax = plt.subplots(figsize=(3, 4))

            # Prepare data for violins (null distributions)
            data = [[d for d in deltas if d == d] for deltas in deltas_list]

            parts = ax.violinplot(
                data,
                positions=range(len(labels)),
                widths=0.35,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )

            # Color map to match entropy tracks
            metric_colors = {
                "H(codon)": "#1f77b4",      # same as default matplotlib blue
                "H(AA)": "#ff7f0e",         # default matplotlib orange
                "H(codon|AA)": "#2ca02c",   # default matplotlib green
            }

            # Publication-style violin aesthetics with color per metric
            for pc, label in zip(parts["bodies"], labels):
                pc.set_facecolor(metric_colors.get(label, "#4C72B0"))
                pc.set_edgecolor("black")
                pc.set_alpha(0.6)
                pc.set_linewidth(0.8)

            if "cmedians" in parts:
                parts["cmedians"].set_color("black")
                parts["cmedians"].set_linewidth(1.5)

            xs = list(range(len(labels)))
            ax.scatter(
                xs,
                d_obs_list,
                color="black",
                s=40,
                zorder=3,
            )

            # Add significance stars above violins
            for i, (deltas, d_obs, p) in enumerate(zip(deltas_list, d_obs_list, p_list)):
                if p == p and p < 0.05:
                    ymax = max(deltas) if deltas else d_obs
                    ax.text(
                        i,
                        ymax * 1.05,
                        "*",
                        ha="center",
                        va="bottom",
                        fontsize=16,
                        fontweight="bold",
                    )

            ax.axhline(0.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels)
            ax.set_ylabel("Δ median (ROI − rest)")
            # ax.set_title(f"Window depletion summary: {wt_name} ({wt_s}-{wt_e-1})")

            # Remove grid + top/right spines
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig.tight_layout()

            out_sum_png = f"{outpref}.summary_plot.violin.png"
            out_sum_pdf = f"{outpref}.summary_plot.violin.pdf"
            fig.savefig(out_sum_png, dpi=200)
            fig.savefig(out_sum_pdf)
            plt.close(fig)

            print(f"Wrote window test violin summary plot: {out_sum_png}")
            print(f"Wrote window test violin summary plot: {out_sum_pdf}")

    # Motif-overlay SLD plots (optional)
    if args.motifs_json is not None:
        motif_map = load_motif_intervals(
            args.motifs_json,
            window_hxb2_start=args.hxb2_start,
            beg_is_1based=args.motif_beg_1based,
            end_is_inclusive=args.motif_end_inclusive,
            major_clade=None,
        )

        if args.motif_types is None or not args.motif_types.strip():
            motif_types = sorted(motif_map.keys())
        else:
            motif_types = [m.strip() for m in args.motif_types.split(",") if m.strip()]

        for m in motif_types:
            intervals = motif_map.get(m, [])
            if not intervals:
                continue
            if args.motif_merge:
                intervals = merge_intervals(intervals)

            out_m = f"{prefix}.SLD.{m}.png"
            plot_sld_with_motifs(S_clip, m, intervals, out_m)

    # Motif depletion tests on SLD (optional)
    if args.motif_tests:
        if args.motifs_json is None:
            raise RuntimeError("--motif-tests requires --motifs-json")

        # Re-use motif_map from earlier if available; otherwise load.
        if 'motif_map' not in locals() or motif_map is None:
            motif_map = load_motif_intervals(
                args.motifs_json,
                window_hxb2_start=args.hxb2_start,
                beg_is_1based=args.motif_beg_1based,
                end_is_inclusive=args.motif_end_inclusive,
                major_clade=None,
            )

        if args.motif_types is None or not args.motif_types.strip():
            motif_types_test = sorted(motif_map.keys())
        else:
            motif_types_test = [m.strip() for m in args.motif_types.split(",") if m.strip()]

        outpref = args.motif_tests_out
        if outpref is None:
            outpref = f"{prefix}.motif_tests"
        os.makedirs(os.path.dirname(outpref) or ".", exist_ok=True)

        summary_path = f"{outpref}.summary.tsv"
        with open(summary_path, "w") as fsum:
            fsum.write("\t".join([
                "scope",
                "motif",
                "n_in",
                "n_out",
                "delta_median_in_minus_out",
                "p_one_sided_depletion",
                "n_perm_effective",
            ]) + "\n")

            for mname in motif_types_test:
                intervals = motif_map.get(mname, [])
                if not intervals:
                    continue
                if args.motif_merge:
                    intervals = merge_intervals(intervals)

                mask = intervals_to_codon_mask(codon_starts_xs, intervals)
                d_obs, n_in, n_out = motif_delta_median(S_clip, mask)
                d_obs2, p, deltas = permutation_test_circular_shift(
                    S_clip,
                    mask,
                    n_perm=args.motif_tests_nperm,
                    seed=args.motif_tests_seed,
                )

                # Write summary
                fsum.write("\t".join([
                    "all",
                    mname,
                    str(n_in),
                    str(n_out),
                    f"{d_obs2:.6f}" if d_obs2 == d_obs2 else "NA",
                    f"{p:.6g}" if p == p else "NA",
                    str(len(deltas)),
                ]) + "\n")

                # Plot null distribution
                if deltas:
                    plt.figure(figsize=(10, 4))
                    plt.hist(deltas, bins=40)
                    plt.axvline(d_obs2, linestyle="--")
                    plt.xlabel("Δ median(SLD_in) − median(SLD_out)")
                    plt.ylabel("Count")
                    plt.title(f"Motif depletion test (circular shift): {mname}  p={p:.3g}")
                    plt.tight_layout()
                    outpng = f"{outpref}.null.{mname}.png"
                    plt.savefig(outpng, dpi=200)
                    plt.close()
                    print(f"Wrote plot: {outpng}")

        print(f"Wrote motif test summary: {summary_path}")

    # Clade-stratified SLD plots (optional)
    if args.clade_plots:
        wanted = [c.strip().upper() for c in args.clades.split(",") if c.strip()]
        # Map headers->seq for non-reference only
        nonref_headers = [h for i, h in enumerate(headers) if i != hxb2_idx]
        nonref_seqs = [s for i, s in enumerate(seqs) if i != hxb2_idx]

        for cl in wanted:
            idxs = [i for i, h in enumerate(nonref_headers) if infer_major_clade(h) == cl]
            if not idxs:
                print(f"No sequences found for clade {cl}; skipping clade plot.")
                continue
            seqs_cl = [nonref_seqs[i] for i in idxs]
            metrics_cl = compute_metrics_for_seqs(seqs_cl, args.min_used_clade)
            S_cl = [m.SLD for m in metrics_cl]
            S_cl_clip = [min(1.0, max(0.0, v)) if v == v else v for v in S_cl]

            plot_track(
                S_cl_clip,
                "Synonymous-leaning diversity (0-1)",
                f"SLD (clade {cl})",
                f"{prefix}.SLD.clade{cl}.png",
                ylim=(0, 1),
            )

            plot_entropy_decomposition(
                metrics_cl,
                f"{prefix}.entropy_decomposition.clade{cl}.png",
                f"Codon vs AA entropy decomposition (clade {cl})",
            )

            motif_map = None
            if args.motifs_json is not None:
                motif_map = load_motif_intervals(
                    args.motifs_json,
                    window_hxb2_start=args.hxb2_start,
                    beg_is_1based=args.motif_beg_1based,
                    end_is_inclusive=args.motif_end_inclusive,
                    major_clade=cl,
                )

            # Motif overlays per clade (optional)
            if motif_map is not None:
                if args.motif_types is None or not args.motif_types.strip():
                    motif_types = sorted(motif_map.keys())
                else:
                    motif_types = [m.strip() for m in args.motif_types.split(",") if m.strip()]

                for mname in motif_types:
                    intervals = motif_map.get(mname, [])
                    if not intervals:
                        continue
                    if args.motif_merge:
                        intervals = merge_intervals(intervals)
                    out_m = f"{prefix}.SLD.clade{cl}.{mname}.png"
                    plot_sld_with_motifs(S_cl_clip, mname, intervals, out_m)

            # Motif depletion tests per clade (optional)
            if args.motif_tests and motif_map is not None:
                # Determine motif types to test (same list as global)
                if args.motif_types is None or not args.motif_types.strip():
                    motif_types_test = sorted(motif_map.keys())
                else:
                    motif_types_test = [m.strip() for m in args.motif_types.split(",") if m.strip()]

                outpref = args.motif_tests_out
                if outpref is None:
                    outpref = f"{prefix}.motif_tests"
                os.makedirs(os.path.dirname(outpref) or ".", exist_ok=True)
                summary_path = f"{outpref}.summary.tsv"

                # Append to the same summary file if it already exists; otherwise create with header.
                write_header = not os.path.exists(summary_path)
                with open(summary_path, "a") as fsum:
                    if write_header:
                        fsum.write("\t".join([
                            "scope",
                            "motif",
                            "n_in",
                            "n_out",
                            "delta_median_in_minus_out",
                            "p_one_sided_depletion",
                            "n_perm_effective",
                        ]) + "\n")

                    for mname in motif_types_test:
                        intervals = motif_map.get(mname, [])
                        if not intervals:
                            continue
                        if args.motif_merge:
                            intervals = merge_intervals(intervals)

                        mask = intervals_to_codon_mask(codon_starts_xs, intervals)
                        d_obs2, p, deltas = permutation_test_circular_shift(
                            S_cl_clip,
                            mask,
                            n_perm=args.motif_tests_nperm,
                            seed=args.motif_tests_seed,
                        )
                        # counts for reporting
                        _, n_in, n_out = motif_delta_median(S_cl_clip, mask)

                        fsum.write("\t".join([
                            f"clade{cl}",
                            mname,
                            str(n_in),
                            str(n_out),
                            f"{d_obs2:.6f}" if d_obs2 == d_obs2 else "NA",
                            f"{p:.6g}" if p == p else "NA",
                            str(len(deltas)),
                        ]) + "\n")

                        if deltas:
                            plt.figure(figsize=(10, 4))
                            plt.hist(deltas, bins=40)
                            plt.axvline(d_obs2, linestyle="--")
                            plt.xlabel("Δ median(SLD_in) − median(SLD_out)")
                            plt.ylabel("Count")
                            plt.title(f"Motif depletion test (circular shift): clade {cl}, {mname}  p={p:.3g}")
                            plt.tight_layout()
                            outpng = f"{outpref}.null.clade{cl}.{mname}.png"
                            plt.savefig(outpng, dpi=200)
                            plt.close()
                            print(f"Wrote plot: {outpng}")

    plot_entropy_decomposition(
        metrics,
        f"{prefix}.entropy_decomposition.png",
        "Codon vs AA entropy decomposition",
    )

    plot_entropy_decomposition_no_sld(
        metrics,
        f"{prefix}.entropy_decomposition.noSLD.png",
        "Codon vs AA entropy decomposition (no SLD)",
    )

    plot_entropy_decomposition_no_sld(
        metrics,
        f"{prefix}.entropy_decomposition.noSLD.pdf",
        "Codon vs AA entropy decomposition (no SLD)",
    )


if __name__ == "__main__":
    main()