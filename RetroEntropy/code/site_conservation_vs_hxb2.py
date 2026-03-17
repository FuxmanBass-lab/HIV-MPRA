#!/usr/bin/env python3
"""
Compute per-site conservation in an aligned region FASTA relative to HXB2.

Conservation definition here:
- For each alignment column, take the HXB2 base (could be A/C/G/T or '-')
- Among sequences with a comparable base (by default exclude gaps and Ns),
  compute fraction that match the HXB2 base.

Also reports:
- base counts (A/C/G/T/N/-)
- Shannon entropy over A/C/G/T (excluding gaps and Ns by default)
- HXB2 coordinate for that column (based on provided start coord OR inferred from the HXB2 slice itself)
- Optional codon-level amino-acid conservation vs HXB2 (requires --env-start and --aa-track)

python code/site_conservation_vs_hxb2.py   --aln results/windows_codon/env1_HXB2_7431_7634.aln.fasta   --hxb2-start 7431   --env-start 6225   --out results/conservation/env1_conservation.tsv   --plot --include-gaps --codon-track --aa-track
python code/site_conservation_vs_hxb2.py   --aln results/windows_codon/env2_HXB2_7761_7862.aln.fasta   --hxb2-start 7761   --env-start 6225   --out results/conservation/env2_conservation.tsv   --plot --include-gaps --codon-track --aa-track

"""

import argparse
import math
import re
from collections import Counter
from typing import List, Tuple, Dict
import os
from pathlib import Path

# matplotlib is optional unless --plot is used

VALID = set("ACGTN-")

def read_fasta(path: str) -> List[Tuple[str, str]]:
    recs = []
    h = None
    buf = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if h is not None:
                    recs.append((h, "".join(buf).upper()))
                h = line[1:].strip()
                buf = []
            else:
                buf.append(line.strip())
        if h is not None:
            recs.append((h, "".join(buf).upper()))
    return recs

def find_hxb2(records: List[Tuple[str, str]]) -> Tuple[str, str]:
    best = None
    for h, s in records:
        score = 0
        if re.search(r"\bHXB2\b", h, re.IGNORECASE): score += 2
        if re.search(r"\bK03455\b", h, re.IGNORECASE): score += 2
        if re.search(r"\bIIIB\b", h, re.IGNORECASE): score += 1
        if score > 0 and (best is None or score > best[0]):
            best = (score, h, s)
    if best is None:
        raise RuntimeError("Could not find HXB2 record (looked for 'HXB2'/'K03455' in headers).")
    return best[1], best[2]



# Helper to load clade map TSV
def load_clade_map_tsv(path: str) -> Dict[str, str]:
    """Load a TSV with header containing at least 'seq_id' and 'clade' columns."""
    mapping: Dict[str, str] = {}
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        cols = {name: i for i, name in enumerate(header)}
        if "seq_id" not in cols or "clade" not in cols:
            raise RuntimeError("--clade-map TSV must have columns 'seq_id' and 'clade'.")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) <= max(cols["seq_id"], cols["clade"]):
                continue
            sid = parts[cols["seq_id"]].strip()
            cl = parts[cols["clade"]].strip()
            if sid and cl:
                mapping[sid] = cl
    return mapping


def infer_major_clade_from_header(header: str) -> str:
    """Infer major clade (A/B/C) from a LANL-style header.

    Uses the first token before whitespace, then takes the prefix before the first '.'
    (e.g., 'A1', 'A', 'B', 'C'). Returns 'A' for any prefix starting with 'A'.
    Returns '' if no A/B/C can be inferred.
    """
    token = header.split()[0]
    prefix = token.split(".")[0].upper()
    if prefix.startswith("A"):
        return "A"
    if prefix.startswith("B"):
        return "B"
    if prefix.startswith("C"):
        return "C"
    return ""

def shannon_entropy_acgt(counts: Dict[str,int]) -> float:
    total = sum(counts.get(b, 0) for b in "ACGT")
    if total == 0:
        return float("nan")
    H = 0.0
    for b in "ACGT":
        c = counts.get(b, 0)
        if c:
            p = c / total
            H -= p * math.log2(p)
    return H


def translate_codon(codon: str) -> str:
    """Translate a 3-nt codon to an amino acid using the standard genetic code.

    Returns:
      - single-letter AA for valid A/C/G/T codons
      - 'X' if any base is not A/C/G/T (including N or -)
    """
    codon = codon.upper()
    if len(codon) != 3:
        return "X"
    if set(codon) <= set("ACGT"):
        table = {
            "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
            "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
            "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
            "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
            "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
            "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
            "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
            "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
            "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
            "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
            "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
            "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
            "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
            "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
            "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
            "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
        }
        return table.get(codon, "X")
    return "X"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aln", required=True, help="Aligned FASTA for a window (e.g. env1...aln.fasta).")
    ap.add_argument("--hxb2-start", type=int, required=True,
                    help="HXB2 coordinate of the first ungapped HXB2 base in THIS window (env1=7432, env2=7762).")
    ap.add_argument("--out", required=True, help="Output TSV path.")
    ap.add_argument("--include-n", action="store_true",
                    help="If set, include 'N' bases in denominator (default excludes N).")
    ap.add_argument("--include-gaps", action="store_true",
                    help="If set, include gaps '-' in denominator (default excludes gaps).")
    ap.add_argument("--plot", action="store_true",
                    help="If set, save plots (PNG) of conservation metrics vs HXB2 coordinate.")
    ap.add_argument("--plot-prefix", default=None,
                    help="Prefix for plot output files. Default: <out without extension>.")
    ap.add_argument("--env-start", type=int, default=None,
                    help="HXB2 coordinate of the first base of the ENV start codon (ATG). Required for --codon-track.")
    ap.add_argument("--codon-track", action="store_true",
                    help="If set, overlay a codon-position track (1/2/3) beneath the match_fraction plot.")
    ap.add_argument("--aa-track", action="store_true",
                    help="If set, compute codon-level amino-acid conservation vs HXB2 and save additional plots/TSV. Requires --env-start.")
    ap.add_argument("--aa-out", default=None,
                    help="Output TSV path for amino-acid conservation. Default: <out without extension>.aa_conservation.tsv")
    ap.add_argument("--clade-map", default=None,
                    help="TSV mapping sequence IDs to clade labels. Must have columns 'seq_id' and 'clade'. If set, also write per-clade normalized entropy plots.")
    ap.add_argument("--clade-min-n", type=int, default=100,
                    help="Minimum number of sequences required in a clade to plot it (default: 100).")
    ap.add_argument("--clade-from-header", action="store_true",
                    help="Infer major clade from the FASTA header (first token before whitespace, prefix before first '.'). E.g., A1/A2/...->A. Only A/B/C are used.")
    ap.add_argument("--major-clades", default="A,B,C",
                    help="Comma-separated major clades to include for clade plots when using --clade-from-header (default: A,B,C).")
    args = ap.parse_args()

    records = read_fasta(args.aln)
    if not records:
        raise RuntimeError("No records found.")

    lens = {len(s) for _, s in records}
    if len(lens) != 1:
        raise RuntimeError(f"Not a strict alignment (multiple lengths found): {sorted(lens)[:10]} ...")
    L = next(iter(lens))

    hxb2_header, hxb2 = find_hxb2(records)

    # Pre-clean sequences to valid alphabet
    seqs = []
    headers = []
    for h, s in records:
        s = s.upper()
        s = "".join(ch if ch in VALID else "N" for ch in s)
        headers.append(h)
        seqs.append(s)

    clade_to_indices: Dict[str, List[int]] = {}
    if args.clade_map is not None and args.clade_from_header:
        raise RuntimeError("Use either --clade-map or --clade-from-header, not both.")

    if args.clade_map is not None:
        clade_map = load_clade_map_tsv(args.clade_map)
        missing = 0
        for idx, h in enumerate(headers):
            sid = h.split()[0]
            cl = clade_map.get(sid)
            if cl is None:
                missing += 1
                continue
            clade_to_indices.setdefault(cl, []).append(idx)
        if missing > 0:
            print(f"Warning: {missing} sequences had no clade mapping (ignored for clade plots).")

    elif args.clade_from_header:
        keep = {c.strip().upper() for c in args.major_clades.split(",") if c.strip()}
        missing = 0
        for idx, h in enumerate(headers):
            cl = infer_major_clade_from_header(h)
            if not cl or cl not in keep:
                missing += 1
                continue
            clade_to_indices.setdefault(cl, []).append(idx)
        if missing > 0:
            print(f"Warning: {missing} sequences did not map to requested major clades ({sorted(keep)}); ignored for clade plots.")

    # filter by minimum size (applies to either mapping mode)
    if clade_to_indices:
        clade_to_indices = {cl: idxs for cl, idxs in clade_to_indices.items() if len(idxs) >= args.clade_min_n}
        if not clade_to_indices:
            print(f"Warning: no clades meet --clade-min-n={args.clade_min_n}; no per-clade plots will be produced.")

    # Map alignment column -> HXB2 coordinate (only where HXB2 base is not a gap)
    # For gap columns in HXB2, coord is NA.
    coord = args.hxb2_start
    col_to_coord = [None] * L
    for i, b in enumerate(hxb2):
        if b != "-":
            col_to_coord[i] = coord
            coord += 1

    coord_to_col = {}
    for col, c in enumerate(col_to_coord):
        if c is not None:
            coord_to_col[c] = col

    with open(args.out, "w") as out:
        out.write("\t".join([
            "aln_col_1based",
            "hxb2_coord",
            "hxb2_base",
            "n_total_seqs",
            "n_used",
            "match_count",
            "match_fraction",
            "entropy_acgt",
            "count_A","count_C","count_G","count_T","count_N","count_gap"
        ]) + "\n")

        nseq = len(seqs)

        plot_coords = []
        plot_match_frac = []
        plot_entropy = []
        plot_codon_pos = []  # 1/2/3 per coordinate (requires --env-start)
        plot_cols = []  # alignment column indices corresponding to plot_coords

        for i in range(L):
            ref = hxb2[i]
            col = [s[i] for s in seqs]

            counts = Counter(col)
            # normalize counts presence
            for k in list(counts.keys()):
                if k not in VALID:
                    counts["N"] += counts[k]
                    del counts[k]

            # Determine which bases contribute to the match calculation
            def usable(b: str) -> bool:
                if (b == "-") and (not args.include_gaps):
                    return False
                if (b == "N") and (not args.include_n):
                    return False
                return True

            used = [b for b in col if usable(b)]
            n_used = len(used)

            # match: base equals HXB2 base (note: if ref is '-', matching gaps is only counted if include_gaps)
            match = sum(1 for b in used if b == ref)

            frac = (match / n_used) if n_used > 0 else float("nan")
            ent = shannon_entropy_acgt(counts)

            hxb2_coord = col_to_coord[i]
            if hxb2_coord is not None:
                plot_coords.append(hxb2_coord)
                plot_cols.append(i)

                if frac == frac:  # not NaN
                    plot_match_frac.append(frac)
                else:
                    plot_match_frac.append(float('nan'))

                if ent == ent:
                    plot_entropy.append(ent)
                else:
                    plot_entropy.append(float('nan'))

                if args.env_start is not None:
                    plot_codon_pos.append(((hxb2_coord - args.env_start) % 3) + 1)
                else:
                    plot_codon_pos.append(None)

            out.write("\t".join(map(str, [
                i + 1,
                hxb2_coord if hxb2_coord is not None else "NA",
                ref,
                nseq,
                n_used,
                match,
                f"{frac:.6f}" if frac == frac else "NA",
                f"{ent:.6f}" if ent == ent else "NA",
                counts.get("A",0),
                counts.get("C",0),
                counts.get("G",0),
                counts.get("T",0),
                counts.get("N",0),
                counts.get("-",0),
            ])) + "\n")

    print(f"Wrote: {args.out}")
    print(f"Reference used: {hxb2_header}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError(f"Plotting requested (--plot) but matplotlib is not available: {e}")

        prefix = args.plot_prefix
        if prefix is None:
            # default: strip extension from args.out
            prefix = os.path.splitext(args.out)[0]

        # Ensure output directory exists
        Path(os.path.dirname(prefix) or '.').mkdir(parents=True, exist_ok=True)

        if args.codon_track and args.env_start is None:
            raise RuntimeError("--codon-track requires --env-start (HXB2 coordinate of ENV start codon ATG).")
        if args.aa_track and args.env_start is None:
            raise RuntimeError("--aa-track requires --env-start (HXB2 coordinate of ENV start codon ATG).")

        aa_out = args.aa_out
        if aa_out is None:
            aa_out = f"{prefix}.aa_conservation.tsv"

        # Filter out NaNs for plotting (matplotlib can handle NaNs but filtering makes axes nicer)
        xs = []
        ys = []
        cps = []
        for x, y, cp in zip(plot_coords, plot_match_frac, plot_codon_pos):
            if y == y:  # not NaN
                xs.append(x)
                ys.append(y)
                cps.append(cp)

        plt.figure(figsize=(24, 4))
        if xs:
            plt.plot(xs, ys)
            if args.codon_track:
                # Draw codon-position track as markers below the curve
                y_track = -0.06
                x1 = [x for x, cp in zip(xs, cps) if cp == 1]
                x2 = [x for x, cp in zip(xs, cps) if cp == 2]
                x3 = [x for x, cp in zip(xs, cps) if cp == 3]
                if x1:
                    plt.scatter(x1, [y_track] * len(x1), marker='o', s=12, label='codon pos 1')
                if x2:
                    plt.scatter(x2, [y_track] * len(x2), marker='s', s=12, label='codon pos 2')
                if x3:
                    plt.scatter(x3, [y_track] * len(x3), marker='^', s=12, label='codon pos 3')
        if args.codon_track:
            plt.ylim(-0.12, 1)
        else:
            plt.ylim(0, 1)
        plt.xlabel("HXB2 coordinate")
        plt.ylabel("Fraction matching HXB2")
        plt.title("Per-site conservation vs HXB2")
        if args.codon_track:
            plt.legend(loc='lower right', frameon=False)
        out1 = f"{prefix}.match_fraction.png"
        plt.tight_layout()
        plt.savefig(out1, dpi=200)
        plt.close()

        xs2 = []
        ys2 = []
        for x, y in zip(plot_coords, plot_entropy):
            if y == y:  # not NaN
                xs2.append(x)
                ys2.append(y)

        plt.figure(figsize=(24, 4))
        if xs2:
            plt.plot(xs2, ys2)
        plt.xlabel("HXB2 coordinate")
        plt.ylabel("Shannon entropy (ACGT)")
        plt.title("Per-site entropy vs HXB2")
        out2 = f"{prefix}.entropy.png"
        plt.tight_layout()
        plt.savefig(out2, dpi=200)
        plt.close()

        print(f"Wrote plot: {out1}")
        print(f"Wrote plot: {out2}")


        if args.aa_track:
            # Compute codon-level AA conservation vs HXB2
            # Codon starts are HXB2 coords where (coord - env_start) % 3 == 0 and coord+1, coord+2 exist.
            # We only use codons that are fully represented in the HXB2 coordinate mapping.
            codon_starts = []
            # Use the plotted coordinate range to keep outputs focused
            if plot_coords:
                cmin = min(plot_coords)
                cmax = max(plot_coords)
            else:
                cmin = None
                cmax = None

            # Build a sorted list of available coordinates for deterministic iteration
            available_coords = sorted(coord_to_col.keys())
            for c in available_coords:
                if ((c - args.env_start) % 3) != 0:
                    continue
                if (c + 1) not in coord_to_col or (c + 2) not in coord_to_col:
                    continue
                if cmin is not None and (c < cmin or (c + 2) > cmax):
                    continue
                codon_starts.append(c)

            # Helper to get codon string for a sequence at a codon start coordinate
            def get_codon(seq: str, start_coord: int) -> str:
                i0 = coord_to_col[start_coord]
                i1 = coord_to_col[start_coord + 1]
                i2 = coord_to_col[start_coord + 2]
                return seq[i0] + seq[i1] + seq[i2]

            aa_plot_x = []
            aa_match_frac = []
            aa_change_frac = []
            aa_entropy = []
            codon3_entropy = []  # nucleotide entropy of 3rd base in each codon (ACGT only)

            with open(aa_out, "w") as fout:
                fout.write("\t".join([
                    "hxb2_codon_start",
                    "env_codon_pos",  # 1-based codon index in env
                    "hxb2_codon",
                    "hxb2_aa",
                    "n_total_seqs",
                    "n_used",
                    "match_count",
                    "match_fraction",
                    "change_fraction",
                    "aa_entropy",
                ]) + "\n")

                for c in codon_starts:
                    ref_codon = get_codon(hxb2, c)
                    ref_aa = translate_codon(ref_codon)

                    used = 0
                    match = 0
                    aa_counts = Counter()

                    for seq in seqs:
                        cod = get_codon(seq, c)
                        aa = translate_codon(cod)

                        # Exclude codons with ambiguous/gap for AA conservation (aa == 'X')
                        if aa == "X":
                            continue
                        used += 1
                        aa_counts[aa] += 1
                        if aa == ref_aa:
                            match += 1

                    frac = (match / used) if used > 0 else float("nan")
                    change = (1.0 - frac) if frac == frac else float("nan")

                    if used > 0:
                        aa_ent = 0.0
                        for aa_sym, cnt in aa_counts.items():
                            p = cnt / used
                            aa_ent -= p * math.log2(p)
                    else:
                        aa_ent = float("nan")

                    env_codon_index = ((c - args.env_start) // 3) + 1

                    # Entropy of 3rd base in the codon (coordinate c+2)
                    col3 = coord_to_col[c + 2]
                    counts3 = Counter(seqs[j][col3] for j in range(len(seqs)))
                    ent3 = shannon_entropy_acgt(counts3)

                    fout.write("\t".join(map(str, [
                        c,
                        env_codon_index,
                        ref_codon,
                        ref_aa,
                        len(seqs),
                        used,
                        match,
                        f"{frac:.6f}" if frac == frac else "NA",
                        f"{change:.6f}" if change == change else "NA",
                        f"{aa_ent:.6f}" if aa_ent == aa_ent else "NA",
                    ])) + "\n")

                    aa_plot_x.append(c)
                    aa_match_frac.append(frac)
                    aa_change_frac.append(change)
                    aa_entropy.append(aa_ent)
                    codon3_entropy.append(ent3)

            print(f"Wrote AA conservation TSV: {aa_out}")

            # Plot AA match fraction
            xs3 = [x for x, y in zip(aa_plot_x, aa_match_frac) if y == y]
            ys3 = [y for y in aa_match_frac if y == y]
            plt.figure(figsize=(24, 4))
            if xs3:
                plt.plot(xs3, ys3)
            plt.ylim(0, 1)
            plt.xlabel("HXB2 coordinate (codon start)")
            plt.ylabel("AA fraction matching HXB2")
            plt.title("Codon-level amino-acid conservation vs HXB2")
            out3 = f"{prefix}.aa_match_fraction.png"
            plt.tight_layout()
            plt.savefig(out3, dpi=200)
            plt.close()

            print(f"Wrote plot: {out3}")

            # Plot AA entropy
            xs6 = [x for x, y in zip(aa_plot_x, aa_entropy) if y == y]
            ys6 = [y for y in aa_entropy if y == y]
            plt.figure(figsize=(24, 4))
            if xs6:
                plt.plot(xs6, ys6)
            plt.xlabel("HXB2 coordinate (codon start)")
            plt.ylabel("Shannon entropy (AA)")
            plt.title("Codon-level amino-acid entropy vs HXB2")
            out6 = f"{prefix}.aa_entropy.png"
            plt.tight_layout()
            plt.savefig(out6, dpi=200)
            plt.close()

            print(f"Wrote plot: {out6}")

            # Plot nucleotide entropy at the 3rd base of each codon
            xs9 = [x for x, y in zip(aa_plot_x, codon3_entropy) if y == y]
            ys9 = [y for y in codon3_entropy if y == y]
            plt.figure(figsize=(24, 4))
            if xs9:
                plt.plot(xs9, ys9)
            plt.xlabel("HXB2 coordinate (codon start)")
            plt.ylabel("Shannon entropy (ACGT) at codon position 3")
            plt.title("Nucleotide entropy of 3rd codon base vs HXB2")
            out9 = f"{prefix}.codon3_entropy.png"
            plt.tight_layout()
            plt.savefig(out9, dpi=200)
            plt.close()

            print(f"Wrote plot: {out9}")

            # Combined entropy plot: nucleotide entropy (per base) and AA entropy (per codon)
            nuc_x = [x for x, y in zip(plot_coords, plot_entropy) if y == y]
            nuc_y = [y for y in plot_entropy if y == y]

            aa_xe = [x for x, y in zip(aa_plot_x, aa_entropy) if y == y]
            aa_ye = [y for y in aa_entropy if y == y]

            plt.figure(figsize=(24, 4))
            if nuc_x:
                plt.plot(nuc_x, nuc_y, label="nucleotide entropy")
            if aa_xe:
                plt.plot(aa_xe, aa_ye, label="AA entropy (codon)")
            plt.xlabel("HXB2 coordinate")
            plt.ylabel("Shannon entropy")
            plt.title("Nucleotide vs amino-acid entropy vs HXB2")
            plt.legend(loc='upper right', frameon=False)
            out7 = f"{prefix}.entropy_nuc_vs_aa.png"
            plt.tight_layout()
            plt.savefig(out7, dpi=200)
            plt.close()

            print(f"Wrote plot: {out7}")

            # Combined NORMALIZED entropy plot: nucleotide entropy and AA entropy scaled to [0,1]
            nuc_y_norm = [ (y / 2.0) for y in nuc_y ]
            aa_max = math.log2(20)
            aa_ye_norm = [ (y / aa_max) for y in aa_ye ]

            plt.figure(figsize=(24, 4))
            if nuc_x:
                plt.plot(nuc_x, nuc_y_norm, label="nucleotide entropy (normalized)")
            if aa_xe:
                plt.plot(aa_xe, aa_ye_norm, label="AA entropy (normalized)")
            plt.ylim(0, 1)
            plt.xlabel("HXB2 coordinate")
            plt.ylabel("Normalized Shannon entropy")
            plt.title("Normalized nucleotide vs amino-acid entropy vs HXB2")
            plt.legend(loc='upper right', frameon=False)
            out8 = f"{prefix}.entropy_nuc_vs_aa.normalized.png"
            plt.tight_layout()
            plt.savefig(out8, dpi=200)
            plt.close()

            print(f"Wrote plot: {out8}")

            # Per-clade normalized entropy plots (optional)
            if (args.clade_map is not None or args.clade_from_header) and clade_to_indices:
                # Precompute per-clade nucleotide normalized entropy at each plotted coordinate
                # (uses plot_cols/plot_coords to stay aligned with the window)
                # We compute entropy directly per clade to avoid NA filtering differences.

                for clade, idxs in sorted(clade_to_indices.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                    # Nucleotide entropy per base
                    nuc_vals = []
                    nuc_xs = []
                    for col, coord in zip(plot_cols, plot_coords):
                        # build counts for this clade at this alignment column
                        counts_c = Counter(seqs[j][col] for j in idxs)
                        ent_c = shannon_entropy_acgt(counts_c)
                        if ent_c == ent_c:
                            nuc_xs.append(coord)
                            nuc_vals.append(ent_c / 2.0)

                    # AA entropy per codon start
                    aa_vals = []
                    aa_xs = []
                    aa_max = math.log2(20)
                    for c, ent_aa in zip(aa_plot_x, aa_entropy):
                        # recompute AA entropy per clade for this codon start (exclude X)
                        aa_counts_c = Counter()
                        used_c = 0
                        for j in idxs:
                            cod = get_codon(seqs[j], c)
                            aa_sym = translate_codon(cod)
                            if aa_sym == "X":
                                continue
                            used_c += 1
                            aa_counts_c[aa_sym] += 1
                        if used_c > 0:
                            Hc = 0.0
                            for aa_sym, cnt in aa_counts_c.items():
                                p = cnt / used_c
                                Hc -= p * math.log2(p)
                            aa_xs.append(c)
                            aa_vals.append(Hc / aa_max)

                    # Plot per clade
                    plt.figure(figsize=(24, 4))
                    if nuc_xs:
                        plt.plot(nuc_xs, nuc_vals, label=f"{clade} nucleotide entropy (norm)")
                    if aa_xs:
                        plt.plot(aa_xs, aa_vals, label=f"{clade} AA entropy (norm)")
                    plt.ylim(0, 1)
                    plt.xlabel("HXB2 coordinate")
                    plt.ylabel("Normalized Shannon entropy")
                    plt.title(f"Normalized nucleotide vs amino-acid entropy vs HXB2 (clade {clade}, n={len(idxs)})")
                    plt.legend(loc='upper right', frameon=False)
                    out_cl = f"{prefix}.entropy_nuc_vs_aa.normalized.{clade}.png"
                    plt.tight_layout()
                    plt.savefig(out_cl, dpi=200)
                    plt.close()

                    print(f"Wrote plot: {out_cl}")

            # Combined plot: nucleotide base-change fraction AND amino-acid change fraction
            # Base-change is computed per HXB2 coordinate as (1 - base match fraction)
            base_x = [x for x, y in zip(plot_coords, plot_match_frac) if y == y]
            base_y = [(1.0 - y) for y in plot_match_frac if y == y]

            aa_x = [x for x, y in zip(aa_plot_x, aa_change_frac) if y == y]
            aa_y = [y for y in aa_change_frac if y == y]

            plt.figure(figsize=(24, 4))
            if base_x:
                plt.plot(base_x, base_y, label="base change (1 - match)")
            if aa_x:
                plt.plot(aa_x, aa_y, label="AA change (codon)")
            plt.ylim(0, 1)
            plt.xlabel("HXB2 coordinate")
            plt.ylabel("Change fraction")
            plt.title("Base vs amino-acid change fraction vs HXB2")
            plt.legend(loc='upper right', frameon=False)
            out5 = f"{prefix}.base_vs_aa_change.png"
            plt.tight_layout()
            plt.savefig(out5, dpi=200)
            plt.close()

            print(f"Wrote plot: {out5}")

if __name__ == "__main__":
    main()