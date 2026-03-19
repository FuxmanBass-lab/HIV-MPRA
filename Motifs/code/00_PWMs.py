#!/usr/bin/env python3
"""
Filter JASPAR MEME to a TF panel and generate a subset MEME plus a PDF of logos.

Usage:
  python 00_PWMs.py \
    --meme ../data/jaspar2024_human_core.meme \
    --tfs  ../data/TFs.tsv \
    --out-meme ../results/jaspar_panel.meme \
    --out-pdf  ../results/jaspar_panel_logos.pdf \
    [--bg A 0.362054 C 0.176903 G 0.239302 T 0.221741]

Notes
- Excludes any motif whose *gene token* contains lowercase letters (likely mouse).
- Keeps only motifs whose gene is in the TF panel TSV. Optional filters:
  * --min_ic : drop motifs with total IC below threshold (vs background)
  * --exclude_genes : drop specific TF symbols (e.g., RELB)
  * --exclude_motifs : drop specific JASPAR IDs (e.g., MA1509.1, MA0137.1)
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D

# Biopython MEME parser supports MEME v4 text format
# from Bio import motifs

# -----------------
# Helpers
# -----------------

DNA = ['A','C','G','T']
DEFAULT_BG = {'A':0.25,'C':0.25,'G':0.25,'T':0.25}

GENE_RE = re.compile(r"^[A-Za-z0-9_.:+-]+$")


def parse_tf_panel(path: Path) -> Tuple[Dict[str,List[str]], set]:
    fam2genes: Dict[str,List[str]] = {}
    genes: set = set()
    with open(path) as f:
        header = f.readline().strip().split('\t')
        # Expect header like: family\tTFs
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            fam, tfs = parts[0], parts[1]
            tf_list = [t.strip() for t in re.split(r"[ ,]", tfs.replace(',', ' ')) if t.strip()]
            fam2genes[fam] = tf_list
            for g in tf_list:
                genes.add(g)
    return fam2genes, genes


def get_gene_from_motif_name(mname: str) -> str:
    """
    JASPAR MEME names are typically like: "MA0139.1 CTCF" or "MA0106.3 RELA".
    Return the gene token (2nd token) if present, else best-effort last token.
    """
    toks = mname.strip().split()
    if len(toks) >= 2:
        return toks[1]
    # some files may have just the gene name
    return toks[0] if toks else ''


def get_motif_id(mname: str) -> str:
    """Return motif accession like 'MA0105.1' from the MEME name line if present."""
    toks = mname.strip().split()
    return toks[0] if toks else ''


def has_lowercase_gene(gene: str) -> bool:
    return any(c.islower() for c in gene)


class SimplePWM:
    def __init__(self, name: str, pwm: Dict[str, List[float]]):
        self.name = name
        self.pwm = pwm
        self.length = len(next(iter(pwm.values()))) if pwm else 0

    def __len__(self):
        return self.length


def _parse_matrix_width(line: str) -> int:
    # Expect: letter-probability matrix: alength= 4 w= 10 nsites= ...
    m = re.search(r"\bw=\s*(\d+)", line)
    if not m:
        raise ValueError(f"Cannot find width in line: {line.strip()}")
    return int(m.group(1))


def meme_iter(path: Path):
    """Parse MEME v4 text format motifs robustly without Biopython.
    Yields SimplePWM objects with attributes .name, .pwm, .length.
    """
    with open(path) as fh:
        lines = fh.readlines()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip('\n')
        if line.startswith('MOTIF '):
            name = line.split('MOTIF ', 1)[1].strip()
            # advance to matrix header
            i += 1
            while i < n and 'letter-probability matrix' not in lines[i]:
                i += 1
            if i >= n:
                break
            w = _parse_matrix_width(lines[i])
            i += 1
            # read w rows with 4 floats each (A C G T order)
            rows: List[List[float]] = []
            while i < n and len(rows) < w:
                s = lines[i].strip()
                if not s:
                    i += 1
                    continue
                # Some MEME files put comments/URLs between motifs; skip non-numeric
                parts = re.split(r"\s+", s)
                try:
                    vals = [float(x) for x in parts[:4]]
                except ValueError:
                    # skip lines that don't start with numbers
                    i += 1
                    continue
                if len(vals) == 4:
                    rows.append(vals)
                i += 1
            if len(rows) != w:
                # best-effort: truncate/skip inconsistent motif
                i += 1
                continue
            pwm: Dict[str, List[float]] = {b: [] for b in DNA}
            for r in rows:
                for b, v in zip(DNA, r):
                    pwm[b].append(v)
            yield SimplePWM(name, pwm)
        else:
            i += 1


def write_meme_subset(out_path: Path, mot_list: List[SimplePWM], bg: Dict[str,float]) -> None:
    with open(out_path, 'w') as out:
        out.write("MEME version 4\n\n")
        out.write("ALPHABET= ACGT\n\n")
        out.write("strands: + -\n\n")
        out.write("Background letter frequencies\n")
        out.write(f"A {bg['A']:.6f} C {bg['C']:.6f} G {bg['G']:.6f} T {bg['T']:.6f}\n\n")
        for m in mot_list:
            name = m.name or "UNKNOWN"
            pwm = m.pwm
            w = m.length
            out.write(f"MOTIF {name}\n")
            out.write(f"letter-probability matrix: alength= 4 w= {w} nsites= 0 E= 0\n")
            for i in range(w):
                row = [pwm[b][i] for b in DNA]
                out.write(" ".join(f"{x: .6f}" for x in row) + "\n")
            out.write("\n")


def col_ic(p: List[float], bg: Dict[str,float]) -> float:
    # Information per column: sum_b p_b * log2(p_b/bg_b)
    ic = 0.0
    for b, pb in zip(DNA, p):
        if pb > 0 and bg[b] > 0:
            ic += pb * math.log2(pb / bg[b])
    return max(ic, 0.0)


def total_ic_for_motif(m: SimplePWM, bg: Dict[str,float]) -> float:
    pwm = m.pwm
    w = m.length
    tot = 0.0
    for i in range(w):
        p = [pwm[b][i] for b in DNA]
        tot += col_ic(p, bg)
    return tot


def draw_logo(ax, m: SimplePWM, bg: Dict[str,float]) -> float:
    """Draw a crisp DNA sequence logo using vector glyphs (TextPath).
    Heights are p(b,i) * IC(i) and letters are stacked largest on top.
    Returns total IC (sum over columns).
    """
    pwm = m.pwm
    w = m.length
    color = {'A':'#1f77b4','C':'#ff7f0e','G':'#2ca02c','T':'#d62728'}

    # Precompute column information content
    col_ic_vals = [col_ic([pwm[b][i] for b in DNA], bg) for i in range(w)]
    ymax = max(col_ic_vals + [1.0])

    # Axes styling
    ax.set_xlim(0, w)
    ax.set_ylim(0, ymax * 1.05)
    ax.set_xticks(range(w))
    ax.set_xticklabels([str(i+1) for i in range(w)], fontsize=7)
    # y-axis in bits
    import math as _math
    y_max_int = int(_math.ceil(ymax))
    ax.set_yticks(list(range(0, y_max_int + 1)))
    ax.set_yticklabels([str(v) for v in range(0, y_max_int + 1)], fontsize=7)
    ax.set_ylabel('bits', fontsize=8)
    for side in ('top','right'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.5)

    # Prepare a unit text path (approx width/height=1)
    # TextPath size=1.0 gives a glyph in font units; we'll normalize by its bounds
    unit_paths = {b: TextPath((0,0), b, size=1, prop={'weight':'bold'}) for b in DNA}
    # Measure heights to normalize
    bbox_h = {b: (p.get_extents().height or 1.0) for b, p in unit_paths.items()}
    bbox_w = {b: (p.get_extents().width or 1.0) for b, p in unit_paths.items()}

    min_show = 0.02  # hide tiny slivers to keep plot clean

    for i in range(w):
        ic = col_ic_vals[i]
        if ic <= 0:
            continue
        # sort bases by contribution height ascending (smallest at bottom)
        parts = sorted(((pwm[b][i] * ic, b) for b in DNA))
        y = 0.0
        for h, b in parts:
            if h <= min_show:
                continue
            tp = unit_paths[b]
            # scale horizontally so each column spans ~0.9 width
            sx = 0.9 / bbox_w[b]
            # scale vertically to height h
            sy = h / bbox_h[b]
            trans = Affine2D().scale(sx, sy).translate(i + 0.5 - 0.45, y)
            patch = PathPatch(tp, lw=0, fc=color[b], transform=trans + ax.transData)
            ax.add_patch(patch)
            y += h

    return sum(col_ic_vals)


# -----------------
# Main
# -----------------

def main():
    ap = argparse.ArgumentParser(description="Filter JASPAR MEME to TF panel and render logos")
    ap.add_argument('--meme', required=True, type=Path, help='Input JASPAR MEME (text, version 4)')
    ap.add_argument('--tfs', required=True, type=Path, help='TFs.tsv (family\tTFs)')
    ap.add_argument('--out-meme', required=True, type=Path, help='Filtered MEME output path')
    ap.add_argument('--out-pdf', required=True, type=Path, help='Logos PDF output path')
    ap.add_argument('--bg', nargs='+', default=None,
                    help='Optional background like: A 0.362054 C 0.176903 G 0.239302 T 0.221741')
    ap.add_argument('--min_ic', type=float, default=None,
                    help='Exclude motifs with total IC < min_ic (computed vs background).')
    ap.add_argument('--exclude_genes', type=str, default='',
                    help='Comma-separated gene symbols to exclude (e.g., RELB,STAT1).')
    ap.add_argument('--exclude_motifs', type=str, default='',
                    help='Comma-separated motif IDs to exclude (e.g., MA1509.1,MA0137.1).')

    args = ap.parse_args()

    # Background
    bg = DEFAULT_BG.copy()
    if args.bg:
        if len(args.bg) != 8:
            raise SystemExit('Provide bg as 8 tokens: A pA C pC G pG T pT')
        bg = {args.bg[i]: float(args.bg[i+1]) for i in range(0, 8, 2)}
        for b in DNA:
            if b not in bg:
                raise SystemExit('Background must define A,C,G,T')

    exclude_genes = {g.strip() for g in args.exclude_genes.split(',') if g.strip()}
    exclude_motif_ids = {m.strip() for m in args.exclude_motifs.split(',') if m.strip()}

    fam2genes, gene_panel = parse_tf_panel(args.tfs)

    # Build keep set: panel genes, uppercase only
    keep_genes = {g for g in gene_panel if g.isupper()}

    kept: List[SimplePWM] = []
    skipped: List[str] = []
    kept_meta = []  # list of dicts: {'motif': m, 'ic': tot_ic, 'gene': gene, 'id': mid}

    for m in meme_iter(args.meme):
        mname = m.name or ''
        gene = get_gene_from_motif_name(mname)
        mid  = get_motif_id(mname)
        if not gene:
            skipped.append(f"NO_GENE\t{mname}")
            continue
        if has_lowercase_gene(gene):
            skipped.append(f"LOWER\t{mname}")
            continue
        if gene not in keep_genes:
            skipped.append(f"NOTPANEL\t{mname}")
            continue
        if gene in exclude_genes:
            skipped.append(f"EXCLUDE_GENE\t{mname}")
            continue
        if mid in exclude_motif_ids:
            skipped.append(f"EXCLUDE_ID\t{mname}")
            continue
        # IC filter (vs background)
        tot_ic = total_ic_for_motif(m, bg)
        if args.min_ic is not None and tot_ic < args.min_ic:
            skipped.append(f"LOW_IC<{args.min_ic}\t{mname}\tIC={tot_ic:.2f}")
            continue
        # keep motif
        kept.append(m)
        kept_meta.append({'motif': m, 'ic': tot_ic, 'gene': gene, 'id': mid})

    # Write subset MEME
    args.out_meme.parent.mkdir(parents=True, exist_ok=True)
    write_meme_subset(args.out_meme, kept, bg)

    # Also write a stricter PI-filtered panel regardless of CLI excludes
    pi_min_ic = 10.0
    pi_exclude_genes = {'RELB'}
    pi_exclude_ids = {'MA1509.1', 'MA0137.1'}  # IRF6 and STAT1 IDs per PI

    kept_pi = [d['motif'] for d in kept_meta
               if d['ic'] >= pi_min_ic and d['gene'] not in pi_exclude_genes and d['id'] not in pi_exclude_ids]

    pi_out = args.out_meme.parent / 'jaspar_panel_filtered.meme'
    write_meme_subset(pi_out, kept_pi, bg)

    # PDF with logos and IC
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out_pdf) as pdf:
        for m in kept:
            fig, ax = plt.subplots(figsize=(8.5, 2.8))
            tot_ic = total_ic_for_motif(m, bg)
            draw_logo(ax, m, bg)
            title = f"{m.name}  |  length={m.length}  |  total IC={tot_ic:.2f} bits"
            fig.suptitle(title, fontsize=12)
            fig.tight_layout(rect=[0, 0.0, 1, 0.90])
            pdf.savefig(fig)
            plt.close(fig)

    # Also drop a simple report listing skipped reasons (optional, next to outputs)
    rep = args.out_meme.with_suffix('.skip.txt')
    with open(rep, 'w') as f:
        for s in skipped:
            f.write(s + '\n')

    print(f"[ok] Kept {len(kept)} motifs. Wrote: {args.out_meme} and {args.out_pdf}")
    print(f"[ok] PI-filtered panel: {pi_out} (n={len(kept_pi)} motifs; IC>=10, minus RELB, MA1509.1, MA0137.1)")


if __name__ == '__main__':
    main()
