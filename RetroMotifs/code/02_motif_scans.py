#!/usr/bin/env python3
"""
02_motif_scans.py — robust, accurate, and dependency-light motif scanner

Pipeline (per hit)
------------------
A window → passes rel_score → per‑motif NMS → (optional) empirical p‑value → per‑TF dedupe across PWMs →
assign to ONE family (composites + precedence) → cross‑family exclusivity at locus → (optional) FDR → counted.

What’s new / key guarantees
---------------------------
- **Fixed background** for scoring (viral genome‑wide):
  A=0.362054, C=0.176903, G=0.239302, T=0.221741 (overrides MEME header backgrounds).
- **rel_score prefilter**: normalized PSSM score in [0,1]; drop windows below `--rel_score` early.
- **Per‑motif NMS** (`--nms_factor`): suppress same‑motif neighbors by center distance ≈ motif_len×nms_factor.
- **Optional empirical significance** (`--shuffles`, `--p_thresh`):
  p = P(max_rel ≥ observed) under mononucleotide background; cached per (seq_len, motif_len, motif_id).
- **Per‑TF dedupe across PWMs** (`--tf_*`): collapse multiple PWMs and slight shifts for the same TF symbol.
- **Composites collapse**: if ≥2 member families co‑occur for a hit, replace with meta‑family (e.g., AP1/ATF).
- **Precedence‑aware exclusivity across families** (`--family_*`): for overlapping loci within an isolate/tile,
  keep exactly one family by: precedence → higher rel_score → shorter motif → earlier start.
- **Optional BH‑FDR** (`--q_thresh`): Benjamini–Hochberg q‑values computed per isolate/tile over kept hits.

Inputs
------
1) MEME **text** motif file (e.g., ../data/jaspar2024_human_core.meme)
2) TF family map `TFs.tsv` with columns: `family\tTFs` (comma‑separated TF symbols)
3) Optional precedence `family_precedence.tsv` with columns: `family\tpriority` (higher wins)
4) Optional composites `family_composites.tsv` with columns: `meta_family\tmembers` (comma‑separated families)
5) Tile FASTAs: `results/tiles/tile_*.fasta` (one isolate per FASTA entry)

Outputs
-------
- `results/motif_counts/tile_<N>.counts.tsv` : rows = isolates, columns = TF families (incl. composite metas);
- `results/motif_counts/tile_<N>.hits.tsv`   : diagnostics for both kept and dropped hits with fields like
  `p_value`, `q_value` (if enabled), `dedup_kept`, `dedup_reason` (e.g., center/iou/shift/precedence/q).

CLI (common usage)
------------------
python code/02_motif_scans.py \
  --meme data/jaspar2024_human_core.meme \
  --tfs  data/TFs.tsv \
  --tiles_dir results/tiles \
  --outdir results/motif_counts \
  --precedence data/family_precedence.tsv \
  --composites data/family_composites.tsv \
  --rel_score 0.92 \
  --nms_factor 1.0 \
  --tf_dedupe_bp 8 --tf_iou 0.5 --tf_shift_bp 1 \
  --family_dedupe_bp 10 --family_iou 0.5 --family_shift_bp 1 \
  --shuffles 300 --p_thresh 5e-4 --q_thresh 1.0

Notes
-----
- MEME parsing is text‑only; no Biopython required. PSSM log‑odds use the **fixed background** above.
- Precedence exclusivity ensures **one family per site**; adjust `--family_*` to tune overlap strictness.
- To enable FDR, set a finite `--q_thresh` (e.g., 0.1). In FIMO mode we read per-position q-values (or compute BH from p-values); in fallback mode you can enable empirical p-values (`--shuffles`) and BH-FDR is then applied.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from collections import defaultdict, namedtuple
from io import StringIO
from typing import Dict, List, Tuple, Iterable, Set
import subprocess, tempfile, csv
import csv as _csv  # avoid shadowing issues
import random
import bisect


import pandas as pd

# -------------------------
# Fixed background frequencies (user-provided, viral genome-wide)
# -------------------------
FIXED_BG = {
    'A': 0.362054,
    'C': 0.176903,
    'G': 0.239302,
    'T': 0.221741,
}

# -------------------------
# FASTA utilities (dependency-light)
# -------------------------

def read_fasta(fp: Path) -> Iterable[Tuple[str, str]]:
    """Yield (header, sequence) from a FASTA file. Header is the part before first whitespace.
    Upper-cases sequence and strips whitespace.
    """
    header = None
    seq_chunks: List[str] = []
    with open(fp) as f:
        for line in f:
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq_chunks).upper()
                header = line[1:].strip().split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
    if header is not None:
        yield header, ''.join(seq_chunks).upper()

# -------------------------
# MEME text parser and simple PSSM
# -------------------------

# FIMO helpers (external MEME Suite)
def write_fimo_bgfile(bg: Dict[str,float], path: Path):
    """Write a simple background file for FIMO's --bgfile option."""
    with open(path, 'w') as f:
        # FIMO expects one symbol per line: "<base> <prob>"
        f.write(f"A {bg['A']:.6f}\n")
        f.write(f"C {bg['C']:.6f}\n")
        f.write(f"G {bg['G']:.6f}\n")
        f.write(f"T {bg['T']:.6f}\n")



class SimplePSSM:
    """Lightweight PSSM with calculate(seq), and precomputed min/max sums.
    Scores are log2 odds vs background (A/C/G/T).
    """
    def __init__(self, pwm_cols: List[Dict[str, float]], bg: Dict[str, float] | None = None):
        self.len = len(pwm_cols)
        if bg is None:
            bg = {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}
        self.bg = {b: float(bg.get(b, 0.25)) for b in 'ACGT'}
        self._scores: List[Dict[str, float]] = []
        for col in pwm_cols:
            cs = {}
            for b in 'ACGT':
                p = max(float(col.get(b, 0.0)), 1e-9)
                cs[b] = math.log2(p / max(self.bg[b], 1e-9))
            self._scores.append(cs)
        per_pos_values = [[d[b] for b in 'ACGT'] for d in self._scores]
        self.min = sum(min(vs) for vs in per_pos_values)
        self.max = sum(max(vs) for vs in per_pos_values)

    def calculate(self, subseq: str) -> float:
        s = 0.0
        for i, ch in enumerate(subseq.upper()):
            if i >= self.len:
                break
            if ch in 'ACGT':
                s += self._scores[i][ch]
            else:
                # Non-ACGT: use background-weighted average at that position
                avg = sum(self._scores[i][b] * self.bg[b] for b in 'ACGT')
                s += avg
        return s

MotifObj = Dict[str, object]


def parse_meme_text(meme_path: Path) -> List[MotifObj]:
    """Parse motifs from a MEME **text** file (e.g., JASPAR core: "MEME version 4").
    Returns list of dicts with keys: name, acc, len, pssm, min, max.
    """
    text = Path(meme_path).read_text()
    lines = [ln.rstrip('\n') for ln in text.splitlines()]

    # Background frequencies: override with fixed user-provided viral background
    bg = dict(FIXED_BG)

    motif_objs: List[MotifObj] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('MOTIF'):
            toks = ln.split()
            acc = toks[1] if len(toks) > 1 else ''
            # If present, use human-readable name (e.g., ALX3). Else fall back to accession.
            motif_name = toks[2] if len(toks) > 2 else acc

            # Seek the letter-probability matrix header
            i += 1
            while i < len(lines) and not lines[i].startswith('letter-probability matrix'):
                i += 1
            if i >= len(lines):
                break

            header = lines[i]
            m = re.search(r"w=\s*(\d+)", header)
            if not m:
                i += 1
                continue
            w = int(m.group(1))

            pwm_cols: List[Dict[str, float]] = []
            for k in range(w):
                i += 1
                if i >= len(lines):
                    break
                vals = [float(x) for x in lines[i].strip().split()[:4]]
                if len(vals) != 4:
                    break
                pwm_cols.append({'A': vals[0], 'C': vals[1], 'G': vals[2], 'T': vals[3]})

            if len(pwm_cols) == w:
                pssm = SimplePSSM(pwm_cols, bg=bg)
                motif_objs.append({
                    'name': motif_name.strip(),
                    'acc': acc.strip(),
                    'len': w,
                    'pssm': pssm,
                    'min': pssm.min,
                    'max': pssm.max,
                })
            continue
        i += 1

    if not motif_objs:
        raise RuntimeError(f"No motifs parsed from MEME text: {meme_path}")

    return motif_objs



def run_fimo_on_tile(fimo_bin: str, meme_path: Path, fasta_path: Path, bg: Dict[str,float]) -> List[Dict[str,str]]:
    """Run FIMO on one tile FASTA and return parsed TSV rows (dicts).
    Uses --bgfile with FIXED_BG and a temporary output directory to preserve q-values.
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        bgfile = td_path / 'bg.txt'
        write_fimo_bgfile(bg, bgfile)
        out_dir = td_path / 'fimo_out'
        cmd = [
            fimo_bin,
            '--bgfile', str(bgfile),
            '--oc', str(out_dir),
            str(meme_path), str(fasta_path)
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"FIMO failed (code {proc.returncode}).\nSTDERR:\n{proc.stderr}\nCMD: {' '.join(cmd)}")
        tsv = out_dir / 'fimo.tsv'
        if not tsv.exists():
            alt = out_dir / 'fimo.txt'
            if not alt.exists():
                raise RuntimeError(f"FIMO output not found: {tsv} (or {alt})")
            lines = [ln for ln in alt.read_text().splitlines() if ln and not ln.startswith('#')]
            if not lines:
                return []
            reader = _csv.DictReader(lines, delimiter='\t')
            return [row for row in reader]
        rows: List[Dict[str,str]] = []
        with open(tsv) as f:
            reader = _csv.DictReader(f, delimiter='\t')
            for r in reader:
                rows.append(r)
        return rows


def fimo_rows_to_hits(rows: List[Dict[str,str]], tile_name: str,
                      gene_to_families: Dict[str,Set[str]],
                      composites: Dict[str,Set[str]],
                      p_thresh: float, q_thresh: float) -> Tuple[List[dict], List[dict]]:
    """Convert FIMO rows into (per_tf_hits, composite_hits) used by the pipeline.
    Filters by score >= 10 and per-position p/q. Uses -log10(p) as a sortable proxy for rel_score.
    """
    per_tf_hits: List[dict] = []
    composite_hits: List[dict] = []
    for r in rows:
        motif_id = (r.get('motif_alt_id') or r.get('motif_id') or r.get('motif') or '').strip()
        if not motif_id:
            continue
        isolate = (r.get('sequence_name') or r.get('sequence') or '').split('|')[0]
        try:
            start1 = int(r.get('start') or r.get('start_pos') or r.get('start1') or 0)
            stop1  = int(r.get('stop') or r.get('stop_pos') or r.get('end1') or 0)
        except Exception:
            continue
        if start1 <= 0 or stop1 <= 0:
            continue
        start = start1 - 1   # FIMO 1-based inclusive -> 0-based start
        end   = stop1        # keep python-style end index
        strand = (r.get('strand') or '+').strip()
        try:
            llr = float(r.get('score') or 0.0)
        except Exception:
            llr = 0.0
        try:
            pval = float(r.get('p-value') or r.get('pvalue') or 1.0)
        except Exception:
            pval = 1.0
        qraw = r.get('q-value')
        qval = float(qraw) if qraw not in (None, '') else None
        matched = (r.get('matched_sequence') or r.get('match') or '').upper()

        # per-position filters: score, p, q
        if llr < 10.0:
            continue
        if p_thresh < 1.0 and pval > p_thresh:
            continue
        if q_thresh < 1.0 and (qval is None or qval > q_thresh):
            continue

        # strength proxy for downstream sorting/NMS
        rel_proxy = -math.log10(max(pval, 1e-300))

        toks = motif_tokens(motif_id)
        genes_here: Set[str] = {t for t in toks if t in gene_to_families}
        fams_here: Set[str] = set()
        for g in genes_here:
            fams_here |= gene_to_families[g]
        is_comp = '::' in motif_id.upper()
        if is_comp:
            fams_here = collapse_to_composites(fams_here, composites)
            composite_hits.append(dict(
                tile=tile_name, isolate=isolate, start=start, end=end, strand=strand,
                motif=motif_id, motif_len=max(0, end-start), rel_score=rel_proxy, score=llr,
                kseq=matched, families_here=';'.join(sorted(fams_here)),
                assigned_family=None, p_value=pval, q_value=qval,
            ))
        else:
            for g in genes_here:
                per_tf_hits.append(dict(
                    tile=tile_name, isolate=isolate, start=start, end=end, strand=strand,
                    motif=motif_id, motif_len=max(0, end-start), rel_score=rel_proxy, score=llr,
                    kseq=matched, gene=g, p_value=pval, q_value=qval,
                ))
    return per_tf_hits, composite_hits

# -------------------------
# Family & composites mapping
# -------------------------

def read_tf_families(tfs_tsv: Path):
    df = pd.read_csv(tfs_tsv, sep='\t')
    if not {"family", "TFs"}.issubset(df.columns):
        raise ValueError("TFs.tsv must have columns: family, TFs")
    families = df["family"].tolist()
    family_to_genes: Dict[str, Set[str]] = {}
    gene_to_families: Dict[str, Set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        fam = str(row["family"]).strip()
        genes = [g.strip().upper() for g in str(row["TFs"]).split(',') if g.strip()]
        family_to_genes[fam] = set(genes)
        for g in genes:
            gene_to_families[g].add(fam)
    return families, family_to_genes, gene_to_families


def motif_tokens(name: str) -> Set[str]:
    toks = re.split(r"[^A-Za-z0-9]+", (name or "").upper())
    return {t for t in toks if t}


def map_motif_to_families(motif_name: str, gene_to_families: Dict[str, Set[str]]) -> Set[str]:
    fams: Set[str] = set()
    for tok in motif_tokens(motif_name):
        fams |= gene_to_families.get(tok, set())
    return fams


def read_precedence(tsv_path: Path) -> List[str]:
    if not tsv_path.exists():
        return []
    df = pd.read_csv(tsv_path, sep='\t')
    if not {"family", "priority"}.issubset(df.columns):
        raise ValueError("precedence TSV must have columns: family, priority")
    df = df.sort_values(["priority", "family"], ascending=[False, True])
    return df["family"].tolist()


def read_composites(tsv_path: Path) -> Dict[str, Set[str]]:
    if not tsv_path.exists():
        return {}
    df = pd.read_csv(tsv_path, sep='\t')
    if not {"meta_family", "members"}.issubset(df.columns):
        raise ValueError("composites TSV must have columns: meta_family, members")
    comp: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        meta = str(row["meta_family"]).strip()
        members = {m.strip() for m in str(row["members"]).split(',') if m.strip()}
        if meta and members:
            comp[meta] = members
    return comp


def collapse_to_composites(families: Set[str], composites: Dict[str, Set[str]]) -> Set[str]:
    """If a hit maps to ≥2 members of any composite, replace members by the meta-family."""
    fams = set(families)
    for meta, members in composites.items():
        if len(fams & members) >= 2:
            fams -= members
            fams.add(meta)
    return fams

# -------------------------
# Scanning utilities
# -------------------------

Hit = namedtuple("Hit", ["start", "end", "strand", "motif_name", "family", "score", "rel", "subseq"])  # used internally; family unused in per-hit path


def rel_score_from_pssm(pssm: SimplePSSM, subseq: str) -> Tuple[float, float]:
    score = pssm.calculate(subseq)
    rel = (score - pssm.min) / (pssm.max - pssm.min) if pssm.max != pssm.min else 0.0
    # clamp for numeric safety
    if rel < 0:
        rel = 0.0
    elif rel > 1:
        rel = 1.0
    return score, rel


def revcomp(seq: str) -> str:
    comp = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
    return seq.translate(comp)[::-1]


def scan_sequence(seq: str, pssm: SimplePSSM, motif_len: int, rel_thresh: float, motif_name: str, family: str | None) -> List[Hit]:
    hits: List[Hit] = []
    n = len(seq)
    if n < motif_len:
        return hits
    # + strand
    for i in range(0, n - motif_len + 1):
        sub = seq[i:i + motif_len]
        score, rel = rel_score_from_pssm(pssm, sub)
        if rel >= rel_thresh:
            hits.append(Hit(i, i + motif_len, '+', motif_name, family, score, rel, sub))
    # - strand
    rc = revcomp(seq)
    for i in range(0, n - motif_len + 1):
        sub = rc[i:i + motif_len]
        score, rel = rel_score_from_pssm(pssm, sub)
        if rel >= rel_thresh:
            # map back to forward coordinates
            start = n - (i + motif_len)
            end = n - i
            hits.append(Hit(start, end, '-', motif_name, family, score, rel, revcomp(sub)))
    return hits



def nms_hits(hits: List[Hit], radius_bp: int) -> List[Hit]:
    """Greedy non-maximum suppression by relative score, within a motif's hit list.
    Keep top-scoring hits; drop neighbors whose centers are within `radius_bp`.
    """
    if not hits:
        return []
    items = []
    for h in hits:
        c = (h.start + h.end) // 2
        items.append((h, c))
    # sort: rel desc, then shorter motif, then start
    items.sort(key=lambda x: (-x[0].rel, (x[0].end - x[0].start), x[0].start))
    kept: List[Hit] = []
    centers: List[int] = []
    for h, c in items:
        if all(abs(c - kc) > radius_bp for kc in centers):
            kept.append(h)
            centers.append(c)
    kept.sort(key=lambda h: (h.start, h.end, h.motif_name))
    return kept

# -------------------------
# Empirical significance (p-values via shuffles) and BH-FDR
# -------------------------
import random
import bisect

def sample_seq(length: int, bg: Dict[str, float]) -> str:
    bases = ['A','C','G','T']
    probs = [bg.get('A',0.25), bg.get('C',0.25), bg.get('G',0.25), bg.get('T',0.25)]
    cs = [probs[0]]
    for i in range(1,4):
        cs.append(cs[-1] + probs[i])
    out = []
    for _ in range(length):
        r = random.random()
        out.append('A' if r < cs[0] else 'C' if r < cs[1] else 'G' if r < cs[2] else 'T')
    return ''.join(out)


def max_rel_on_seq(seq: str, pssm: "SimplePSSM", motif_len: int) -> float:
    n = len(seq)
    best = 0.0
    if n < motif_len:
        return 0.0
    for i in range(0, n - motif_len + 1):
        sub = seq[i:i+motif_len]
        _, rel = rel_score_from_pssm(pssm, sub)
        if rel > best:
            best = rel
    rc = revcomp(seq)
    for i in range(0, n - motif_len + 1):
        sub = rc[i:i+motif_len]
        _, rel = rel_score_from_pssm(pssm, sub)
        if rel > best:
            best = rel
    return best


def empirical_p_for_hit(rel_val: float, seq_len: int, pssm: "SimplePSSM", motif_len: int,
                        bg: Dict[str,float], shuffles: int,
                        cache: Dict[Tuple[int,int,int], List[float]]) -> float:
    """Estimate p = P(max_rel >= rel_val) via mononucleotide shuffles under bg.
    Cache per (seq_len, motif_len, motif_id)."""
    key = (seq_len, motif_len, id(pssm))
    dist = cache.get(key)
    if dist is None:
        dist = []
        for _ in range(shuffles):
            s = sample_seq(seq_len, bg)
            dist.append(max_rel_on_seq(s, pssm, motif_len))
        dist.sort()
        cache[key] = dist
    idx = bisect.bisect_left(dist, rel_val)
    ge = len(dist) - idx
    return (ge + 1.0) / (len(dist) + 1.0)


def bh_fdr(pvals: List[float]) -> List[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0]*m
    prev = 1.0
    for rank, i in enumerate(order, start=1):
        val = pvals[i] * m / rank
        if val > 1.0:
            val = 1.0
        if val < prev:
            prev = val
        q[i] = prev
    return q

# -------------------------
# Cross-family exclusivity (precedence-aware NMS)
# -------------------------

def precedence_exclusive_nms(hlist: List[dict], precedence_list: List[str],
                             radius_bp: int, iou_thresh: float, shift_bp: int) -> Tuple[List[dict], List[dict]]:
    """Greedy keep one hit across *all families* at overlapping loci using precedence.
    Sorting priority: higher precedence first (earlier in list), then higher rel_score, then shorter motif, then start.
    Conflicts when: center distance <= radius_bp OR IoU >= iou_thresh OR |start|/|end| diff <= shift_bp.
    """
    if not hlist:
        return [], []

    # Precedence rank (lower is higher priority)
    prec_rank = {fam: i for i, fam in enumerate(precedence_list)} if precedence_list else {}

    def rank(d: dict) -> Tuple[int, float, int, int]:
        fam = d.get('assigned_family')
        pr = prec_rank.get(fam, len(prec_rank))
        rel = float(d.get('rel_score', 0.0))
        length = int(d['end']) - int(d['start'])
        return (pr, -rel, length, int(d['start']))

    def center(d: dict) -> int:
        return int(d['start']) + (int(d['end']) - int(d['start'])) // 2

    def iou(a: dict, b: dict) -> float:
        a0, a1 = int(a['start']), int(a['end'])
        b0, b1 = int(b['start']), int(b['end'])
        inter = max(0, min(a1, b1) - max(a0, b0))
        if inter == 0:
            return 0.0
        union = (a1 - a0) + (b1 - b0) - inter
        return inter / union if union > 0 else 0.0

    def contains(a: dict, b: dict) -> bool:
        a0, a1 = int(a['start']), int(a['end'])
        b0, b1 = int(b['start']), int(b['end'])
        return a0 <= b0 and a1 >= b1

    items = sorted(list(hlist), key=rank)
    kept: List[dict] = []
    dropped: List[dict] = []
    cluster_id = 0

    for d in items:
        keep = True
        reason = ''
        for k in kept:
            # explicit containment: either interval fully contains the other
            if contains(d, k) or contains(k, d):
                keep, reason = False, 'precedence-contain'; break
            if abs(center(d) - center(k)) <= radius_bp:
                keep, reason = False, 'precedence-center'; break
            if iou(d, k) >= iou_thresh:
                keep, reason = False, 'precedence-iou'; break
            if abs(int(d['start']) - int(k['start'])) <= shift_bp or abs(int(d['end']) - int(k['end'])) <= shift_bp:
                keep, reason = False, 'precedence-shift'; break
        if keep:
            d['cluster_id'] = f"assigned_family:{d.get('assigned_family','NA')}_c{cluster_id}"
            d['dedup_kept'] = True
            d['dedup_reason'] = ''
            kept.append(d)
            cluster_id += 1
        else:
            d['dedup_kept'] = False
            d['dedup_reason'] = reason
            dropped.append(d)

    kept.sort(key=lambda d: ((d.get('isolate') or ''), (d.get('tile') or ''), (d.get('assigned_family') or ''), d['start'], d['end']))
    dropped.sort(key=lambda d: ((d.get('isolate') or ''), (d.get('tile') or ''), (d.get('assigned_family') or ''), d['start'], d['end']))
    return kept, dropped

# -------------------------
# Family-level NMS helper
# -------------------------

def nms_family_assigned_hits(hlist: List[dict], radius_bp: int) -> Tuple[List[dict], List[dict]]:
    """Deduplicate assigned hits per family by center-distance NMS.
    Returns (kept, dropped). Each element is the original dict, annotated with 'cluster_id' and 'dedup_kept'.
    """
    if not hlist:
        return [], []
    # group by family
    kept_all: List[dict] = []
    drop_all: List[dict] = []
    for fam in sorted({h['assigned_family'] for h in hlist if h.get('assigned_family')}):
        fam_hits = [h for h in hlist if h.get('assigned_family') == fam]
        # build sortable tuples: (-rel, length, start, idx)
        items = []
        for idx, h in enumerate(fam_hits):
            length = int(h['end']) - int(h['start'])
            items.append(( -float(h['rel_score']), length, int(h['start']), idx))
        items.sort()
        centers: List[int] = []
        cluster_ids: List[int] = []
        cur_cluster = 0
        kept_idx: Set[int] = set()
        for (_negrel, _len, start, idx) in items:
            center = start + (fam_hits[idx]['end'] - fam_hits[idx]['start'])//2
            # find if near an existing center
            assigned_cluster = None
            for ci, c in enumerate(centers):
                if abs(center - c) <= radius_bp:
                    assigned_cluster = ci
                    break
            if assigned_cluster is None:
                centers.append(center)
                cluster_ids.append(cur_cluster)
                kept_idx.add(idx)
                cur_cluster += 1
            else:
                # already covered by a stronger hit (because sorted by -rel)
                pass
        # annotate
        for i, h in enumerate(fam_hits):
            center = h['start'] + (h['end'] - h['start'])//2
            # find nearest center (within radius) and whether this instance is the kept one
            kept = False
            cid = None
            for ci, c in enumerate(centers):
                if abs(center - c) <= radius_bp:
                    cid = ci
                    # kept instance is the first (highest score) that created this center
                    # recompute who was kept for this center
                    # Identify the original kept index by looking back into 'items'
                    # Simpler: mark kept when (abs(center-c)==0 and i in kept_idx) OR if this is exactly the representative
                    break
            # Determine kept by recomputing representative for this center
            if cid is not None:
                # find all members of this center
                members = []
                for (_negrel, _len, s2, idx2) in items:
                    c2 = s2 + (fam_hits[idx2]['end'] - fam_hits[idx2]['start'])//2
                    if abs(c2 - centers[cid]) <= radius_bp:
                        members.append(idx2)
                # representative is min in items order (highest rel)
                rep = min(members, key=lambda j: ( -float(fam_hits[j]['rel_score']), (fam_hits[j]['end']-fam_hits[j]['start']), fam_hits[j]['start']))
                kept = (i == rep)
                h['cluster_id'] = f"{fam}_c{cid}"
                h['dedup_kept'] = kept
                if kept:
                    kept_all.append(h)
                else:
                    drop_all.append(h)
            else:
                # shouldn't happen; treat as its own cluster
                h['cluster_id'] = f"{fam}_c{cur_cluster}"
                h['dedup_kept'] = True
                kept_all.append(h)
                cur_cluster += 1
    # sort kept for stable output
    kept_all.sort(key=lambda d: (d['isolate'], d['tile'], d['assigned_family'], d['start'], d['end']))
    drop_all.sort(key=lambda d: (d['isolate'], d['tile'], d['assigned_family'], d['start'], d['end']))
    return kept_all, drop_all

# -------------------------
# TF-level helpers & generic dedupe
# -------------------------
def motif_to_genes(name: str, valid_genes: set[str]) -> set[str]:
    """Return TF symbols present in the motif name that we actually care about."""
    toks = motif_tokens(name)
    return {t for t in toks if t in valid_genes}

def is_composite_motif(name: str) -> bool:
    return '::' in (name or '').upper()

def dedupe_hits_generic(hlist: list[dict], radius_bp: int, iou_thresh: float, shift_bp: int, key: str):
    """Greedy dedupe within groups (by `key`, e.g., 'gene' or 'assigned_family').
    Drop a candidate if it matches a kept hit in the same group by ANY of:
      - center distance <= radius_bp
      - IoU >= iou_thresh
      - |start_diff| <= shift_bp OR |end_diff| <= shift_bp
      - identical forward k-mer (kseq)
    Returns (kept, dropped) with 'cluster_id', 'dedup_kept', 'dedup_reason'.
    """
    if not hlist:
        return [], []

    def center(d): return int(d['start']) + (int(d['end']) - int(d['start'])) // 2
    def iou(a, b):
        a0, a1 = int(a['start']), int(a['end'])
        b0, b1 = int(b['start']), int(b['end'])
        inter = max(0, min(a1, b1) - max(a0, b0))
        if inter == 0: return 0.0
        union = (a1 - a0) + (b1 - b0) - inter
        return inter / union if union > 0 else 0.0
    def contains(a, b):
        a0, a1 = int(a['start']), int(a['end'])
        b0, b1 = int(b['start']), int(b['end'])
        return a0 <= b0 and a1 >= b1

    groups = {}
    for d in hlist:
        if key in d and d[key] is not None:
            groups.setdefault(str(d[key]), []).append(d)

    kept_all, drop_all = [], []
    for gval, items in groups.items():
        items.sort(key=lambda d: (-float(d['rel_score']), (int(d['end'])-int(d['start'])), int(d['start'])))
        kept, cluster_id = [], 0
        for d in items:
            keep, reason = True, ''
            for k in kept:
                # explicit containment: merge nested intervals
                if contains(d, k) or contains(k, d):
                    keep, reason = False, 'contain'; break
                if abs(center(d) - center(k)) <= radius_bp:
                    keep, reason = False, 'center'; break
                if iou(d, k) >= iou_thresh:
                    keep, reason = False, 'iou'; break
                if abs(int(d['start']) - int(k['start'])) <= shift_bp or abs(int(d['end']) - int(k['end'])) <= shift_bp:
                    keep, reason = False, 'shift'; break
                if d.get('kseq') and k.get('kseq') and d['kseq'] == k['kseq']:
                    keep, reason = False, 'seq'; break
            if keep:
                d['cluster_id'] = f"{key}:{gval}_c{cluster_id}"
                d['dedup_kept'] = True
                d['dedup_reason'] = ''
                kept.append(d); cluster_id += 1
            else:
                d['cluster_id'] = kept[-1]['cluster_id'] if kept else f"{key}:{gval}_c0"
                d['dedup_kept'] = False
                d['dedup_reason'] = reason
                drop_all.append(d)
        kept_all.extend(kept)

    kept_all.sort(key=lambda d: ((d.get('isolate') or ''), (d.get('tile') or ''), (d.get(key) or ''), d['start'], d['end']))
    drop_all.sort(key=lambda d: ((d.get('isolate') or ''), (d.get('tile') or ''), (d.get(key) or ''), d['start'], d['end']))
    return kept_all, drop_all

# -------------------------
# Main per-tile processing
# -------------------------

def process_tile_fasta(
    tile_fa: Path,
    motif_objs: List[MotifObj],
    families: List[str],
    family_to_genes: Dict[str, Set[str]],
    gene_to_families: Dict[str, Set[str]],
    rel_thresh: float,
    merge_bp: int,
    hits_out: Path,
    precedence_list: List[str],
    composites: Dict[str, Set[str]],
    nms_factor: float,
    family_dedupe_bp: int,
    family_iou: float,
    family_shift_bp: int,
    tf_dedupe_bp: int,
    tf_iou: float,
    tf_shift_bp: int,
    shuffles: int,
    p_thresh: float,
    q_thresh: float,
    fimo_rows: List[Dict[str,str]] | None = None,
    fimo_mode: bool = False,
) -> pd.DataFrame:
    """Scan one tile FASTA across all isolates; return a counts DataFrame.
    Rows: isolate accession (header up to first '|'); Cols: provided families list.
    """
    isolates: List[str] = []
    counts_rows: List[List[int]] = []
    hits_rows: List[Dict[str, object]] = []

    # cache of null distributions keyed by (seq_len, motif_len, motif_id)
    null_cache: Dict[Tuple[int,int,int], List[float]] = {}

    # Precompute motif->families mapping to restrict scanning to relevant motifs
    motif_to_families: Dict[str, Set[str]] = {}
    for mo in motif_objs:
        fams = map_motif_to_families(str(mo['name']), gene_to_families)
        if fams:
            motif_to_families[str(mo['name'])] = fams

    # If using FIMO, pre-index rows by isolate for this tile
    fimo_by_isolate = defaultdict(list)
    if fimo_mode and fimo_rows:
        for r in fimo_rows:
            iso = (r.get('sequence_name') or r.get('sequence') or '').split('|')[0]
            fimo_by_isolate[iso].append(r)

    for header, seq in read_fasta(tile_fa):
        isolate = header.split('|')[0]
        isolates.append(isolate)
        fam_counts = {f: 0 for f in families}
        assigned_hits: List[dict] = []

        # Build valid gene set once
        valid_genes: Set[str] = set().union(*family_to_genes.values())

        # Scan motifs; keep only those mapping to our TFs or composites
        per_tf_hits: List[dict] = []
        composite_hits: List[dict] = []

        if fimo_mode:
            rows_iso = fimo_by_isolate.get(isolate, [])
            per_tf_hits, composite_hits = fimo_rows_to_hits(rows_iso, tile_fa.stem, gene_to_families, composites, p_thresh, q_thresh)
        else:
            # existing in-script scanning path (kept as fallback)
            valid_genes: Set[str] = set().union(*family_to_genes.values())
            for mo in motif_objs:
                name = str(mo['name'])
                genes = motif_to_genes(name, valid_genes)
                is_comp = is_composite_motif(name)
                if not genes and not is_comp:
                    continue
                pssm: SimplePSSM = mo['pssm']  # type: ignore
                mlen: int = int(mo['len'])     # type: ignore
                raw_hits = scan_sequence(seq, pssm, mlen, rel_thresh, name, None)
                if not raw_hits:
                    continue
                radius = max(3, int(round(mlen * nms_factor)))
                keep_hits = nms_hits(raw_hits, radius_bp=radius)

                for h in keep_hits:
                    p_value = None
                    if shuffles > 0:
                        p_value = empirical_p_for_hit(h.rel, len(seq), pssm, mlen, FIXED_BG, shuffles, null_cache)
                        if p_value is not None and p_value > p_thresh:
                            continue
                    kseq = h.subseq
                    if is_comp:
                        fams_here = collapse_to_composites(map_motif_to_families(name, gene_to_families), composites)
                        composite_hits.append({
                            'tile': tile_fa.stem, 'isolate': isolate,
                            'start': h.start, 'end': h.end, 'strand': h.strand,
                            'motif': h.motif_name, 'motif_len': mlen, 'rel_score': h.rel,
                            'kseq': kseq, 'families_here': ';'.join(sorted(fams_here)),
                            'assigned_family': None, 'p_value': p_value,
                        })
                    else:
                        for g in genes:
                            per_tf_hits.append({
                                'tile': tile_fa.stem, 'isolate': isolate,
                                'start': h.start, 'end': h.end, 'strand': h.strand,
                                'motif': h.motif_name, 'motif_len': mlen, 'rel_score': h.rel,
                                'kseq': kseq, 'gene': g, 'p_value': p_value,
                            })

        # (A) Per-TF dedupe across PWMs
        kept_tf, dropped_tf = dedupe_hits_generic(per_tf_hits, radius_bp=tf_dedupe_bp,
                                                  iou_thresh=tf_iou, shift_bp=tf_shift_bp, key='gene')

        # Map deduped TF hits to families (resolve with precedence)
        for d in kept_tf:
            fams = set()
            for fam, genes in family_to_genes.items():
                if d['gene'] in genes:
                    fams.add(fam)
            fams = collapse_to_composites(fams, composites)
            assigned = None
            if precedence_list:
                for fam in precedence_list:
                    if fam in fams:
                        assigned = fam; break
            if assigned is None and fams:
                assigned = sorted(fams)[0]
            if assigned:
                d2 = d.copy()
                d2['families_here'] = ';'.join(sorted(fams))
                d2['assigned_family'] = assigned
                assigned_hits.append(d2)

        # (B) Handle composite motif hits directly
        for d in composite_hits:
            fams_here = set(d['families_here'].split(';')) if d['families_here'] else set()
            assigned = None
            if precedence_list:
                for fam in precedence_list:
                    if fam in fams_here:
                        assigned = fam; break
            if assigned is None and fams_here:
                assigned = sorted(fams_here)[0]
            d['assigned_family'] = assigned
            assigned_hits.append(d)

        # 2) Exclusivity across families at overlapping loci (respect precedence)
        kept, dropped = precedence_exclusive_nms(assigned_hits, precedence_list,
                                                 radius_bp=family_dedupe_bp,
                                                 iou_thresh=family_iou,
                                                 shift_bp=family_shift_bp)
        # Optionally, within-family dedupe after exclusivity:
        # kept, dropped_more = dedupe_hits_generic(kept, radius_bp=family_dedupe_bp,
        #                                          iou_thresh=family_iou, shift_bp=family_shift_bp,
        #                                          key='assigned_family')
        # dropped.extend(dropped_more)
        # Optional BH-FDR over kept hits (per isolate/tile scope)
        if q_thresh < 1.0:
            pvals = [h.get('p_value') for h in kept]
            idx_map = [i for i, pv in enumerate(pvals) if pv is not None]
            pv_list = [pvals[i] for i in idx_map]
            qvals = bh_fdr(pv_list)
            # assign and split
            kept2, dropped2 = [], []
            for j, i in enumerate(idx_map):
                kept[i]['q_value'] = qvals[j]
            for i, h in enumerate(kept):
                if 'q_value' in h and h['q_value'] > q_thresh:
                    h['dedup_kept'] = False
                    h['dedup_reason'] = 'q'
                    dropped2.append(h)
                else:
                    kept2.append(h)
            kept = kept2
            dropped.extend(dropped2)

        # Recompute fam_counts from kept only
        fam_counts = {f: 0 for f in families}
        for h in kept:
            fam = h.get('assigned_family')
            if fam in fam_counts:
                fam_counts[fam] += 1

        # Diagnostics: write both kept and dropped with flags
        hits_rows.extend(kept)
        hits_rows.extend(dropped)

        counts_rows.append([fam_counts[f] for f in families])

    df = pd.DataFrame(counts_rows, columns=families, index=isolates)
    df.index.name = 'isolate'

    # Save diagnostics
    if hits_rows:
        hdf = pd.DataFrame(hits_rows)
        hdf.to_csv(hits_out, sep='\t', index=False)

    return df

# -------------------------
# CLI
# -------------------------

def main():
    p = argparse.ArgumentParser(description="Family-aware motif scanning (per-hit assignment; no cross-hit merging)")
    p.add_argument("--meme", required=True, help="Path to MEME text motif file (e.g., JASPAR core)")
    p.add_argument("--tfs", required=True, help="Path to TF family mapping TSV (family\tTFs)")
    p.add_argument("--tiles_dir", default="results/tiles", help="Directory containing tile_*.fasta files")
    p.add_argument("--outdir", default="results/motif_counts", help="Output directory for count matrices and hits TSVs")
    p.add_argument("--precedence", default="../data/family_precedence.tsv", help="TSV with columns family,priority for exclusivity precedence")
    p.add_argument("--composites", default="../data/family_composites.tsv", help="TSV with columns meta_family,members to collapse heterodimers")
    p.add_argument("--rel_score", type=float, default=0.90, help="Minimum relative PSSM score [0-1]")
    p.add_argument("--merge_bp", type=int, default=2, help="(unused in per-hit mode; kept for CLI compatibility)")
    p.add_argument("--nms_factor", type=float, default=0.75, help="Per-motif NMS radius as a fraction of motif length (default 0.75)")
    p.add_argument("--family_dedupe_bp", type=int, default=8,
                   help="Within an isolate/tile, deduplicate assigned hits per family using NMS radius (bp). Default 8 bp.")
    p.add_argument("--tiles", default="", help="Comma-separated list of tile numbers to process (default: all found)")
    p.add_argument("--family_iou", type=float, default=0.5,
               help="IoU threshold for family-level dedupe (default 0.5)")
    p.add_argument("--family_shift_bp", type=int, default=1,
               help="Shift tolerance (bp) for family-level dedupe (default 1)")
    p.add_argument("--tf_dedupe_bp", type=int, default=6,
               help="Within an isolate/tile, dedupe per TF symbol across PWMs (bp radius). Default 6")
    p.add_argument("--tf_iou", type=float, default=0.5,
               help="IoU threshold for per-TF dedupe (default 0.5)")
    p.add_argument("--tf_shift_bp", type=int, default=1,
               help="Shift tolerance (bp) for per-TF dedupe (default 1)")
    p.add_argument("--shuffles", type=int, default=0,
                   help="# of mononucleotide shuffles per (sequence length, motif) to estimate null of max rel-score; 0 disables p/q filtering")
    p.add_argument("--p_thresh", type=float, default=1.0,
                   help="Empirical p-value threshold (<=) to keep hits when --shuffles>0")
    p.add_argument("--q_thresh", type=float, default=1.0,
                   help="Benjamini–Hochberg q-value threshold (<=) applied to kept hits per isolate/tile when --shuffles>0")
    p.add_argument('--use_fimo', action='store_true',
               help='Use external FIMO to obtain per-position log-likelihood scores and p/q-values; skips in-script scanning and empirical shuffles.')
    p.add_argument('--fimo_bin', default='fimo',
                help='Path to the FIMO executable (default: fimo).')
    args = p.parse_args()

    meme_path = Path(args.meme)
    tfs_tsv = Path(args.tfs)
    tiles_dir = Path(args.tiles_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load config tables
    families, family_to_genes, gene_to_families = read_tf_families(tfs_tsv)
    precedence_list = read_precedence(Path(args.precedence))
    composites = read_composites(Path(args.composites))

    # Extend family columns with any composite metas
    composite_metas = [m for m in composites.keys() if m not in families]
    if precedence_list:
        col_families = [f for f in precedence_list if f in set(families) | set(composite_metas)]
        remaining = [f for f in (families + composite_metas) if f not in col_families]
        col_families += remaining
    else:
        col_families = families + sorted(composite_metas)

    # Parse MEME text motifs
    motif_objs = parse_meme_text(meme_path)

    # Filter to tile list if provided
    sel_tiles = set(x.strip() for x in args.tiles.split(',') if x.strip())
    fasta_paths = sorted(tiles_dir.glob("tile_*.fasta"))
    if sel_tiles:
        fasta_paths = [p for p in fasta_paths if p.stem.split('_')[1] in sel_tiles]

    if not fasta_paths:
        raise SystemExit(f"No FASTAs found under {tiles_dir}")

    for fa in fasta_paths:
        tile = fa.stem  # e.g., tile_6
        fimo_rows = None
        if args.use_fimo:
            fimo_rows = run_fimo_on_tile(args.fimo_bin, meme_path, fa, FIXED_BG)
        df = process_tile_fasta(
            fa, motif_objs, col_families, family_to_genes, gene_to_families,
            rel_thresh=args.rel_score, merge_bp=args.merge_bp,
            hits_out=outdir / f"{tile}.hits.tsv",
            precedence_list=precedence_list,
            composites=composites,
            nms_factor=args.nms_factor,
            family_dedupe_bp=args.family_dedupe_bp,
            family_iou=args.family_iou,
            family_shift_bp=args.family_shift_bp,
            tf_dedupe_bp=args.tf_dedupe_bp,
            tf_iou=args.tf_iou,
            tf_shift_bp=args.tf_shift_bp,
            shuffles=0 if args.use_fimo else args.shuffles,  # disable empirical p when FIMO is on
            p_thresh=args.p_thresh,
            q_thresh=args.q_thresh,
            fimo_rows=fimo_rows,
            fimo_mode=args.use_fimo
        )
        df.to_csv(outdir / f"{tile}.counts.tsv", sep='\t')
        print(f"[motif_scans] Wrote {outdir / f'{tile}.counts.tsv'} ({df.shape[0]} isolates x {df.shape[1]} families)")


if __name__ == "__main__":
    main()

#
# -------------------------
# Empirical significance (p-values via shuffles) and BH-FDR
# -------------------------

def sample_seq(length: int, bg: Dict[str, float]) -> str:
    bases = ['A','C','G','T']
    probs = [bg.get('A',0.25), bg.get('C',0.25), bg.get('G',0.25), bg.get('T',0.25)]
    cs = [probs[0]]
    for i in range(1,4):
        cs.append(cs[-1] + probs[i])
    out = []
    for _ in range(length):
        r = random.random()
        out.append('A' if r < cs[0] else 'C' if r < cs[1] else 'G' if r < cs[2] else 'T')
    return ''.join(out)


def max_rel_on_seq(seq: str, pssm: SimplePSSM, motif_len: int) -> float:
    n = len(seq)
    best = 0.0
    if n < motif_len:
        return 0.0
    # + strand
    for i in range(0, n - motif_len + 1):
        sub = seq[i:i+motif_len]
        _, rel = rel_score_from_pssm(pssm, sub)
        if rel > best:
            best = rel
    # - strand
    rc = revcomp(seq)
    for i in range(0, n - motif_len + 1):
        sub = rc[i:i+motif_len]
        _, rel = rel_score_from_pssm(pssm, sub)
        if rel > best:
            best = rel
    return best


def empirical_p_for_hit(rel_val: float, seq_len: int, pssm: SimplePSSM, motif_len: int,
                        bg: Dict[str,float], shuffles: int,
                        cache: Dict[Tuple[int,int,int], List[float]]) -> float:
    """Estimate p = P(max_rel >= rel_val) via mononucleotide shuffles under bg.
    Cache per (seq_len, motif_len, motif_id) for speed.
    """
    key = (seq_len, motif_len, id(pssm))
    dist = cache.get(key)
    if dist is None:
        dist = []
        for _ in range(shuffles):
            s = sample_seq(seq_len, bg)
            dist.append(max_rel_on_seq(s, pssm, motif_len))
        dist.sort()
        cache[key] = dist
    idx = bisect.bisect_left(dist, rel_val)
    ge = len(dist) - idx
    return (ge + 1.0) / (len(dist) + 1.0)


def bh_fdr(pvals: List[float]) -> List[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0]*m
    prev = 1.0
    for rank, i in enumerate(order, start=1):
        val = pvals[i] * m / rank
        if val > 1.0:
            val = 1.0
        if val < prev:
            prev = val
        q[i] = prev
    # enforce monotonic non-decreasing in original order by scanning backwards on sorted
    # (prev loop already ensures the standard BH monotone property via running min)
    return q