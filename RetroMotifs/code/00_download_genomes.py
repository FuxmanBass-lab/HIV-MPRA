#!/usr/bin/env python3
"""
00_download_genomes.py

Download full genomes (FASTA) for all isolates referenced in a TSV, then
compute a genome-wide background model (A/C/G/T frequencies) across the
downloaded sequences.

Example:
  python 00_download_genomes.py \
    --sequence_tsv ../data/sequence.tsv \
    --outdir ../data/genomes \
    --email you@lab.edu \
    --api_key $NCBI_API_KEY \
    --max_workers 4 \
    --skip_existing

Outputs:
  - FASTA files in outdir (one per accession, .fasta)
  - background.tsv (tab: base, count, freq)
  - background.json (A/C/G/T frequencies)
  - genomes.list (downloaded accession list)

Notes:
  * Uses Biopython Entrez if available; falls back to HTTP E-utilities.
  * Respects NCBI rate limits (3 req/s without key; 10 req/s with key).
  * Retries with exponential backoff on transient errors.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import io
import json
import os
import sys
import time
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Set, Tuple, Optional

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from Bio import Entrez  # type: ignore
    HAVE_BIO = True
except Exception:
    HAVE_BIO = False

try:
    import requests  # type: ignore
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False

# -------------------------
# Utility: read TSV and collect accessions
# -------------------------

def read_accessions_from_tsv(tsv_path: Path, genome_col: str = "genome") -> List[str]:
    accs: List[str] = []
    with open(tsv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if genome_col not in reader.fieldnames:
            raise ValueError(f"Column '{genome_col}' not found in {tsv_path}")
        for row in reader:
            acc = (row.get(genome_col) or "").strip()
            if acc:
                accs.append(acc)
    # de-duplicate but preserve order
    seen: Set[str] = set()
    uniq: List[str] = []
    for a in accs:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq

# -------------------------
# Downloaders
# -------------------------

@dataclass
class DLConfig:
    email: str
    api_key: Optional[str]
    tool: str = "retro-mpra"
    max_retries: int = 5
    timeout: int = 60
    sleep_no_key: float = 0.35  # ~3/sec
    sleep_with_key: float = 0.10  # ~10/sec


class NCBIDownloader:
    def __init__(self, cfg: DLConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        if HAVE_BIO:
            Entrez.email = cfg.email
            if cfg.api_key:
                Entrez.api_key = cfg.api_key
            Entrez.tool = cfg.tool

    def _sleep(self):
        time.sleep(self.cfg.sleep_with_key if self.cfg.api_key else self.cfg.sleep_no_key)

    def fetch_fasta(self, acc: str) -> str:
        # Try Biopython first
        last_err: Optional[Exception] = None
        if HAVE_BIO:
            for attempt in range(self.cfg.max_retries):
                try:
                    with Entrez.efetch(db="nuccore", id=acc, rettype="fasta", retmode="text") as handle:
                        data = handle.read()
                    if data and data.startswith(">"):
                        return data
                    # If empty or not FASTA, try without version suffix
                    base = acc.split(".")[0]
                    if base != acc:
                        with Entrez.efetch(db="nuccore", id=base, rettype="fasta", retmode="text") as handle:
                            data = handle.read()
                        if data and data.startswith(">"):
                            return data
                    raise RuntimeError(f"Empty or invalid FASTA for {acc}")
                except Exception as e:
                    last_err = e
                    self._sleep()
            # fall through to HTTP
        if not HAVE_REQUESTS:
            raise RuntimeError(f"Failed to download {acc}: Biopython/requests unavailable. Last error: {last_err}")
        # HTTP fallback
        params = {
            "db": "nuccore",
            "id": acc,
            "rettype": "fasta",
            "retmode": "text",
            "tool": self.cfg.tool,
            "email": self.cfg.email,
        }
        if self.cfg.api_key:
            params["api_key"] = self.cfg.api_key
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                r = requests.get(url, params=params, timeout=self.cfg.timeout)
                if r.status_code == 200 and r.text.startswith(">"):
                    return r.text
                if r.status_code in (429, 500, 502, 503, 504):
                    # backoff
                    time.sleep(min(5, (attempt + 1) * 0.5))
                    continue
                # Try without version
                base = acc.split(".")[0]
                if base != acc:
                    params2 = dict(params)
                    params2["id"] = base
                    r2 = requests.get(url, params=params2, timeout=self.cfg.timeout)
                    if r2.status_code == 200 and r2.text.startswith(">"):
                        return r2.text
                raise RuntimeError(f"Bad HTTP ({r.status_code}) or invalid payload for {acc}")
            except Exception as e:
                last_err = e
                self._sleep()
        raise RuntimeError(f"Failed to download {acc} after retries. Last error: {last_err}")

# -------------------------
# Background estimation
# -------------------------

def write_fasta(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def compute_background_from_dir(genomes_dir: Path) -> Tuple[Dict[str, int], Dict[str, float]]:
    counts = {b: 0 for b in "ACGT"}
    total = 0
    for ext in ("*.fa", "*.fasta", "*.fna"):
        for fp in genomes_dir.glob(ext):
            with open(fp, "r") as f:
                for line in f:
                    if not line or line.startswith(">"):
                        continue
                    s = line.strip().upper()
                    for ch in s:
                        if ch in counts:
                            counts[ch] += 1
                            total += 1
    if total == 0:
        freqs = {b: 0.25 for b in "ACGT"}
    else:
        freqs = {b: counts[b] / total for b in "ACGT"}
    return counts, freqs

# -------------------------
# Main
# -------------------------

def main():
    p = argparse.ArgumentParser(
        description="Download isolate genomes and compute background base frequencies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sequence_tsv", required=True, help="TSV with a 'genome' column of accessions")
    p.add_argument("--outdir", required=True, help="Directory to write FASTAs and background files")
    p.add_argument("--email", required=True, help="Contact email for NCBI E-utilities")
    p.add_argument("--api_key", default="", help="NCBI API key (optional, increases rate limit)")
    p.add_argument("--max_workers", type=int, default=4, help="Concurrent downloads")
    p.add_argument("--skip_existing", action="store_true", help="Skip accessions with existing FASTA files")
    args = p.parse_args()

    tsv_path = Path(args.sequence_tsv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    accessions = read_accessions_from_tsv(tsv_path)
    if not accessions:
        print(f"[download_genomes] No accessions found in {tsv_path}", file=sys.stderr)
        sys.exit(1)

    cfg = DLConfig(email=args.email, api_key=args.api_key or None)
    dl = NCBIDownloader(cfg)

    print(f"[download_genomes] Will download {len(accessions)} accessions to {outdir}")

    def target(acc: str) -> Tuple[str, bool, Optional[str]]:
        fasta_path = outdir / f"{acc}.fasta"
        if args.skip_existing and fasta_path.exists() and fasta_path.stat().st_size > 0:
            return acc, True, None
        try:
            text = dl.fetch_fasta(acc)
            write_fasta(fasta_path, text)
            return acc, True, None
        except Exception as e:
            return acc, False, str(e)

    ok: List[str] = []
    fail: List[Tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futs = {ex.submit(target, acc): acc for acc in accessions}
        for i, fut in enumerate(as_completed(futs), 1):
            acc, success, err = fut.result()
            if success:
                ok.append(acc)
                print(f"[download_genomes] [{i}/{len(accessions)}] downloaded {acc}")
            else:
                fail.append((acc, err or "unknown error"))
                print(f"[download_genomes] [{i}/{len(accessions)}] FAILED {acc}: {err}", file=sys.stderr)

    # Write list file
    with open(outdir / "genomes.list", "w") as f:
        for acc in ok:
            f.write(f"{acc}\n")

    # Compute background over successfully downloaded FASTAs
    counts, freqs = compute_background_from_dir(outdir)

    # Save background
    with open(outdir / "background.tsv", "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["base", "count", "freq"])
        for b in "ACGT":
            w.writerow([b, counts[b], f"{freqs[b]:.8f}"])
    with open(outdir / "background.json", "w") as f:
        json.dump(freqs, f, indent=2)

    print("[download_genomes] Background frequencies:")
    for b in "ACGT":
        print(f"  {b}: {freqs[b]:.6f} (n={counts[b]})")

    if fail:
        print(f"[download_genomes] {len(fail)} accessions failed:")
        for acc, err in fail[:20]:
            print(f"  - {acc}: {err}")
        if len(fail) > 20:
            print(f"  ... {len(fail)-20} more")

if __name__ == "__main__":
    main()
