#!/usr/bin/env python3
"""
Compute edit-distance–based similarity distributions among multiple versions
of the same 200 bp genome tile and plot violins per tile.

INPUT
  A delimited file (TSV/CSV) with at least these columns:
    - tile_id   : unique identifier for each sequence instance
    - sequence  : nucleotide sequence string (A/C/G/T/N, ~200 bp)
    - tile      : tile label (e.g., 'HIV-1:REJO:28:+', 'Human-T-Lymphoma-Virus:Type-1:7:+')
    - organism  : organism label (e.g., 'HIV_1', 'HIV_2', 'Human_T_Lymphoma_Virus')

WHAT IT DOES
  1) Normalizes the 'tile' into (virus_key, tile_number, strand).
     Examples:
       'HIV-1:REJO:28:+'              -> ('HIV1', 28, '+')
       'HIV-2:SMTH:185:+'             -> ('HIV2', 185, '+')
       'Human-T-Lymphoma-Virus:...:7:+' -> ('HTLV', 7, '+')
  2) Filters rows to a user-specified set of targets like:
       HIV1: 6+,9+,11+,13+
       HIV2: 185+,188+
       HTLV: 2+,4+,7+,10+,12+
  3) For each (virus_key, number, '+') group, computes all pairwise Levenshtein
     distances among sequences and converts to similarity = 1 - (dist / max_len).
  4) Saves:
       - per-pair similarity file (CSV)
       - per-tile summary stats (CSV)
       - violin plot PNG

USAGE
  python tile_similarity_violin.py \
      --in file.tsv \
      --out-prefix results/sim_by_tile \
      --sep '\t' \
      --tiles "HIV1:6+,9+,11+,13+;HIV2:185+,188+;HTLV:2+,4+,7+,10+,12+"

  python analyze.py \
     --plot-only \
     --pairwise ../results/.pairwise.csv \
     --out-prefix ../results/ \
     --pdf

  Notes:
    - --sep auto-detected from file extension if not provided (.tsv -> '\t', else ',').
    - You can change tiles with the --tiles string. Virus keys must be one of: HIV1, HIV2, HTLV.
    - Only '+' strand tiles are selected here by design (matches your list).
"""

import argparse
import itertools
import math
import os
import re
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

import concurrent.futures
try:
    import Levenshtein as _Lev
except Exception:
    _Lev = None


# ----------------------------- Levenshtein (edit distance) ----------------------------- #
def levenshtein(a: str, b: str) -> int:
    """
    Standard dynamic programming Levenshtein edit distance.
    Space-optimized to O(min(len(a), len(b))) which is fine for ~200 bp strings.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    # Ensure len(a) <= len(b) to use less memory
    if la > lb:
        a, b = b, a
        la, lb = lb, la

    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        cj = b[j - 1]
        curr = [j] + [0] * la
        for i in range(1, la + 1):
            cost = 0 if a[i - 1] == cj else 1
            curr[i] = min(
                prev[i] + 1,     # deletion
                curr[i - 1] + 1, # insertion
                prev[i - 1] + cost  # substitution
            )
        prev = curr
    return prev[la]


def edit_distance(a: str, b: str) -> int:
    """Use python-Levenshtein if available; otherwise fall back to DP implementation."""
    if _Lev is not None:
        return _Lev.distance(a, b)
    return levenshtein(a, b)


from itertools import combinations

def _process_group(args_tuple):
    (vk, num, strand), seqs = args_tuple
    out = []
    for (id1, s1), (id2, s2) in combinations(seqs, 2):
        d = edit_distance(s1, s2)
        denom = max(len(s1), len(s2))
        nd = (d / denom) if denom > 0 else 0.0  # normalized distance
        sim = 1.0 - nd
        out.append({
            "virus_key": vk,
            "tile_num": num,
            "strand": strand,
            "group": f"{vk} {num}{strand}",
            "tile_id_1": id1,
            "tile_id_2": id2,
            "len_1": len(s1),
            "len_2": len(s2),
            "edit_distance": d,
            "normalized_distance": nd,
            "similarity": sim,
        })
    return out


# ----------------------------- Parsing / Normalization ----------------------------- #
def normalize_virus_key(organism: str, tile_str: str) -> str:
    """
    Map organism/tile strings to canonical virus keys expected in --tiles filter:
      'HIV1', 'HIV2', 'HTLV'
    We look at both 'organism' and 'tile' to be robust.
    """
    org = (organism or "").replace("-", "_").upper()
    tile_upper = (tile_str or "").upper()
    if "HIV_1" in org or "HIV-1" in tile_upper:
        return "HIV1"
    if "HIV_2" in org or "HIV-2" in tile_upper:
        return "HIV2"
    if "LYMPHOMA" in org or "HUMAN-T-LYMPHOMA-VIRUS" in tile_upper or "HTLV" in tile_upper:
        return "HTLV"
    # Fallback heuristic: if the tile begins with HIV-1/HIV-2 etc.
    if tile_upper.startswith("HIV-1"):
        return "HIV1"
    if tile_upper.startswith("HIV-2"):
        return "HIV2"
    return "UNKNOWN"


def parse_tile_number_and_strand(tile_str: str) -> Tuple[int, str]:
    """
    Extract the tile number and strand from a 'tile' like:
      'HIV-1:REJO:28:+' -> (28, '+')
      'Human-T-Lymphoma-Virus:Type-1:7:+' -> (7, '+')
    Robust to suffixes after the strand (e.g., '+_Modified_MT033882.1').
    """
    parts = tile_str.split(":")
    num = None
    strand = None
    if len(parts) >= 2:
        last = parts[-1].strip()
        # Allow decorations after the first char, e.g., '+_Modified_MT033882.1'
        if last and last[0] in {"+", "-"}:
            strand = last[0]
        # Find the last integer-like token before the strand
        for tok in reversed(parts[:-1]):
            m = re.match(r"^\s*(\d+)\s*$", tok)
            if m:
                num = int(m.group(1))
                break
    if num is None or strand not in {"+", "-"}:
        raise ValueError(f"Could not parse tile number/strand from: {tile_str}")
    return num, strand


def build_filter_set(tiles_arg: str) -> set:
    """
    Parse a string like:
      "HIV1:6+,9+,11+,13+;HIV2:185+,188+;HTLV:2+,4+,7+,10+,12+"
    into a set of tuples: { ('HIV1', 6, '+'), ... }
    """
    targets = set()
    if not tiles_arg:
        return targets
    for block in tiles_arg.split(";"):
        block = block.strip()
        if not block:
            continue
        if ":" not in block:
            raise ValueError(f"Bad --tiles block: {block}")
        virus_key, nums = block.split(":", 1)
        virus_key = virus_key.strip().upper()
        for item in nums.split(","):
            item = item.strip()
            m = re.match(r"^(\d+)\s*([+-])$", item)
            if not m:
                raise ValueError(f"Bad tile token: {item} in block '{block}'")
            n = int(m.group(1))
            s = m.group(2)
            targets.add((virus_key, n, s))
    return targets


# ----------------------------- Main pipeline ----------------------------- #
def main():
    p = argparse.ArgumentParser(description="Compute pairwise edit distances per tile and plot violins of normalized distance (d/maxlen).")
    p.add_argument("--in", dest="infile", required=False, help="Input TSV/CSV with columns tile_id,sequence,tile,organism")
    p.add_argument(
        "--sep",
        default=None,
        help=(
            "Field separator. Default: auto (.tsv -> \\t, else ,). "
            "Regex or multi-character separators (e.g., \\s+ or ::) are supported and will "
            "automatically use pandas' python engine to avoid warnings."
        ),
    )
    p.add_argument("--tiles", required=False,
                   help="Tile filter like 'HIV1:6+,9+,11+,13+;HIV2:185+,188+;HTLV:2+,4+,7+,10+,12+'")
    p.add_argument("--out-prefix", required=True, help="Output prefix for CSVs and PNG (directory will be created).")
    p.add_argument("--min_pairs", type=int, default=2,
                   help="Require at least this many sequences (=> >=1 pair) per tile to include in plot. Default: 2")
    p.add_argument("--limit_per_tile", type=int, default=0,
                   help="Optional cap on number of sequences per tile (0 = no cap). Useful to speed up very large groups.")
    p.add_argument("--drop_dups", action="store_true",
                   help="If set, drop exact duplicate sequences within the same tile before pairwise comparisons.")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Parallel workers over tile groups (1 = no parallelism). Recommended: number of CPU cores.")
    p.add_argument("--plot-only", action="store_true",
                   help="Do not recompute distances. Load existing per-pair CSV and just (re)plot.")
    p.add_argument("--pairwise", default=None,
                   help="Path to an existing per-pair CSV produced by this script (required with --plot-only).")
    p.add_argument("--pdf", action="store_true",
                   help="Also save the violin plot as PDF (in addition to PNG). In --plot-only mode, PDF is recommended.")
    args = p.parse_args()

    # Validate required args only when computing
    if not args.plot_only:
        if not args.infile or not args.tiles:
            raise SystemExit("--in and --tiles are required unless --plot-only is set")

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    # ----------------------------- PLOT-ONLY SHORT-CIRCUIT ----------------------------- #
    if args.plot_only:
        if not args.pairwise:
            raise SystemExit("--plot-only requires --pairwise path to an existing per-pair CSV")
        res = pd.read_csv(args.pairwise)
        # Rebuild display labels consistent with compute mode
        def display_vk(vk: str) -> str:
            return {"HIV1": "HIV-1", "HIV2": "HIV-2", "HTLV": "HTLV-1"}.get(str(vk), str(vk))
        if "display_group" not in res.columns:
            # Expect columns virus_key, tile_num, strand
            if not set(["virus_key", "tile_num", "strand"]).issubset(res.columns):
                raise SystemExit("pairwise CSV must include columns: virus_key, tile_num, strand (or precomputed display_group)")
            res["display_group"] = res.apply(lambda r: f"{display_vk(r['virus_key'])} {int(r['tile_num'])}{r['strand']}", axis=1)

        # Desired order from PI (keep only present)
        desired_order = [
            "HIV-1 6+", "HIV-1 9+", "HIV-1 11+", "HIV-1 13+",
            "HIV-2 185+", "HIV-2 188+",
            "HTLV-1 2+", "HTLV-1 4+", "HTLV-1 7+", "HTLV-1 10+", "HTLV-1 12+",
        ]
        present = [g for g in desired_order if g in set(res["display_group"]) ]
        if not present:
            raise SystemExit("No expected groups found in pairwise CSV for plotting.")

        # Plot normalized distance violin (no recomputation)
        plt.figure(figsize=(10, 5))
        data = [res.loc[res["display_group"] == g, "normalized_distance"].values for g in present]
        plt.violinplot(dataset=data, showmeans=True, showextrema=True, showmedians=False)
        plt.xticks(ticks=range(1, len(present) + 1), labels=present, rotation=45, ha="right")
        plt.ylabel("Normalized edit distance (d / max length)")
        plt.title("Normalized edit-distance distributions by tile")
        # Separators between virus families
        boundaries = []
        try:
            idx_hiv1_last = max(i for i, g in enumerate(present, start=1) if g.startswith("HIV-1"))
            boundaries.append(idx_hiv1_last + 0.5)
        except ValueError:
            pass
        try:
            idx_hiv2_last = max(i for i, g in enumerate(present, start=1) if g.startswith("HIV-2"))
            boundaries.append(idx_hiv2_last + 0.5)
        except ValueError:
            pass
        for x in boundaries:
            plt.axvline(x=x, linestyle=":")
        plt.tight_layout()

        png_path = f"{args.out_prefix}.violin.png"
        plt.savefig(png_path, dpi=300)
        if args.pdf:
            pdf_path = f"{args.out_prefix}.violin.pdf"
            plt.savefig(pdf_path)
        plt.close()
        print("[OK] Wrote plot only:\n  - {}{}".format(
            png_path,
            f"\n  - {args.out_prefix}.violin.pdf" if args.pdf else ""
        ))
        return

    # Auto separator and engine selection
    sep = args.sep
    if sep is None:
        # Default based on extension; users may still pass regex or multi-char seps via --sep
        sep = "\t" if args.infile.lower().endswith(".tsv") else ","

    # Choose engine: use fast 'c' for single-char seps; 'python' for regex/multi-char like '\\s+' or '::'
    engine_kwargs = {}
    try:
        use_python = False
        if sep is None:
            use_python = True
        else:
            # If separator looks like a regex (contains backslash or special regex chars) or has length > 1
            regex_chars = set(".*+?^$[](){}|\\")
            if len(sep) > 1 or any(ch in regex_chars for ch in sep):
                use_python = True
        if use_python:
            engine_kwargs["engine"] = "python"
    except Exception:
        # Fallback safely to python engine on any weird sep
        engine_kwargs["engine"] = "python"

    df = pd.read_csv(args.infile, sep=sep, dtype=str, **engine_kwargs).fillna("")

    required_cols = {"tile_id", "sequence", "tile", "organism"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Normalize & parse
    norm_rows = []
    for _, r in df.iterrows():
        virus_key = normalize_virus_key(r["organism"], r["tile"])
        try:
            num, strand = parse_tile_number_and_strand(r["tile"])
        except Exception:
            continue  # skip unparseable
        norm_rows.append({
            "tile_id": r["tile_id"],
            "sequence": r["sequence"].strip().upper(),
            "tile_raw": r["tile"],
            "organism": r["organism"],
            "virus_key": virus_key,
            "tile_num": num,
            "strand": strand
        })
    ndf = pd.DataFrame(norm_rows)

    # Build target filter set
    targets = build_filter_set(args.tiles)

    # Filter to requested (virus_key, tile_num, strand)
    ndf["vk_num_strand"] = list(zip(ndf["virus_key"].str.upper(), ndf["tile_num"], ndf["strand"]))
    ndf = ndf[ndf["vk_num_strand"].isin(targets)].copy()

    # Clean sequences and (optionally) drop exact duplicates within tile
    def valid_seq(s: str) -> bool:
        return bool(re.match(r"^[ACGTN]+$", s))

    ndf = ndf[ndf["sequence"].apply(valid_seq)].copy()
    if args.drop_dups:
        ndf = ndf.drop_duplicates(subset=["vk_num_strand", "sequence"])

    # (Optional) Limit per tile to speed up
    if args.limit_per_tile and args.limit_per_tile > 0:
        ndf = (
            ndf.groupby("vk_num_strand", group_keys=False)
               .head(args.limit_per_tile)
               .reset_index(drop=True)
        )

    # Pairwise distances per tile (with optional parallelism)
    group_items = []
    for key, g in ndf.groupby(["virus_key", "tile_num", "strand"], sort=True):
        seqs = g[["tile_id", "sequence"]].values.tolist()
        if len(seqs) < args.min_pairs:
            continue
        group_items.append((key, seqs))

    results = []
    if not group_items:
        raise SystemExit("No groups with enough sequences. Check your --tiles filter and input data.")

    if args.n_jobs and args.n_jobs > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            for chunk in ex.map(_process_group, group_items):
                results.extend(chunk)
    else:
        for item in group_items:
            results.extend(_process_group(item))

    res = pd.DataFrame(results)

    # Save per-pair results (includes edit_distance and normalized_distance)
    pair_csv = f"{args.out_prefix}.pairwise.csv"
    res.to_csv(pair_csv, index=False)

    # Build display names and enforce PI order
    def display_vk(vk: str) -> str:
        return {"HIV1": "HIV-1", "HIV2": "HIV-2", "HTLV": "HTLV-1"}.get(vk, vk)

    res["display_group"] = res.apply(lambda r: f"{display_vk(r['virus_key'])} {r['tile_num']}{r['strand']}", axis=1)

    desired_order = [
        "HIV-1 6+", "HIV-1 9+", "HIV-1 11+", "HIV-1 13+",
        "HIV-2 185+", "HIV-2 188+",
        "HTLV-1 2+", "HTLV-1 4+", "HTLV-1 7+", "HTLV-1 10+", "HTLV-1 12+",
    ]
    # Keep only groups present, preserving order
    present = [g for g in desired_order if g in set(res["display_group"]) ]

    # Summary stats per group (normalized distance primary)
    sum_norm = (
        res.groupby("display_group")["normalized_distance"]
          .agg(
              n="count",
              median_normalized_distance="median",
              mean_normalized_distance="mean",
              q25_normalized_distance=lambda s: s.quantile(0.25),
              q75_normalized_distance=lambda s: s.quantile(0.75),
          )
          .reset_index()
    )

    # Optional raw distance median for the PI
    sum_raw = (
        res.groupby("display_group")["edit_distance"].agg(median_distance="median").reset_index()
    )

    summary = pd.merge(sum_norm, sum_raw, on="display_group", how="left")
    summary["display_group"] = pd.Categorical(summary["display_group"], categories=present, ordered=True)
    summary = summary.sort_values("display_group").reset_index(drop=True)

    summary_csv = f"{args.out_prefix}.summary.csv"
    summary.to_csv(summary_csv, index=False)

    # ----------------------------- Plot violins (Normalized Distance) ----------------------------- #
    plt.figure(figsize=(10, 5))

    # Data in desired display order
    data = [res.loc[res["display_group"] == g, "normalized_distance"].values for g in present]
    parts = plt.violinplot(dataset=data, showmeans=True, showextrema=True, showmedians=False)

    plt.xticks(ticks=range(1, len(present) + 1), labels=present, rotation=45, ha="right")
    plt.ylabel("Normalized edit distance (d / max length)")
    plt.title("Normalized edit-distance distributions by tile")

    # Add separators between virus types: after last HIV-1 and last HIV-2 index
    # Determine boundary indices in 1-based xtick positions
    boundaries = []
    try:
        idx_hiv1_last = max(i for i, g in enumerate(present, start=1) if g.startswith("HIV-1"))
        boundaries.append(idx_hiv1_last + 0.5)
    except ValueError:
        pass
    try:
        idx_hiv2_last = max(i for i, g in enumerate(present, start=1) if g.startswith("HIV-2"))
        boundaries.append(idx_hiv2_last + 0.5)
    except ValueError:
        pass

    for x in boundaries:
        plt.axvline(x=x, linestyle=":")

    plt.tight_layout()

    png_path = f"{args.out_prefix}.violin.png"
    plt.savefig(png_path, dpi=300)
    plt.close()

    # Pretty-print median distances in the requested order
    med_lines = []
    for g in present:
        row = summary.loc[summary["display_group"] == g, ["median_normalized_distance", "median_distance", "n"]]
        if not row.empty:
            med_nd = row.iloc[0]["median_normalized_distance"]
            med_d = row.iloc[0]["median_distance"]
            n = int(row.iloc[0]["n"])
            med_lines.append(f"{g}: median_norm_dist={med_nd:.4f}, median_dist={med_d:.2f} (n={n} pairs)")

    print("[OK] Wrote:\n  - {}\n  - {}\n  - {}".format(pair_csv, summary_csv, png_path))
    if med_lines:
        print("Median distances:")
        for line in med_lines:
            print("  " + line)


if __name__ == "__main__":
    main()