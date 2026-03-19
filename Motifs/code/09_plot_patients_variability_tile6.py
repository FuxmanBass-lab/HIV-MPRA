#!/usr/bin/env python3
"""
Plot per-patient motif configuration variation for tile 6.

Input: TSV like the one you showed:
patient_ID  total_isolates_tile6  unmatched_tile6  0000|0100  0001|0011  ...

Output: a heatmap-like figure where:
- x = patients
- y = 7 motif positions:
    0: NFKB1
    1: NFKB2
    2: NFKB3
    3: SHARED (NFKB4 or SP1)
    4: SP2
    5: SP3
    6: SP4
- each patient column may be split into multiple vertical slices proportional
  to how many isolates had each signature.

python3 09_plot_patients_variability_tile6.py \
    ../results/patients_variability/patient_tile6_signature_counts.tsv \
    -o ../results/patients_variability/patient_tile6_variation.pdf

"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# colors
NFKB = "#66a3a3"    # muted teal
SP   = "#9370db"    # muted purple
ABS  = "#eeeeee"
EDGE = "#555555"


def parse_signature(sig: str):
    """
    Split something like '0110|1111' into ('0110', '1111').
    We assume 4 bits for nfkb, 4 bits for sp.
    """
    nfkb_bits, sp_bits = sig.split("|", 1)
    return nfkb_bits, sp_bits


def motif_color_for_row(sig: str, row_idx: int):
    """
    Map a signature to a color at a given motif row (0..6).

    Row meaning:
      0 -> NFKB1  -> nfkb_bits[0]
      1 -> NFKB2  -> nfkb_bits[1]
      2 -> NFKB3  -> nfkb_bits[2]
      3 -> SHARED -> nfkb_bits[3] OR sp_bits[0]
           (if nfkb_bits[3] == '1' -> NFKB
            elif sp_bits[0] == '1' -> SP
            else ABS)
      4 -> SP2 -> sp_bits[1]
      5 -> SP3 -> sp_bits[2]
      6 -> SP4 -> sp_bits[3]
    """
    nfkb_bits, sp_bits = parse_signature(sig)

    # safety
    nfkb_bits = nfkb_bits.ljust(4, "0")
    sp_bits = sp_bits.ljust(4, "0")

    if row_idx == 0:
        return NFKB if nfkb_bits[0] == "1" else ABS
    elif row_idx == 1:
        return NFKB if nfkb_bits[1] == "1" else ABS
    elif row_idx == 2:
        return NFKB if nfkb_bits[2] == "1" else ABS
    elif row_idx == 3:
        if nfkb_bits[3] == "1":
            return NFKB
        elif sp_bits[0] == "1":
            return SP
        else:
            return ABS
    elif row_idx == 4:
        return SP if sp_bits[1] == "1" else ABS
    elif row_idx == 5:
        return SP if sp_bits[2] == "1" else ABS
    elif row_idx == 6:
        return SP if sp_bits[3] == "1" else ABS
    else:
        return ABS


def main():
    ap = argparse.ArgumentParser(description="Plot patient variation heatmap for tile 6 motif configs.")
    ap.add_argument("table", help="TSV file with patient_ID, total_isolates_tile6, unmatched_tile6, and signature columns")
    ap.add_argument("-o", "--output", default="patient_tile6_variation.png", help="Output image file")
    args = ap.parse_args()

    df = pd.read_csv(args.table, sep="\t")

    # columns after the first 3 are signatures
    sig_cols = list(df.columns[3:])

    n_rows = 7  # fixed, per your spec
    row_h = 0.14  # inches per motif row (reduced from 0.28)
    extra_bottom = 1.6  # a bit more for labels
    extra_top = 0.3
    fig_h = n_rows * row_h + extra_bottom + extra_top
    fig_w = fig_h * 4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))


    def sparsity(sig):
        if sig == "UNMATCHED":
            return float('inf')
        nfkb_bits, sp_bits = parse_signature(sig)
        bits = nfkb_bits + sp_bits
        return bits.count('0')

    # We’ll put each patient at x = i, width = 1.0
    for i, (_, row) in enumerate(df.iterrows()):
        patient = row["patient_ID"]
        total = int(row["total_isolates_tile6"])
        unmatched = int(row["unmatched_tile6"])

        # Build list of (signature, count) for signatures present
        sig_counts = []
        for sig in sig_cols:
            c = int(row[sig])
            if c > 0:
                sig_counts.append((sig, c))

        # If there are unmatched isolates, treat them as "all absent"
        if unmatched > 0:
            sig_counts.append(("UNMATCHED", unmatched))

        # sort sig_counts by sparsity (least sparse first), count desc, then signature
        sig_counts.sort(key=lambda x: (sparsity(x[0]), -x[1], x[0]))

        # If total is 0, just draw empty column
        if total == 0:
            # draw one empty slice
            for r in range(n_rows):
                rect = Rectangle((i, r), 1.0, 1.0,
                                 facecolor=ABS, edgecolor=EDGE, linewidth=0.5)
                ax.add_patch(rect)
            continue

        # now draw slices across the patient column
        x_cursor = i
        remaining = 1.0
        for idx2, (sig, count) in enumerate(sig_counts):
            frac = count / total
            if idx2 < len(sig_counts) - 1:
                width = frac
            else:
                width = remaining  # fill to end to avoid gaps in vector editors
            remaining -= width

            for r in range(n_rows):
                if sig == "UNMATCHED":
                    color = ABS
                else:
                    color = motif_color_for_row(sig, r)
                rect = Rectangle(
                    (x_cursor, r),
                    width,
                    1.0,
                    facecolor=color,
                    edgecolor='none',
                    linewidth=0,
                    antialiased=False,
                    zorder=1,
                )
                ax.add_patch(rect)

            x_cursor += width

        # Draw one rectangle border around the entire patient column
        border_rect = Rectangle((i, 0), 1.0, n_rows,
                                facecolor='none', edgecolor=EDGE, linewidth=0.8, zorder=5)
        ax.add_patch(border_rect)

        # Patient label below the heatmap, in axis coordinates
        ax.text(i + 0.5, -0.01, patient,
                ha="center", va="top", rotation=90, fontsize=6, transform=ax.get_xaxis_transform(), clip_on=False)

    # Axes formatting
    ax.set_xlim(0, len(df))
    ax.set_ylim(0, n_rows)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.invert_yaxis()  # y=0 at top
    ax.set_xticks([])

    # Add horizontal gridlines to separate motif rows (appear on top of rectangles)
    for y in range(n_rows + 1):
        ax.hlines(y, 0, len(df), colors="#cccccc", linewidth=0.4, zorder=3)

    # tidy
    ax.set_frame_on(False)

    # legend (manual)
    # small patches for NFKB / SP / ABS
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='s', color='w', label='NFKB/REL', markerfacecolor=NFKB, markersize=8),
        Line2D([0], [0], marker='s', color='w', label='SP/KLF', markerfacecolor=SP, markersize=8),
        Line2D([0], [0], marker='s', color='w', label='Absent', markerfacecolor=ABS, markersize=8),
    ]
    ax.legend(handles=legend_elems, loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)

    plt.subplots_adjust(bottom=0.35, left=0.14, right=0.82, top=1 - (extra_top / fig_h))
    plt.savefig(args.output, dpi=300)
    # plt.show()


if __name__ == "__main__":
    main()