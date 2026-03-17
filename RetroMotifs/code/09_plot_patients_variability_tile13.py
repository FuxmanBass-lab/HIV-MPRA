#!/usr/bin/env python3
"""
Plot per-patient motif configuration variation for tile 13
using the collapsed 4-row layout.

Signature format in the TSV:
    AAA|B|CC|D

where:
    AAA -> 3 IRF2 positions
    B   -> 1 IRF3 position
    CC  -> 2 E2F positions
    D   -> 1 SP/KLF position

Biological site interpretation BEFORE collapsing:
    site1: AAA[0] (IRF2)
    site2: AAA[1] (IRF2) OR CC[0] (E2F)
    site3: AAA[2] (IRF2) OR B (IRF3)
    site4: CC[1]  (E2F)
    site5: D      (SP/KLF)

We COLLAPSE site1 + site2 → 4 rows total:

    row 0: aggregated(site1, site2):
              if AAA[0] == '1' or AAA[1] == '1' → IRF2 color
              elif CC[0] == '1'                 → E2F color
              else                               → absent
    row 1: site3:
              if B == '1' → IRF3 color
              elif AAA[2] == '1' → IRF2 color
              else → absent
    row 2: site4:
              if CC[1] == '1' → E2F color
              else → absent
    row 3: site5:
              if D == '1' → SP/KLF color
              else → absent

python3 09_plot_patients_variability_tile13.py \
    ../results/patients_variability/patient_tile13_signature_counts.tsv \
    -o ../results/patients_variability/patient_tile13_variation.pdf
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# colors
COL_IRF2 = "#7FB6E0"   # IRF2-ish (light blue)
COL_IRF3 = "#005F73"   # IRF3-ish (darker teal)
COL_E2F  = "#8FBF88"   # green
COL_SPK  = "#9370db"   # purple
COL_ABS  = "#eeeeee"
EDGE     = "#555555"


def parse_signature(sig: str):
    """
    Take 'AAA|B|CC|D' -> (AAA, B, CC, D), padded safely.
    """
    a, b, c, d = sig.split("|", 3)
    a = a.ljust(3, "0")
    b = b.ljust(1, "0")
    c = c.ljust(2, "0")
    d = d.ljust(1, "0")
    return a, b, c, d


def motif_color_for_row(sig: str, row_idx: int) -> str:
    """
    Return the color for this signature at row_idx (0..3),
    using the collapsed logic described above.
    """
    if sig == "UNMATCHED":
        return COL_ABS

    a, b, c, d = parse_signature(sig)

    # row 0: collapsed site1+site2
    if row_idx == 0:
        # IRF2 in site1 or site2
        if a[0] == "1" or a[1] == "1":
            return COL_IRF2
        # otherwise E2F in site2
        elif c[0] == "1":
            return COL_E2F
        else:
            return COL_ABS

    # row 1: site3: IRF3 (B) > IRF2 (AAA[2])
    elif row_idx == 1:
        if b[0] == "1":
            return COL_IRF3
        elif a[2] == "1":
            return COL_IRF2
        else:
            return COL_ABS

    # row 2: site4: E2F second bit
    elif row_idx == 2:
        return COL_E2F if c[1] == "1" else COL_ABS

    # row 3: site5: SP/KLF
    elif row_idx == 3:
        return COL_SPK if d[0] == "1" else COL_ABS

    return COL_ABS


def sparsity(sig: str) -> int:
    """
    Count how many of the 4 collapsed rows are empty for this signature.
    We sort slices by this so "fullest" ones are drawn first.
    """
    if sig == "UNMATCHED":
        return 9999

    a, b, c, d = parse_signature(sig)

    empty = 0

    # row 0
    if not (a[0] == "1" or a[1] == "1" or c[0] == "1"):
        empty += 1
    # row 1
    if not (b[0] == "1" or a[2] == "1"):
        empty += 1
    # row 2
    if c[1] != "1":
        empty += 1
    # row 3
    if d[0] != "1":
        empty += 1

    return empty


def main():
    ap = argparse.ArgumentParser(description="Plot patient variation heatmap for TILE 13 (collapsed 4-site).")
    ap.add_argument("table", help="TSV with patient_ID, total_isolates_tile13, unmatched_tile13, and signature columns")
    ap.add_argument("-o", "--output", default="patient_tile13_variation.png")
    args = ap.parse_args()

    df = pd.read_csv(args.table, sep="\t")
    sig_cols = list(df.columns[3:])

    n_rows = 4
    row_h = 0.14
    extra_bottom = 1.6
    extra_top = 0.3
    fig_h = n_rows * row_h + extra_bottom + extra_top
    fig_w = fig_h * 4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for i, (_, row) in enumerate(df.iterrows()):
        patient = row["patient_ID"]
        total = int(row["total_isolates_tile13"])
        unmatched = int(row["unmatched_tile13"])

        # collect present signatures
        sig_counts = []
        for sig in sig_cols:
            c = int(row[sig])
            if c > 0:
                sig_counts.append((sig, c))
        if unmatched > 0:
            sig_counts.append(("UNMATCHED", unmatched))

        # sort by least sparse, then by count desc, then by name
        sig_counts.sort(key=lambda x: (sparsity(x[0]), -x[1], x[0]))

        if total == 0:
            # draw 4 empty cells
            for r in range(n_rows):
                ax.add_patch(
                    Rectangle((i, r), 1.0, 1.0, facecolor=COL_ABS, edgecolor=EDGE, linewidth=0.5)
                )
            continue

        x_cursor = i
        remaining = 1.0
        for j, (sig, count) in enumerate(sig_counts):
            frac = count / total
            if j < len(sig_counts) - 1:
                width = frac
            else:
                width = remaining  # fill to end to avoid gaps
            remaining -= width

            for r in range(n_rows):
                color = motif_color_for_row(sig, r)
                ax.add_patch(
                    Rectangle(
                        (x_cursor, r),
                        width,
                        1.0,
                        facecolor=color,
                        edgecolor="none",
                        linewidth=0,
                        antialiased=False,
                        zorder=1,
                    )
                )

            x_cursor += width

        # border around patient col
        ax.add_patch(
            Rectangle((i, 0), 1.0, n_rows,
                      facecolor="none", edgecolor=EDGE, linewidth=0.8, zorder=5)
        )

        # patient label
        ax.text(
            i + 0.5,
            -0.01,
            patient,
            ha="center",
            va="top",
            rotation=90,
            fontsize=6,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )

    # axes styling
    ax.set_xlim(0, len(df))
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # horizontal separators
    for y in range(n_rows + 1):
        ax.hlines(y, 0, len(df), colors="#cccccc", linewidth=0.4, zorder=3)

    # legend
    legend_elems = [
        Line2D([0], [0], marker='s', color='w', label='IRF2',   markerfacecolor=COL_IRF2, markersize=8),
        Line2D([0], [0], marker='s', color='w', label='IRF3',   markerfacecolor=COL_IRF3, markersize=8),
        Line2D([0], [0], marker='s', color='w', label='E2F',    markerfacecolor=COL_E2F,  markersize=8),
        Line2D([0], [0], marker='s', color='w', label='SP/KLF', markerfacecolor=COL_SPK,  markersize=8),
        Line2D([0], [0], marker='s', color='w', label='Absent', markerfacecolor=COL_ABS,  markersize=8),
    ]
    ax.legend(handles=legend_elems, loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)

    plt.subplots_adjust(bottom=0.35, left=0.14, right=0.82, top=1 - (extra_top / fig_h))
    plt.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()