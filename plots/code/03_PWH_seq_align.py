#!/usr/bin/env python3
"""
Patient-level MPRA sequence alignment and log2 fold-change visualization.

This script generates per-patient sequence alignment heatmaps for selected HIV-1
MPRA tiles by combining MAFFT-aligned isolate sequences with log2 fold-change
measurements across stimulation conditions.

Inputs
------
data/patient_isolates_metadata_psdcountactivity.csv

Outputs
-------
results/PWH_seq_align/
"""

from pathlib import Path
import re
import subprocess
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter, FormatStrFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

CSV_PATH = DATA_DIR / "patient_isolates_metadata_psdcountactivity.csv"
OUTPUT_DIR = RESULTS_DIR / "PWH_seq_align"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TILE_REGION_FILTER = ":13:"
DESIRED_TILE_NUMS = {6, 9, 11, 13}
MIN_DNA_MEAN = 50
MIN_SEQS_PER_PATIENT = 3
MIN_UNIQUE_SEQS = 3

ALL_CONDITIONS = [
    "BE_Unstim_log2FoldChange",
    "psdcnt_Stim_log2FoldChange",
    "psdcnt_TNF_log2FoldChange",
    "psdcnt_IFNG_log2FoldChange",
]

EXCLUDE_BY_TILE = {
    6: {"psdcnt_IFNG_log2FoldChange"},
    11: {"psdcnt_TNF_log2FoldChange"},
    13: {"psdcnt_TNF_log2FoldChange"},
}

CONDITION_LABELS = {
    "BE_Unstim_log2FoldChange": "Unstim",
    "psdcnt_Stim_log2FoldChange": "Stim",
    "psdcnt_TNF_log2FoldChange": "TNF",
    "psdcnt_IFNG_log2FoldChange": "IFNG",
}

BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3, "-": 4}
SEQ_COLORS = ["green", "blue", "orange", "red", "lightgray"]


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    required_columns = [
        "tile",
        "dna_mean",
        "final_pat_code",
        "isolate_sequence",
        *ALL_CONDITIONS,
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["dna_mean"] = pd.to_numeric(df["dna_mean"], errors="coerce")
    df = df[df["dna_mean"] >= MIN_DNA_MEAN].copy()

    invalid_codes = {"", "0", "na", "#na"}
    df = df.dropna(subset=["final_pat_code"])
    df = df[~df["final_pat_code"].astype(str).str.strip().isin(invalid_codes)].copy()
    df = df.dropna(subset=ALL_CONDITIONS).copy()

    return df


def run_mafft(sequences: list[str], labels: list[str]) -> tuple[list[str], list[str]]:
    temp_input = OUTPUT_DIR / "_tmp_seqs.fasta"
    temp_output = OUTPUT_DIR / "_tmp_aligned.fasta"

    records = [
        SeqRecord(Seq(seq), id=label, description="")
        for seq, label in zip(sequences, labels)
    ]
    SeqIO.write(records, temp_input, "fasta")

    with temp_output.open("w") as handle:
        subprocess.run(
            ["mafft", "--auto", str(temp_input)],
            stdout=handle,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    aligned_records = list(SeqIO.parse(temp_output, "fasta"))

    temp_input.unlink(missing_ok=True)
    temp_output.unlink(missing_ok=True)

    return [str(record.seq) for record in aligned_records], [record.id for record in aligned_records]


def build_sequence_matrix(aligned_seqs: list[str]) -> np.ndarray:
    max_len = max(len(seq) for seq in aligned_seqs)
    matrix = np.full((len(aligned_seqs), max_len), BASE_TO_INT["-"], dtype=int)

    for i, seq in enumerate(aligned_seqs):
        for j, base in enumerate(seq):
            matrix[i, j] = BASE_TO_INT.get(base.upper(), BASE_TO_INT["-"])

    return matrix


def plot_patient(
    tile: str,
    patient: str,
    sp_aligned: pd.DataFrame,
    aligned_seqs: list[str],
    conditions_for_tile: list[str],
    out_path: Path,
) -> None:
    n_rows = len(sp_aligned)
    matrix = build_sequence_matrix(aligned_seqs)
    max_len = matrix.shape[1]

    values = [sp_aligned[col].values for col in conditions_for_tile]
    dup_vals = sp_aligned["dup_count"].values

    fig = plt.figure(figsize=(18, 3.33))
    fig.subplots_adjust(left=0.10, right=0.92, top=0.78, bottom=0.28)

    gridspec = fig.add_gridspec(1, 3, width_ratios=[1.5, 7.0, 0.50], wspace=0.05)
    ax_hist = fig.add_subplot(gridspec[0])
    ax_seq = fig.add_subplot(gridspec[1])
    ax_lfc = fig.add_subplot(gridspec[2], sharey=ax_seq)

    max_dup = max(dup_vals)
    ax_hist.barh(np.arange(len(dup_vals)), dup_vals, color="#CC1717C3", left=0)

    for i, value in enumerate(dup_vals):
        offset = max_dup * 0.08
        text_x = value + offset if value < max_dup * 0.95 else value - offset * 3
        ax_hist.text(
            text_x,
            i + 0.5,
            str(value),
            va="center",
            ha="left",
            fontsize=10,
            color="black",
            clip_on=False,
        )

    ax_hist.set_xlim(0, max_dup + 0.5)
    ax_hist.invert_xaxis()
    ax_hist.set_xticks(range(0, max_dup + 1, 5) if max_dup >= 5 else [1, 2, 3, 4, 5])
    ax_hist.tick_params(axis="x", bottom=True, labelbottom=True, top=False, labeltop=False, labelsize=9, pad=1)
    ax_hist.set_xlabel("Duplicate Count", fontsize=12, labelpad=1)
    ax_hist.xaxis.set_label_position("bottom")
    ax_hist.set_ylim(len(dup_vals), 0)
    ax_hist.set_yticks([])
    for spine in ax_hist.spines.values():
        spine.set_visible(False)

    cmap_seq = colors.ListedColormap(SEQ_COLORS)
    norm_seq = colors.BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap_seq.N)
    x_edges = np.arange(max_len + 1)
    y_edges = np.arange(n_rows + 1)

    ax_seq.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        cmap=cmap_seq,
        norm=norm_seq,
        shading="auto",
        edgecolors="none",
    )
    ax_seq.set_xlim(0, max_len)
    ax_seq.set_xticks(np.arange(0, max_len, 50))
    ax_seq.set_xticklabels(np.arange(1, max_len + 1, 50))
    ax_seq.tick_params(axis="x", rotation=45, bottom=True, top=False, pad=12)
    ax_seq.set_yticks([])

    combined = np.vstack(values).T
    n_cols = len(conditions_for_tile)
    white_red = LinearSegmentedColormap.from_list("white_red", ["white", "red"])

    im_unstim = ax_lfc.pcolormesh(
        np.array([0, 1]),
        y_edges,
        combined[:, [0]],
        cmap=plt.get_cmap("Greens"),
        vmin=0,
        vmax=6,
        shading="auto",
    )

    im_rest = None
    if n_cols > 1:
        im_rest = ax_lfc.pcolormesh(
            np.arange(1, n_cols + 1),
            y_edges,
            combined[:, 1:],
            cmap=white_red,
            vmin=0,
            vmax=1.5,
            shading="auto",
        )

    ax_lfc.set_xlim(0, n_cols)
    ax_lfc.set_xticks(np.arange(0.5, n_cols, 1.0))
    ax_lfc.set_xticklabels(
        [CONDITION_LABELS[col] for col in conditions_for_tile],
        rotation=45,
        ha="center",
        fontsize=10,
    )
    ax_lfc.set_yticks([])
    ax_lfc.xaxis.set_ticks_position("top")

    bbox = ax_lfc.get_position()
    cbar_height = bbox.height / 3.0
    center_y = bbox.y0 + (bbox.height - cbar_height) / 2.0

    cax_green = fig.add_axes([bbox.x1 + 0.025, center_y, 0.015, cbar_height])
    cbar_green = fig.colorbar(im_unstim, cax=cax_green, orientation="vertical")
    cbar_green.set_ticks([0, 1, 2, 3, 4, 5, 6])
    cbar_green.ax.yaxis.set_major_formatter(FixedFormatter(["0", "1", "2", "3", "4", "5", "6"]))
    cbar_green.ax.tick_params(length=3, labelsize=10, pad=2)
    cbar_green.outline.set_linewidth(0.8)

    if im_rest is not None:
        cax_red = fig.add_axes([bbox.x1 + 0.050, center_y, 0.015, cbar_height])
        cbar_red = fig.colorbar(im_rest, cax=cax_red, orientation="vertical")
        cbar_red.set_ticks([0.0, 0.5, 1.0, 1.5])
        cbar_red.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        cbar_red.ax.tick_params(length=3, labelsize=10, pad=2)
        cbar_red.outline.set_linewidth(0.8)

    base_handles = [
        Patch(facecolor="green", edgecolor="none", label="A"),
        Patch(facecolor="blue", edgecolor="none", label="C"),
        Patch(facecolor="orange", edgecolor="none", label="G"),
        Patch(facecolor="red", edgecolor="none", label="T"),
        Patch(facecolor="lightgray", edgecolor="none", label="– (gap)"),
    ]
    fig.legend(handles=base_handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.03), frameon=False, fontsize=10)

    plt.suptitle(f"Tile {tile}, Patient {patient}", fontsize=21)
    plt.tight_layout(rect=[0, 0.24, 1, 0.94])

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    df = load_data(CSV_PATH)
    all_csv_records: list[dict] = []

    for tile in df["tile"].unique():
        if TILE_REGION_FILTER not in tile:
            continue

        match = re.search(r":(\d+):", tile)
        if not match:
            continue

        tile_num = int(match.group(1))
        if tile_num not in DESIRED_TILE_NUMS:
            continue

        df_tile = df[df["tile"] == tile].copy()
        tile_dir = OUTPUT_DIR / f"tile_{tile}"
        tile_dir.mkdir(parents=True, exist_ok=True)

        counts = df_tile["final_pat_code"].value_counts()
        keep = counts[counts > MIN_SEQS_PER_PATIENT].index
        df_tile = df_tile[df_tile["final_pat_code"].isin(keep)].copy()
        if df_tile.empty:
            continue

        exclude = EXCLUDE_BY_TILE.get(tile_num, set())
        conditions = [col for col in ALL_CONDITIONS if col not in exclude]

        for patient in df_tile["final_pat_code"].unique():
            sp = df_tile[df_tile["final_pat_code"] == patient].copy()
            if len(sp) <= MIN_SEQS_PER_PATIENT:
                continue

            dup_counts = sp["isolate_sequence"].value_counts()
            sp = sp.drop_duplicates(subset=["isolate_sequence"]).copy()
            sp["dup_count"] = sp["isolate_sequence"].map(dup_counts).fillna(1).astype(int)
            if len(sp) < MIN_UNIQUE_SEQS:
                continue

            csv_path = tile_dir / f"patient_{patient}.csv"
            sp.to_csv(csv_path, index=False)
            all_csv_records.append({"tile": tile, "patient": patient, "path": csv_path})

            sp["seq_idx"] = sp.groupby("final_pat_code").cumcount() + 1
            sp["patient_seq"] = sp["final_pat_code"].astype(str) + "_" + sp["seq_idx"].astype(str)
            sp = sp.sort_values("seq_idx")

            sequences = sp["isolate_sequence"].tolist()
            labels = sp["patient_seq"].tolist()

            aligned_seqs, aligned_labels = run_mafft(sequences, labels)
            sp_aligned = sp.set_index("patient_seq").loc[aligned_labels]

            pdf_path = tile_dir / f"patient_{patient}.pdf"
            plot_patient(
                tile=tile,
                patient=patient,
                sp_aligned=sp_aligned,
                aligned_seqs=aligned_seqs,
                conditions_for_tile=conditions,
                out_path=pdf_path,
            )

        pdf_files = list(tile_dir.glob("*.pdf"))
        if pdf_files:
            zip_path = tile_dir / f"{tile}_all_patients_pdfs.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for pdf_file in pdf_files:
                    archive.write(pdf_file, arcname=pdf_file.name)
            print(f"Zipped {len(pdf_files)} PDFs to {zip_path}")

    if all_csv_records:
        combined_path = OUTPUT_DIR / "all_patients_combined.csv"
        frames = []
        for record in all_csv_records:
            tmp = pd.read_csv(record["path"])
            tmp["tile"] = record["tile"]
            tmp["patient"] = record["patient"]
            frames.append(tmp)
        pd.concat(frames, ignore_index=True).to_csv(combined_path, index=False)
        print(f"Saved {combined_path}")

    csv_files = list(OUTPUT_DIR.glob("tile_*/patient_*.csv"))
    if csv_files:
        all_data = []
        for csv_path in csv_files:
            try:
                tmp = pd.read_csv(csv_path)
                tmp["tile"] = csv_path.parent.name
                tmp["patient"] = csv_path.stem.replace("patient_", "")
                all_data.append(tmp)
            except Exception as exc:
                print(f"Warning: could not read {csv_path}: {exc}")

        big_csv = OUTPUT_DIR / "all_tiles_all_patients_combined.csv"
        pd.concat(all_data, ignore_index=True).to_csv(big_csv, index=False)
        print(f"Saved {big_csv}")
    else:
        print("No per-patient CSVs found; skipping all-tiles combined output.")


if __name__ == "__main__":
    main()