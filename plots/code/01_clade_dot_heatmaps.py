#!/usr/bin/env python3
"""
Isolate-level MPRA clade analysis and per-tile activation dot heatmaps.

This script filters HIV-1 isolate measurements, collapses subtype labels to major
clades, exports processed isolate-level tables, and generates per-tile activation
dot heatmaps for selected tiles.

Inputs
------
data/patient_isolates_metadata_psdcountactivity.csv

Outputs
-------
results/clade_dot_heatmaps/
"""

from pathlib import Path
import math
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_CSV = DATA_DIR / "patient_isolates_metadata_psdcountactivity.csv"

OUT_ROOT = RESULTS_DIR / "clade_dot_heatmaps"
OUT_ROOT_PER_TILE = OUT_ROOT / "per_tile"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_ROOT_PER_TILE.mkdir(parents=True, exist_ok=True)

CLADE_ORDER = ["A", "B", "C", "D", "01_AE", "F", "G", "H", "O"]

CONDITIONS = [
    "psdcnt_Stim_log2FoldChange",
    "psdcnt_TNF_log2FoldChange",
    "psdcnt_IFNG_log2FoldChange",
]

KEEP_TILES = {"6", "9", "11", "13"}
ROW_TILE_ORDER = ["6", "9", "11", "13"]

MIN_DNA_MEAN = 50.0
GLOBAL_CI_MAX = 2.0
MAX_RADIUS_PT = 15.0
MIN_RADIUS_PT = 2.0

CMAP_FC = sns.color_palette("vlag", n_colors=256, as_cmap=True)
NORM_FC = mpl.colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

SUBTYPE_MAP = {
    "A1": "A",
    "A2": "A",
    "A4": "A",
    "A6": "A",
    "F1": "F",
    "F2": "F",
    "F1F2": "F",
}


def collapse_subtype(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().upper()
    if value == "01_AE":
        return "01_AE"
    if value.startswith("A"):
        return "A"
    if value in {"F", "F1", "F2", "F1F2"}:
        return "F"
    return value


def parse_tile_index(value):
    value = str(value)
    match = re.search(r":(\d{1,3})(?=:[\+\-])", value)
    if match:
        return match.group(1)
    for token in value.split(":"):
        if token.isdigit():
            return token
    return None


def compute_p95_p5(series):
    values = pd.to_numeric(series, errors="coerce").dropna().values
    if len(values) == 0:
        return np.nan
    p5, p95 = np.percentile(values, [5, 95])
    return p95 - p5


def compute_iqr(series):
    series = pd.to_numeric(series, errors="coerce")
    return series.quantile(0.75) - series.quantile(0.25)


def radius_from_width(width):
    radius = ((MAX_RADIUS_PT - MIN_RADIUS_PT) * width / GLOBAL_CI_MAX) + MIN_RADIUS_PT
    return np.clip(radius, MIN_RADIUS_PT, MAX_RADIUS_PT)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    if "tile" not in df.columns:
        raise ValueError("Input file is missing required column: tile")
    if "Strain_Subtype" not in df.columns:
        raise ValueError("Input file is missing required column: Strain_Subtype")
    if "dna_mean" not in df.columns:
        raise ValueError("Input file is missing required column: dna_mean")

    missing_conditions = [col for col in CONDITIONS if col not in df.columns]
    if missing_conditions:
        raise ValueError(f"Input file is missing required condition columns: {missing_conditions}")

    df = df[df["tile"].astype(str).str.contains("HIV", case=False, na=False)].copy()
    df["dna_mean"] = pd.to_numeric(df["dna_mean"], errors="coerce")
    df = df[df["dna_mean"] >= MIN_DNA_MEAN].copy()

    df["clade_post"] = df["Strain_Subtype"].apply(collapse_subtype)
    df = df[df["clade_post"].isin(CLADE_ORDER)].copy()

    df["tile_idx"] = df["tile"].apply(parse_tile_index)
    df = df[df["tile_idx"].isin(KEEP_TILES)].copy()

    for col in CONDITIONS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main() -> None:
    df = load_data(INPUT_CSV)

    df.to_csv(OUT_ROOT / "isolate_clade_table.csv", index=False)

    group_cols = ["tile_idx", "clade_post"]

    median = df.groupby(group_cols)[CONDITIONS].median().unstack("clade_post")
    iqr = df.groupby(group_cols)[CONDITIONS].agg(compute_iqr).unstack("clade_post")
    ci = df.groupby(group_cols)[CONDITIONS].agg(compute_p95_p5).unstack("clade_post")

    grouped_all = df.groupby("tile_idx")
    all_median = grouped_all[CONDITIONS].median()
    all_iqr = grouped_all[CONDITIONS].agg(compute_iqr)
    all_ci = grouped_all[CONDITIONS].agg(compute_p95_p5)

    for tile in ROW_TILE_ORDER:
        if tile not in median.index:
            continue

        fig, ax = plt.subplots(figsize=(6, 2.5))
        clades = ["All"] + [c for c in CLADE_ORDER if c in median.columns.get_level_values(1)]

        for i, condition in enumerate(CONDITIONS):
            for j, clade in enumerate(clades):
                if clade == "All":
                    median_value = all_median.loc[tile, condition]
                    width_value = all_ci.loc[tile, condition]
                else:
                    if (condition, clade) not in median.columns:
                        continue
                    median_value = median.loc[tile, (condition, clade)]
                    width_value = ci.loc[tile, (condition, clade)]

                if pd.isna(median_value) or pd.isna(width_value):
                    continue

                radius = radius_from_width(width_value)

                ax.scatter(
                    j,
                    i,
                    s=math.pi * radius**2,
                    color=CMAP_FC(NORM_FC(median_value)),
                    edgecolor="black",
                    linewidth=0.5,
                )

        ax.set_xticks(range(len(clades)))
        ax.set_xticklabels(clades, rotation=45, ha="right")
        ax.set_yticks(range(len(CONDITIONS)))
        ax.set_yticklabels(["Stim", "TNF", "IFNG"])
        ax.set_title(f"Tile {tile}")
        ax.set_xlim(-0.5, len(clades) - 0.5)
        ax.set_ylim(len(CONDITIONS) - 0.5, -0.5)

        scalar_mappable = mpl.cm.ScalarMappable(norm=NORM_FC, cmap=CMAP_FC)
        scalar_mappable.set_array([])
        fig.colorbar(scalar_mappable, ax=ax, fraction=0.03, pad=0.02)

        plt.tight_layout()
        plt.savefig(OUT_ROOT_PER_TILE / f"tile_{tile}.pdf")
        plt.close()

        rows = []
        for condition in CONDITIONS:
            for clade in clades:
                if clade == "All":
                    median_value = all_median.loc[tile, condition]
                    iqr_value = all_iqr.loc[tile, condition]
                    width_value = all_ci.loc[tile, condition]
                else:
                    if (condition, clade) not in median.columns:
                        continue
                    median_value = median.loc[tile, (condition, clade)]
                    iqr_value = iqr.loc[tile, (condition, clade)]
                    width_value = ci.loc[tile, (condition, clade)]

                if pd.isna(median_value):
                    continue

                radius = radius_from_width(width_value)
                rows.append(
                    [
                        tile,
                        condition,
                        clade,
                        median_value,
                        iqr_value,
                        width_value,
                        radius,
                    ]
                )

        df_out = pd.DataFrame(
            rows,
            columns=[
                "tile_idx",
                "condition",
                "clade",
                "median_log2FC",
                "IQR",
                "P95_P5",
                "radius_pt",
            ],
        )
        df_out.to_csv(OUT_ROOT_PER_TILE / f"tile_{tile}_table.csv", index=False)


if __name__ == "__main__":
    main()