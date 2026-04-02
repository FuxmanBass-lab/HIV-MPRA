#!/usr/bin/env python3
"""
Patient-level MPRA analysis with DNA/RNA stratification, matrix export, and plotting.

This script processes HIV-1 tile measurements from a single input table, applies
patient-level filtering, collapses subtype labels to major clades for display,
generates patient-grouped matrices, and exports DNA- and RNA-specific summary plots.

Inputs
------
data/patient_isolates_metadata_psdcountactivity.csv

Outputs
-------
results/patient_scatters/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_PATH = DATA_DIR / "patient_isolates_metadata_psdcountactivity.csv"
OUTPUT_DIR = RESULTS_DIR / "patient_scatters"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CONFIG = {
    "point_alpha": 0.5,
    "dna_color": "#1f77b4",
    "rna_color": "#d62728",
    "tile_prefix": "HIV-1",
    "min_dna_mean": 50.0,
    "min_obs_per_patient_per_tile": 5,
    "min_obs_per_patient_per_tile_dna": 5,
    "min_obs_per_patient_per_tile_rna": 5,
    "drop_invalid_patient_codes": True,
    "invalid_patient_codes": {"", "0", "na", "#na", "nan", "None"},
}


VALUE_COLUMNS = [
    "BE_Unstim_log2FoldChange",
    "CD4_log2FoldChange",
    "psdcnt_TNF_log2FoldChange",
    "psdcnt_IFNG_log2FoldChange",
    "psdcnt_Stim_log2FoldChange",
]

REQUIRED_COLUMNS = [
    "accession",
    "final_pat_code",
    "dna_mean",
    "Strain_Subtype",
    "tile",
    "molecule_type",
    *VALUE_COLUMNS,
]

TRUE_CLADE_SET = {"A", "B", "C", "D", "E", "F", "G", "H", "O", "01_AE"}
CLADE_PRIORITY = ["A", "B", "C", "01_AE", "D", "E", "F", "G", "H", "O"]


def normalize_moltype(value) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    s = str(value).strip().lower()
    if not s:
        return "Unknown"
    if "dna" in s:
        return "DNA"
    if "rna" in s:
        return "RNA"
    return "Unknown"


def load_input_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    if "Accession" in df.columns and "accession" not in df.columns:
        df = df.rename(columns={"Accession": "accession"})
    if "ACCESSION" in df.columns and "accession" not in df.columns:
        df = df.rename(columns={"ACCESSION": "accession"})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["dna_mean"] = pd.to_numeric(df["dna_mean"], errors="coerce")
    df["molecule_type"] = df["molecule_type"].apply(normalize_moltype)

    for col in VALUE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def is_known_subtype(value) -> bool:
    if value is None or pd.isna(value):
        return False
    s = str(value).strip()
    if not s:
        return False
    return s.lower() not in {"unknown", "nan", "none", "na", "n/a"}


def collapse_subtype_label(value) -> str:
    if value is None or pd.isna(value):
        return "Unknown"

    s = str(value).strip()
    if not s:
        return "Unknown"

    token = s.split("+", 1)[0].strip().upper()

    if token == "01_AE":
        return "01_AE"
    if token.startswith("A"):
        return "A"
    if token in {"F", "F1", "F2", "F1F2"}:
        return "F"

    return token


def pick_patient_label(subtypes) -> str:
    values = [collapse_subtype_label(v) for v in subtypes if is_known_subtype(v)]
    values = [v for v in values if v != "Unknown"]
    if not values:
        return "Unknown"

    def sort_key(label: str):
        if label in CLADE_PRIORITY:
            return (0, CLADE_PRIORITY.index(label), label)
        return (1, 999, label)

    return sorted(set(values), key=sort_key)[0]


def is_recombinant_label(label: str) -> bool:
    collapsed = collapse_subtype_label(label)
    return collapsed not in TRUE_CLADE_SET


def patient_sort_key(column_name: str):
    label = ""
    if "|" in column_name:
        _, label = column_name.split("|", 1)

    collapsed = collapse_subtype_label(label)

    if collapsed == "Unknown":
        return (999, 2, "Unknown", column_name)

    if collapsed in CLADE_PRIORITY:
        return (CLADE_PRIORITY.index(collapsed), 0, collapsed, column_name)

    recombinant_rank = 0 if is_recombinant_label(collapsed) else 1
    return (len(CLADE_PRIORITY), recombinant_rank, collapsed, column_name)


def classify_patient(row) -> str:
    has_dna = row.get("DNA", 0) > 0
    has_rna = row.get("RNA", 0) > 0
    if has_dna and has_rna:
        return "DNA+RNA"
    if has_dna:
        return "DNA-only"
    if has_rna:
        return "RNA-only"
    return "Unknown"


def dna_status(n_dna: int) -> str:
    if n_dna <= 0:
        return "dropped_no_dna"
    if n_dna < CONFIG["min_obs_per_patient_per_tile_dna"]:
        return f"dropped_lt{CONFIG['min_obs_per_patient_per_tile_dna']}_dna"
    return f"kept_ge{CONFIG['min_obs_per_patient_per_tile_dna']}_dna"


def rna_status(n_rna: int) -> str:
    if n_rna <= 0:
        return "dropped_no_rna"
    if n_rna < CONFIG["min_obs_per_patient_per_tile_rna"]:
        return f"dropped_lt{CONFIG['min_obs_per_patient_per_tile_rna']}_rna"
    return f"kept_ge{CONFIG['min_obs_per_patient_per_tile_rna']}_rna"


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_numeric_array(series_or_array) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(series_or_array), errors="coerce").dropna().to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text))


def apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["dna_mean"] >= CONFIG["min_dna_mean"]].copy()
    out = out.dropna(subset=["final_pat_code"])
    out["final_pat_code"] = out["final_pat_code"].astype(str).str.strip()

    if CONFIG["drop_invalid_patient_codes"]:
        out = out[~out["final_pat_code"].isin(CONFIG["invalid_patient_codes"])].copy()

    return out


def build_matrix(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    work = df[["final_pat_code", "Strain_Subtype", value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["final_pat_code", value_col])

    patient_label_map = (
        work.groupby("final_pat_code")["Strain_Subtype"]
        .apply(pick_patient_label)
        .to_dict()
    )

    grouped = work.groupby("final_pat_code", dropna=False)[value_col].apply(list)
    if grouped.empty:
        return pd.DataFrame()

    columns = {}
    max_len = 0

    for patient_id, values in grouped.items():
        if not isinstance(values, list):
            values = [values]
        patient_label = patient_label_map.get(patient_id, "Unknown")
        column_name = f"{patient_id}|{patient_label}"
        columns[column_name] = values
        max_len = max(max_len, len(values))

    matrix = pd.DataFrame(
        {
            column_name: values + [np.nan] * (max_len - len(values))
            for column_name, values in columns.items()
        }
    )

    for col in matrix.columns:
        matrix[col] = pd.to_numeric(matrix[col], errors="coerce")

    return matrix[sorted(matrix.columns, key=patient_sort_key)]


def save_matrix(df: pd.DataFrame, value_col: str, output_path: Path) -> pd.DataFrame:
    matrix = build_matrix(df, value_col)
    matrix.to_csv(output_path, index=True)
    return matrix


def plot_matrix_scatter(
    df_dna: pd.DataFrame,
    df_rna: pd.DataFrame,
    title: str,
    outbase: Path,
    y_label: str,
) -> None:
    plt.figure(figsize=(22, 10))
    x_positions = np.arange(max(len(df_dna.columns), len(df_rna.columns)))

    for i, col in enumerate(df_dna.columns):
        y_vals = clean_numeric_array(df_dna[col])
        x_vals = np.full(len(y_vals), i - 0.15)
        plt.scatter(
            x_vals,
            y_vals,
            alpha=CONFIG["point_alpha"],
            s=35,
            color=CONFIG["dna_color"],
            edgecolors="none",
            label="DNA" if i == 0 else None,
        )

    for i, col in enumerate(df_rna.columns):
        y_vals = clean_numeric_array(df_rna[col])
        x_vals = np.full(len(y_vals), i + 0.15)
        plt.scatter(
            x_vals,
            y_vals,
            alpha=CONFIG["point_alpha"],
            s=35,
            color=CONFIG["rna_color"],
            edgecolors="none",
            label="RNA" if i == 0 else None,
        )

    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("Patients")
    plt.xticks(x_positions, x_positions + 1, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outbase.with_suffix(".png"), dpi=300)
    plt.savefig(outbase.with_suffix(".pdf"))
    plt.close()


def plot_matrix_scatter_single(
    df: pd.DataFrame,
    title: str,
    outbase: Path,
    y_label: str,
    molecule: str,
) -> None:
    color = CONFIG["dna_color"] if molecule == "DNA" else CONFIG["rna_color"]

    plt.figure(figsize=(22, 10))
    for i, col in enumerate(df.columns):
        y_vals = clean_numeric_array(df[col])
        x_vals = np.full(len(y_vals), i)
        plt.scatter(
            x_vals,
            y_vals,
            alpha=CONFIG["point_alpha"],
            s=35,
            color=color,
            edgecolors="none",
        )

    plt.title(f"{title} — {molecule}")
    plt.ylabel(y_label)
    plt.xlabel("Patients")
    plt.xticks(np.arange(len(df.columns)), df.columns, rotation=90)
    plt.tight_layout()
    plt.savefig(outbase.with_suffix(".png"), dpi=300)
    plt.savefig(outbase.with_suffix(".pdf"))
    plt.close()


def plot_std_violin(
    df_dna: pd.DataFrame,
    df_rna: pd.DataFrame,
    title: str,
    outbase: Path,
    y_label: str,
) -> None:
    stds_dna = clean_numeric_array(df_dna.apply(numeric_series).std(axis=0, skipna=True))
    stds_rna = clean_numeric_array(df_rna.apply(numeric_series).std(axis=0, skipna=True))

    std_frames = []
    if len(stds_dna) > 0:
        std_frames.append(pd.DataFrame({"molecule": "DNA", "std": stds_dna}))
    if len(stds_rna) > 0:
        std_frames.append(pd.DataFrame({"molecule": "RNA", "std": stds_rna}))

    if std_frames:
        pd.concat(std_frames, ignore_index=True).to_csv(
            outbase.with_name(outbase.name + "_per_patient_std_combined.csv"),
            index=False,
        )

    plt.figure(figsize=(6, 8))
    parts = []
    positions = []

    if len(stds_dna) >= 2:
        parts.append(stds_dna)
        positions.append(1)
    if len(stds_rna) >= 2:
        parts.append(stds_rna)
        positions.append(2)

    if len(parts) > 0:
        plt.violinplot(parts, positions=positions, showmeans=False, showmedians=True)

    if len(stds_dna) > 0:
        jitter = np.random.normal(loc=1.0, scale=0.03, size=len(stds_dna))
        plt.scatter(
            jitter,
            stds_dna,
            s=60,
            alpha=0.7,
            color=CONFIG["dna_color"],
            label="DNA",
        )

    if len(stds_rna) > 0:
        jitter = np.random.normal(loc=2.0, scale=0.03, size=len(stds_rna))
        plt.scatter(
            jitter,
            stds_rna,
            s=60,
            alpha=0.7,
            color=CONFIG["rna_color"],
            label="RNA",
        )

    plt.ylabel(f"Per-patient standard deviation of {y_label}")
    plt.title(title)
    plt.xticks([1, 2], ["DNA", "RNA"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(outbase.with_name(outbase.name + "_std_violin.png"), dpi=300)
    plt.savefig(outbase.with_name(outbase.name + "_std_violin.pdf"))
    plt.close()


def run_plots_for_tile(
    tile: str,
    matrix_dict: dict[str, dict[str, pd.DataFrame]],
    no_split_plots_dir: Path,
    split_plots_dir: Path,
) -> None:
    specs = {
        "BE_Unstim_log2FoldChange": {
            "title": f"{tile} — Jurkat basal activity",
            "y_label": "BE_Unstim_log2FoldChange",
            "tag": "UNSTIM",
        },
        "CD4_log2FoldChange": {
            "title": f"{tile} — Primary CD4 activity",
            "y_label": "CD4_log2FoldChange",
            "tag": "CD4",
        },
        "psdcnt_TNF_log2FoldChange": {
            "title": f"{tile} — TNF response",
            "y_label": "psdcnt_TNF_log2FoldChange",
            "tag": "TNF",
        },
        "psdcnt_IFNG_log2FoldChange": {
            "title": f"{tile} — IFNG response",
            "y_label": "psdcnt_IFNG_log2FoldChange",
            "tag": "IFNG",
        },
        "psdcnt_Stim_log2FoldChange": {
            "title": f"{tile} — Stim response",
            "y_label": "psdcnt_Stim_log2FoldChange",
            "tag": "STIM",
        },
    }

    for value_col, spec in specs.items():
        df_no_split = matrix_dict[value_col]["NO_SPLIT"]
        df_dna = matrix_dict[value_col]["DNA"]
        df_rna = matrix_dict[value_col]["RNA"]

        if not df_no_split.empty:
            no_split_base = no_split_plots_dir / f"{tile}_{spec['tag']}_all_patients"
            plot_matrix_scatter_single(df_no_split, spec["title"], no_split_base, spec["y_label"], "All")

        if not df_dna.empty and not df_rna.empty:
            combined_base = split_plots_dir / f"{tile}_{spec['tag']}_DNA_vs_RNA"
            plot_matrix_scatter(df_dna, df_rna, spec["title"], combined_base, spec["y_label"])
            plot_std_violin(df_dna, df_rna, spec["title"], combined_base, spec["y_label"])

        if not df_dna.empty:
            dna_base = split_plots_dir / f"{tile}_{spec['tag']}_DNA_only"
            plot_matrix_scatter_single(df_dna, spec["title"], dna_base, spec["y_label"], "DNA")

        if not df_rna.empty:
            rna_base = split_plots_dir / f"{tile}_{spec['tag']}_RNA_only"
            plot_matrix_scatter_single(df_rna, spec["title"], rna_base, spec["y_label"], "RNA")


def main() -> None:
    df = load_input_table(INPUT_PATH)
    df = apply_global_filters(df)

    summary_dir = OUTPUT_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(summary_dir / "patient_filtered_activity.csv", index=False)

    clade_counts = df.groupby("final_pat_code")["Strain_Subtype"].nunique()
    multi_clade_patients = clade_counts[clade_counts > 1].index
    df[df["final_pat_code"].isin(multi_clade_patients)].to_csv(
        summary_dir / "patients_multi_clade_activity.csv",
        index=False,
    )

    hiv1_tiles = sorted(
        t for t in df["tile"].dropna().unique() if str(t).startswith(CONFIG["tile_prefix"])
    )

    for tile in hiv1_tiles:
        tile_df = df[df["tile"] == tile].copy()

        patient_counts = tile_df["final_pat_code"].value_counts()
        eligible_patients = patient_counts[
            patient_counts >= CONFIG["min_obs_per_patient_per_tile"]
        ].index
        tile_df = tile_df[tile_df["final_pat_code"].isin(eligible_patients)].copy()

        if tile_df.empty:
            continue

        tile_dir = OUTPUT_DIR / f"tile_{safe_name(tile)}"
        no_split_dir = tile_dir / "dna_rna_no_split"
        split_dir = tile_dir / "dna_rna_split"

        no_split_matrices_dir = no_split_dir / "matrices"
        no_split_plots_dir = no_split_dir / "plots"

        split_matrices_dir = split_dir / "matrices"
        split_plots_dir = split_dir / "plots"
        split_dna_matrices_dir = split_matrices_dir / "DNA"
        split_rna_matrices_dir = split_matrices_dir / "RNA"

        for directory in [
            no_split_matrices_dir,
            no_split_plots_dir,
            split_matrices_dir,
            split_plots_dir,
            split_dna_matrices_dir,
            split_rna_matrices_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        split_df = tile_df.copy()
        split_df["molecule_type"] = split_df["molecule_type"].apply(normalize_moltype)
        split_df.to_csv(split_dir / f"patient_filtered_{safe_name(tile)}_with_molecule_type.csv", index=False)

        accession_counts = (
            split_df["molecule_type"]
            .value_counts(dropna=False)
            .rename_axis("molecule_type")
            .reset_index(name="count")
        )
        accession_counts.to_csv(split_dir / f"accession_dna_rna_distribution_{safe_name(tile)}.csv", index=False)

        plt.figure(figsize=(4, 4))
        accession_counts.set_index("molecule_type")["count"].plot(kind="bar")
        plt.title(f"{tile} accession classes")
        plt.ylabel("Number of accessions")
        plt.xlabel("Molecule type")
        plt.tight_layout()
        plt.savefig(split_dir / f"accession_dna_rna_distribution_{safe_name(tile)}.png", dpi=300)
        plt.close()

        patient_molecule_counts = (
            split_df.groupby(["final_pat_code", "molecule_type"]).size().unstack(fill_value=0)
        )
        patient_molecule_counts["category"] = patient_molecule_counts.apply(classify_patient, axis=1)
        category_counts = (
            patient_molecule_counts["category"]
            .value_counts()
            .rename_axis("category")
            .reset_index(name="n_patients")
        )
        category_counts.to_csv(split_dir / f"patient_dna_rna_category_counts_{safe_name(tile)}.csv", index=False)

        plt.figure(figsize=(4, 4))
        category_counts.set_index("category")["n_patients"].plot(kind="bar")
        plt.title(f"{tile} patient classes")
        plt.ylabel("Number of patients")
        plt.xlabel("Category")
        plt.tight_layout()
        plt.savefig(split_dir / f"patient_dna_rna_category_counts_{safe_name(tile)}.png", dpi=300)
        plt.close()

        dna_df_all = split_df[split_df["molecule_type"] == "DNA"].copy()
        rna_df_all = split_df[split_df["molecule_type"] == "RNA"].copy()

        patient_total = split_df.groupby("final_pat_code").size().rename("n_total").reset_index()
        patient_dna = dna_df_all.groupby("final_pat_code").size().rename("n_dna").reset_index()
        patient_rna = rna_df_all.groupby("final_pat_code").size().rename("n_rna").reset_index()

        patient_report = (
            patient_total
            .merge(patient_dna, on="final_pat_code", how="left")
            .merge(patient_rna, on="final_pat_code", how="left")
        )
        patient_report["n_total"] = patient_report["n_total"].fillna(0).astype(int)
        patient_report["n_dna"] = patient_report["n_dna"].fillna(0).astype(int)
        patient_report["n_rna"] = patient_report["n_rna"].fillna(0).astype(int)
        patient_report["dna_matrix_status"] = patient_report["n_dna"].apply(dna_status)
        patient_report["rna_matrix_status"] = patient_report["n_rna"].apply(rna_status)

        patient_report.sort_values(["final_pat_code"]).to_csv(
            split_dir / f"patient_filter_report_{safe_name(tile)}.csv",
            index=False,
        )

        eligible_dna_patients = set(
            patient_report.loc[
                patient_report["dna_matrix_status"]
                == f"kept_ge{CONFIG['min_obs_per_patient_per_tile_dna']}_dna",
                "final_pat_code",
            ].astype(str)
        )
        eligible_rna_patients = set(
            patient_report.loc[
                patient_report["rna_matrix_status"]
                == f"kept_ge{CONFIG['min_obs_per_patient_per_tile_rna']}_rna",
                "final_pat_code",
            ].astype(str)
        )

        dna_df = dna_df_all[dna_df_all["final_pat_code"].astype(str).isin(eligible_dna_patients)].copy()
        rna_df = rna_df_all[rna_df_all["final_pat_code"].astype(str).isin(eligible_rna_patients)].copy()

        pd.DataFrame({"patient": sorted(eligible_dna_patients)}).to_csv(
            split_dir / f"patients_kept_ge{CONFIG['min_obs_per_patient_per_tile_dna']}_DNA_{safe_name(tile)}.csv",
            index=False,
        )
        pd.DataFrame({"patient": sorted(eligible_rna_patients)}).to_csv(
            split_dir / f"patients_kept_ge{CONFIG['min_obs_per_patient_per_tile_rna']}_RNA_{safe_name(tile)}.csv",
            index=False,
        )

        matrix_dict = {}
        for value_col in VALUE_COLUMNS:
            no_split_matrix = save_matrix(
                tile_df,
                value_col,
                no_split_matrices_dir / f"patient_{safe_name(value_col)}_matrix_{safe_name(tile)}.csv",
            )

            dna_matrix = save_matrix(
                dna_df,
                value_col,
                split_dna_matrices_dir / f"patient_{safe_name(value_col)}_matrix_{safe_name(tile)}_DNA.csv",
            )

            rna_matrix = save_matrix(
                rna_df,
                value_col,
                split_rna_matrices_dir / f"patient_{safe_name(value_col)}_matrix_{safe_name(tile)}_RNA.csv",
            )

            matrix_dict[value_col] = {
                "NO_SPLIT": no_split_matrix,
                "DNA": dna_matrix,
                "RNA": rna_matrix,
            }

        run_plots_for_tile(tile, matrix_dict, no_split_plots_dir, split_plots_dir)


if __name__ == "__main__":
    main()