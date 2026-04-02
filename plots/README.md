# MPRA Patient-Isolate Analysis Pipeline

This repository contains three analysis scripts for patient-associated HIV-1 MPRA data:

1. `01_clade_dot_heatmaps.py`  
   Generates isolate-level clade summaries and per-tile activation dot heatmaps.

2. `02_obtain_PWH_isolates.py`  
   Generates patient-level DNA/RNA-stratified matrices, scatter plots, and per-patient variability summaries.

3. `03_PWH_seq_align.py`  
   Generates per-patient MAFFT-aligned sequence heatmaps combined with log2-fold-change panels across stimulation conditions.

##Create the conda environment:

```bash
conda env create -f environment.yml
conda activate mpra-env

