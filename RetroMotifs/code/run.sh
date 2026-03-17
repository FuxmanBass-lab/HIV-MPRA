#!/bin/bash
#$ -N run_motif
#$ -cwd
#$ -o ../logs/run.out
#$ -e ../logs/run.err
#$ -l h_rt=24:00:00
#$ -l h_vmem=128G
#$ -pe smp 12

set -euo pipefail

# ------------------------------
# Init environment modules if available
# ------------------------------
if [ -f /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
  . /usr/share/Modules/init/bash
fi
has_module() { command -v module >/dev/null 2>&1; }

# ------------------------------
# Activate Conda env
# ------------------------------
export CONDA_SOLVER=classic
set +u
source /projectnb/vcres/myousry/miniconda3/etc/profile.d/conda.sh
conda activate mpra
set -u
export target_platform=${target_platform:-}

# ------------------------------
# Ensure FIMO is available and get its absolute path
# ------------------------------
if has_module; then
  # Load a MEME Suite module if present (non-fatal)
  module load meme/5.5.5 || true
fi

FIMO_BIN="$(command -v fimo || true)"
if [ -z "$FIMO_BIN" ]; then
  # Try the conda env directly
  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/fimo" ]; then
    FIMO_BIN="$CONDA_PREFIX/bin/fimo"
  fi
fi

if [ -z "$FIMO_BIN" ]; then
  echo "[error] Could not locate the FIMO executable.\n" \
       "       Options:\n" \
       "       • Load a MEME module (e.g.,: module load meme/5.5.5)\n" \
       "       • Or: conda install -n mpra -c bioconda meme=5.5.5\n" \
       "       • Or: set FIMO_BIN to an absolute path." >&2
  exit 1
fi

echo "[info] Using FIMO at: $FIMO_BIN"
"$FIMO_BIN" --version || true

# ------------------------------
# Paths
# ------------------------------
MEME_FILE="../results/tf_panel/jaspar_panel_filtered.meme"
TFS_FILE="../data/TFs.tsv"
TILES_DIR="../results/tiles"
OUTDIR="../results/motif_counts"
PRECEDENCE="../data/family_precedence.tsv"
COMPOSITES="../data/family_composites.tsv"

# ------------------------------
# Run pipeline (FIMO mode)
# ------------------------------
python 02_motif_scans.py \
  --use_fimo \
  --fimo_bin "$FIMO_BIN" \
  --meme "$MEME_FILE" \
  --tfs  "$TFS_FILE" \
  --tiles_dir "$TILES_DIR" \
  --outdir "$OUTDIR" \
  --precedence "$PRECEDENCE" \
  --composites "$COMPOSITES" \
  --p_thresh 1e-4 \
  --q_thresh 0.10 \
  --nms_factor 0.8 \
  --tf_dedupe_bp 8 --tf_iou 0.4 --tf_shift_bp 2 \
  --family_dedupe_bp 10 --family_iou 0.4 --family_shift_bp 2


  # NOTE: The pipeline applies a minimum FIMO score of 10 in addition to the p-value threshold (see 02_motif_scans.py).