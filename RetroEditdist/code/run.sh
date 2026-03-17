#!/bin/bash
#$ -N run
#$ -cwd
#$ -o ../logs/run.out
#$ -e ../logs/run.err
#$ -l h_rt=48:00:00
#$ -l h_vmem=128G
#$ -pe smp 16


# Activate Conda env
source /projectnb/vcres/myousry/miniconda3/etc/profile.d/conda.sh
conda activate mpra

python analyze.py \
  --plot-only \
  --pairwise ../results/.pairwise.csv \
  --out-prefix ../results/ \
  --pdf