#!/bin/bash
#SBATCH -p gpu_ai
#SBATCH -J locomo_one
#SBATCH -n 1
#SBATCH -o /share/home/tanylang/locomo/logs/locomo_one_%j.log
#SBATCH -e /share/home/tanylang/locomo/logs/locomo_one_%j.err

set -e

module load singularity/4.0.2

export NO_PROXY=192.168.10.25,localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export SINGULARITYENV_NO_PROXY="$NO_PROXY"
export SINGULARITYENV_no_proxy="$no_proxy"

singularity exec \
  --bind /share:/share \
  /share/home/tanylang/singularity/python-312.sif \
  /share/home/tanylang/gpt_researcher/venv/bin/python \
  /share/home/tanylang/locomo/run_locomo_one.py
