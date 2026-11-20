#!/bin/bash
#SBATCH --job-name=cuda-grpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
# Adjust the partition, account, and GPU type for your cluster.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module purge
module load cuda/12.1
module load python/3.11
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

python run_pipeline.py --profile rtx3090 --output_root outputs/slurm_run
