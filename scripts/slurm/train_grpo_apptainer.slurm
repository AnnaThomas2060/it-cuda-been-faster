#!/bin/bash
#SBATCH --job-name=cuda-grpo-sif
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
# Adjust partition, account, and GPU type. Set APPTAINER_IMAGE before sbatch.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs/eval_results

IMAGE="${APPTAINER_IMAGE:?Set APPTAINER_IMAGE to the container image path}"

apptainer exec --nv \
  --bind "${PWD}:/workspace" \
  --bind "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    cd /workspace
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TOKENIZERS_PARALLELISM=false
    python run_pipeline.py --profile rtx3090 --output_root outputs/container_run
  '
