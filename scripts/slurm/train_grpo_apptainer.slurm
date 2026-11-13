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
    python scripts/check_env.py --config configs/multiturn_grpo.yaml
    python train_grpo.py \
      --arm both \
      --mode multi_turn \
      --config configs/multiturn_grpo.yaml \
      --dispatcher subprocess \
      --output_dir outputs/container_run
    python evaluate.py \
      --config configs/multiturn_grpo.yaml \
      --adapter_path outputs/container_run/arm_a/final_adapter \
      --prompt_mode contrastive \
      --dispatcher subprocess \
      --output_file outputs/eval_results/arm_a.json
    python evaluate.py \
      --config configs/multiturn_grpo.yaml \
      --adapter_path outputs/container_run/arm_b/final_adapter \
      --prompt_mode reference_only \
      --dispatcher subprocess \
      --output_file outputs/eval_results/arm_b.json
    python scripts/compare_arms.py \
      --arm_a outputs/eval_results/arm_a.json \
      --arm_b outputs/eval_results/arm_b.json \
      --output outputs/eval_results/comparison.json
  '
