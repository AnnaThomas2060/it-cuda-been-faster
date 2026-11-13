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

python scripts/check_env.py --config configs/multiturn_grpo.yaml

python train_grpo.py \
  --arm both \
  --mode multi_turn \
  --config configs/multiturn_grpo.yaml \
  --dispatcher subprocess \
  --output_dir outputs/slurm_run

python evaluate.py \
  --config configs/multiturn_grpo.yaml \
  --adapter_path outputs/slurm_run/arm_a/final_adapter \
  --prompt_mode contrastive \
  --dispatcher subprocess \
  --output_file outputs/eval_results/arm_a.json

python evaluate.py \
  --config configs/multiturn_grpo.yaml \
  --adapter_path outputs/slurm_run/arm_b/final_adapter \
  --prompt_mode reference_only \
  --dispatcher subprocess \
  --output_file outputs/eval_results/arm_b.json

python scripts/compare_arms.py \
  --arm_a outputs/eval_results/arm_a.json \
  --arm_b outputs/eval_results/arm_b.json \
  --split all \
  --output outputs/eval_results/comparison.json
