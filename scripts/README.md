# Execution Scripts

Utilities for cluster preflight checks, teacher verification, arm comparison, and SLURM launch. Training and evaluation entrypoints live at the repository root in `train_grpo.py` and `evaluate.py`.

## Scripts

- `check_env.py`: validates Python packages, CUDA visibility, `nvcc`, `ninja`, `compute-sanitizer`, disk space, YAML configs, and pipeline imports. Use `--allow_cpu` for laptop checks and `--json` for machine readable output.
- `verify_teachers.py`: compiles, verifies, and benchmarks every stored teacher kernel, then writes `data/teacher_measurements.json`. That file overrides the placeholder speedups used by the prompt builder.
- `compare_arms.py`: aligns Arm A and Arm B evaluation summaries and reports paired per-task differences, a bootstrap interval for the median difference, and split level correctness.
- `slurm/train_grpo.slurm`: two GPU SLURM job that runs preflight, both training arms, both evaluations, and the comparison.
- `slurm/train_grpo_apptainer.slurm`: the same workflow inside an Apptainer image.

## Typical Order

```bash
python scripts/check_env.py --config configs/multiturn_grpo.yaml
python scripts/verify_teachers.py --config configs/multiturn_grpo.yaml
python train_grpo.py --arm both --config configs/multiturn_grpo.yaml
python evaluate.py --config configs/multiturn_grpo.yaml
python scripts/compare_arms.py \
  --arm_a outputs/eval_results/arm_a.json \
  --arm_b outputs/eval_results/arm_b.json
```

`verify_teachers.py` requires a GPU with a working CUDA toolkit. It writes unverified entries only to the console and leaves the measurement file unchanged for failed kernels.
