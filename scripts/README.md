# Execution Scripts

This directory houses executable runners for pipeline tasks, rollouts, and training branches.

## Runners
- `train_sft.py`: Stage 1 supervised fine-tuning using LoRA and SinkGD/AdamW.
- `train_rest.py`: Branch A iterative reinforced self-training loop.
- `train_multiturn_grpo.py`: Branch B multi-turn Dr. GRPO policy optimization.
- `train_contrastive_rl.py`: Branch C hybrid contrastive RL with archive pairing.
- `eval_benchmarks.py`: Evaluation runner over KernelBench and Terminal-Bench suites.
