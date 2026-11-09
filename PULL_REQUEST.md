# Pull Request: Implement RL Training Loop, Two-Tier Reward Function, and Evaluation Pipeline

## Assignee and Scope
* Author: Anna Thomas (Training and Reward Lead)
* Target Branch: `main`
* Feature Branch: `implement/rl-reward-eval-pipeline`

## Summary of Completed Work

This pull request delivers the core reinforcement learning training, reward design, and evaluation infrastructure for the project "It CUDA been faster: Automating GPU Optimizations using RL". The implementation enables a controlled comparison between Arm A (Contrastive Scored Context GRPO) and Arm B (Reference Only GRPO) to determine whether historical implementation conditioning improves custom CUDA kernel generation.

### 1. Data and Task Registry (`data/`)
* `data/tasks.py`: Implemented task definitions for Swish Activation, Column Maximum, Channel RMSNorm, and holdout tasks (Average Pooling 2D, Softmax, Cumulative Sum). Configured exact bitwise match validation for Column Maximum and 1e-4 tolerance checks for floating point operations.
* `data/example_store.py`: Integrated validated CUDA-L1 teacher kernels with baseline speedup metrics. Implemented graceful handling for single-teacher tasks (Channel RMSNorm).

### 2. Execution and Verification Engine (`engine/`)
* `engine/prompt_builder.py`: Implemented prompt construction for both experimental arms. Reference-only format enforces C++ signatures and interface contracts, while contrastive format injects historical implementations and measured speedup scores.
* `engine/compiler.py`: Built static syntax analysis and inline C++ compilation via PyTorch load_inline. Detects and rejects forbidden PyTorch mathematical shortcuts (e.g., torch::relu, torch::sigmoid, at::native) and assigns unique module names per compilation to prevent symbol collisions.
* `engine/verifier.py`: Built multi-shape and multi-seed (4 seeded runs) numerical validation suite. Checks output tensor shapes, dtypes, numerical tolerances, and verifies that input tensors are not modified in place.
* `engine/benchmark.py`: Implemented 7-block chronological latency timer using CUDA events. Computes block median ratios against PyTorch eager baselines, enforces a 0.005 population variance stability rejection gate, and floors valid speedups to two decimal places.

### 3. Reward Pipeline and Model Architecture (`rewards/`, `models/`, `configs/`)
* `rewards/grpo_reward.py`: TRL-compatible reward callback executing the two-stage gate. Includes failure taxonomy tracking (static check, compilation, runtime crash, numerical divergence, timing instability) and explicitly logs zero-reward groups.
* `models/loader.py`: Configured 4-bit NormalFloat (NF4) quantized base model loading with double quantization and bfloat16 compute. Attached LoRA adapter targeting attention projections (q_proj, k_proj, v_proj, o_proj) with rank 8, alpha 16, and verified parameter counts.
* `configs/grpo_config.py`: Centralized configuration for hyperparameters, LoRA settings, and evaluation thresholds.

### 4. Training and Evaluation Scripts (`train_grpo.py`, `evaluate.py`)
* `train_grpo.py`: Implemented matched-baseline GRPO training script using TRL GRPOTrainer with beta=0.0 (no KL penalty), G=2 candidates, temperature=0.8, and AdamW optimizer. Saves trained adapter checkpoints to dedicated arm directories.
* `evaluate.py`: Implemented greedy decoding evaluation (temperature=0.0, 1024 tokens) computing correctness rates, fast 1.0x rates, fast 1.1x rates, and median speedups across training and holdout tasks.

### 5. Colab Pro Execution and Testing (`notebooks/`, `tests/`)
* `notebooks/colab_rl_pipeline.ipynb`: Interactive Jupyter notebook ready to pull and run on Google Colab Pro equipped with an A100 GPU. Includes automated dependency installations, inline compilation tests, teacher kernel benchmarks, GRPO training, and side-by-side metric comparison tables.
* `tests/test_pipeline.py`: Comprehensive test suite verifying task lookups, example stores, prompt formatting, static code filters, and reward taxonomy behavior.

## What Still Needs to Be Worked On

The following items represent the remaining cross-functional dependencies and production rollout steps:

1. Isolated Worker Multi-GPU Architecture (with Pramod Goyal):
   * Transition from single-process Colab execution to dual-GPU worker isolation on the production cluster (GPU 0 for generation/training, GPU 1 for compilation and timing).
   * Integrate NVIDIA compute-sanitizer into the worker compilation pipeline to detect race conditions and illegal memory accesses prior to timing.

2. Full 30B MoE SFT Ingestion (with Mohamed Ziyadh Mohamed):
   * Swap the development fallback model with the final SFT checkpoint of Qwen3-Coder-30B-A3B-Instruct once Mohamed finishes the supervised fine-tuning stage.
   * Expand the Channel RMSNorm teacher store by curating a second verified teacher implementation.

3. Cross-GPU Timing Validation Gate:
   * Active validation on multi-GPU nodes requiring kernels reporting speedup above 3.0x to be re-timed on the secondary GPU with less than 10% discrepancy.

4. Extended Evaluation Benchmarking:
   * Execute evaluation sweeps across prospective holdout workloads (Softmax and Cumulative Sum) once training loss reaches convergence.

## Verification Checklist
* [x] Python 3.10 syntax and type hint compatibility verified.
* [x] Unit tests passing with zero errors (python tests/test_pipeline.py).
* [x] No em/en dashes present across code, docstrings, or documentation.
* [x] Ready for deployment on Google Colab Pro.
