# It CUDA Been Faster: Automating GPU Optimizations Using RL

Graduate research project by Anna Thomas, Mohamed Ziyadh Mohamed, and Pramod Goyal.

Reinforcement learning training, two-stage reward design, and greedy evaluation pipeline for training coding large language models to generate high-performance custom CUDA kernels.

## Motivation

Most inference code for LLMs uses significantly less compute than the theoretical limit of modern GPU hardware. Writing high-performance CUDA kernels requires specialized expertise and is hardware specific, making porting across architectures labor intensive. This project trains small language models (SLMs) using LoRA, SinkGD, multi-turn reinforcement learning with thought masking, and contrastive RL to automate CUDA kernel optimization and help maximize hardware efficiency on NVIDIA GPUs.

## Team Roles

| Team Member | Module Path | Primary Responsibilities |
| :--- | :--- | :--- |
| Anna Thomas | `engine/`, `rewards/`, `models/`, `configs/`, `train_grpo.py`, `evaluate.py` | RL pipeline orchestration, reward design, anti-hacking rules, contrastive GRPO training, and evaluation. |
| Mohamed Ziyadh Mohamed | `src/baselines/`, `src/analytics/` | Hand-crafted CUDA reference baselines, reward tracking, Roofline utilization analysis, and fake-gain detection. |
| Pramod Goyal | `src/verifier/` | Sandboxed GPU compilation harness (nvcc), execution correctness verification, and CUPTI / CUDA Events latency profiler. |

## Overview and Experimental Arms

This repository implements the RL training and reward pipeline for Anna Thomas's component of the MSAI635 project. The research investigates whether conditioning the model on previous implementations and their measured performance scores (contrastive RL) helps it generate faster CUDA kernels compared to task specifications alone.

1. **Arm A (Scored Context / Contrastive GRPO):** The prompt contains the PyTorch task specification along with historical same-task implementations and their measured relative speedup scores.
2. **Arm B (Reference Only GRPO):** The prompt contains only the PyTorch task specification and interface contract without historical implementations.

Both experimental conditions utilize identical base weights, LoRA attention adapters, optimizer hyperparameter states, random seeds (635), task schedules, and latency gates.

## Architecture and Modules

```text
it-cuda-been-faster/
├── .github/workflows/ci.yml   # GitHub Actions CI workflow
├── configs/                   # Hyperparameters, LoRA targets, and benchmark gates
│   ├── grpo_config.py         # Central Python dataclass configuration
│   ├── sft_lora.yaml          # Stage 1 LoRA SFT hyperparameters
│   ├── rest.yaml              # Branch A ReST hyperparameters
│   ├── multiturn_grpo.yaml    # Branch B Multi-Turn Dr. GRPO hyperparameters
│   └── contrastive_rl.yaml    # Branch C Contrastive-RL hyperparameters
├── data/                      # Task registry, reference implementations, and teacher store
│   ├── tasks.py               # KernelBench task registry, PyTorch refs, and shapes
│   ├── example_store.py       # Curated teacher kernels with baseline speedup metrics
│   └── README.md              # Data module overview
├── docs/                      # Architectural specifications and IPC contracts
│   ├── ARCHITECTURE.md        # Detailed multi-turn pipeline and reward design
│   └── CONTRACTS.md           # Inter-vertical Pydantic data schemas
├── engine/                    # Execution, compilation, verification, and timing engine
│   ├── prompt_builder.py      # Reference-only and contrastive prompt generators
│   ├── compiler.py            # Static contract checker and load_inline compiler
│   ├── verifier.py            # Multi-shape, multi-seed numerical equivalence suite
│   └── benchmark.py           # 7-block chronological latency timer with variance filter
├── rewards/                   # Reward functions and taxonomy tracking
│   └── grpo_reward.py         # TRL-compatible reward callback and failure taxonomy logger
├── models/                    # Model loading and LoRA adapters
│   └── loader.py              # 4-bit NF4 base model loader with rank-8 LoRA adapter
├── notebooks/                 # Interactive Jupyter notebooks
│   └── colab_rl_pipeline.ipynb # Ready-to-run notebook for Google Colab Pro
├── scripts/                   # Helper scripts and entrypoints
│   └── README.md              # Scripts module overview
├── src/                       # Team shared scaffolding and contracts
│   ├── contracts/             # Typed IPC schemas and offline mock verifier
│   ├── pipeline/              # Multi-turn rollout manager and env dispatcher
│   ├── rewards/               # Anti-hacking guards, speedup metrics, and archive
│   ├── trainers/              # LoRA, SinkGD, SFT, ReST, GRPO, Contrastive trainers
│   ├── verifier/              # Sandboxed GPU compilation and profiler (Pramod)
│   ├── baselines/             # Reference CUDA kernel baselines (Ziyadh)
│   └── analytics/             # Telemetry analysis and fake-gain auditing (Ziyadh)
├── tests/                     # Unit test suites
│   ├── test_pipeline.py       # Unit test suite verifying RL pipeline components
│   └── test_smoke_imports.py  # Smoke tests for scaffolding and contracts
├── train_grpo.py              # Main training script for matched baseline runs
├── evaluate.py                # Greedy evaluation script computing coverage and speedups
├── requirements.txt           # Pinned software stack
└── pyproject.toml             # Package configuration
```

## Two-Stage Reward Gate

1. **Stage 1 (Correctness Gatekeeper):**
   - **Static inspection:** Enforces `__global__` declaration, `torch::Tensor run_cuda` host signature, and forbids PyTorch operator shortcuts.
   - **Compilation:** Compiles cleanly via `torch.utils.cpp_extension.load_inline` with `-O3` and `--use_fast_math`.
   - **Numerical equivalence:** Verifies outputs across multiple tensor shapes and 4 seeded runs within 1e-4 tolerance (column max requires exact bitwise equality). Ensures input tensors are not mutated in place.

2. **Stage 2 (Latency Benchmarking and Noise Gate):**
   - Executes 10 warmup iterations on the target GPU.
   - Measures 7 chronological blocks of 10 paired timing runs with CUDA events.
   - Calculates population variance across block ratios. If variance exceeds 0.005, the candidate is rejected as unstable (reward = 0.0).
   - Floors stable speedups to two decimal places.

## Setup and Installation

Install dependencies in a Python 3.10+ environment:

```bash
pip install -r requirements.txt
pip install -e .
```

Ensure the PyTorch CUDA allocator is configured to reduce memory fragmentation:

```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

## Running on Google Colab Pro

Open `notebooks/colab_rl_pipeline.ipynb` in Google Colab with an A100 GPU runtime. Step through the cells to verify teacher kernels, test prompt generation, load the quantized model, and train both experimental arms.

## Running from Command Line

Execute matched GRPO training runs:

```bash
# Train Arm A (Contrastive Scored Context)
python train_grpo.py --arm arm_a --use_dev_model

# Train Arm B (Reference Only)
python train_grpo.py --arm arm_b --use_dev_model
```

Run greedy evaluation:

```bash
python evaluate.py --adapter_path ./outputs/arm_a_contrastive/final_adapter --prompt_mode contrastive
python evaluate.py --adapter_path ./outputs/arm_b_reference_only/final_adapter --prompt_mode reference_only
```

Run unit tests:

```bash
pytest tests/
python tests/test_pipeline.py
```

## References

- Li, X., Sun, X., Wang, A., Li, J., & Shum, C. (2026). CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning. ICLR.
- Baronio, C., Marsella, P., Pan, B., & Alberti, S. (2025). Kevin-32B: Multi-Turn RL for Writing CUDA Kernels. Cognition Blog.
- Ouyang, S., et al. (2025). KernelBench: Can LLMs Write Efficient GPU Kernels? ICML.
- Terminal-Bench: Evaluating Foundation Models on Realistic Software Engineering Tasks.
