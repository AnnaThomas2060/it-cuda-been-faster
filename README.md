# It CUDA been faster: Automating GPU Optimizations using RL

Graduate research project by Anna Thomas, Mohamed Ziyadh Mohamed, and Pramod Goyal.

## Motivation
Most inference code for LLMs uses significantly less compute than the theoretical limit of modern GPU hardware. Writing high-performance CUDA kernels requires specialized expertise and is hardware specific, making porting across architectures labor intensive. This project trains small language models (SLMs) using LoRA, SinkGD, multi-turn reinforcement learning with thought masking, and contrastive RL to automate CUDA kernel optimization and help maximize hardware efficiency on NVIDIA GPUs.

## Team Roles

| Team Member | Module Path | Primary Responsibilities |
| :--- | :--- | :--- |
| Anna Thomas | `src/pipeline/`, `src/rewards/`, `src/trainers/`, `src/contracts/` | RL pipeline orchestration, reward design, anti-hacking rules, and model fine-tuning (LoRA, SinkGD, ReST, Dr. GRPO, Contrastive RL). |
| Mohamed Ziyadh Mohamed | `src/baselines/`, `src/analytics/` | Hand-crafted CUDA reference baselines, reward tracking, Roofline utilization analysis, and fake-gain detection. |
| Pramod Goyal | `src/verifier/` | Sandboxed GPU compilation harness (nvcc), execution correctness verification, and CUPTI / CUDA Events latency profiler. |

## Repository Structure

```text
it-cuda-been-faster/
├── .github/workflows/ci.yml       # GitHub Actions CI workflow
├── configs/                       # Hyperparameter configurations for training runs
│   ├── sft_lora.yaml              # Stage 1 LoRA SFT hyperparameters
│   ├── rest.yaml                  # Branch A ReST hyperparameters
│   ├── multiturn_grpo.yaml        # Branch B Multi-Turn Dr. GRPO hyperparameters
│   └── contrastive_rl.yaml        # Branch C Contrastive-RL hyperparameters
├── data/                          # Dataset definitions, caches, and manifests
├── docs/                          # Architectural specs and IPC contracts
│   ├── ARCHITECTURE.md            # Detailed multi-turn pipeline and reward design
│   └── CONTRACTS.md               # Inter-vertical Pydantic data schemas
├── scripts/                       # Training, evaluation, and benchmark runners
├── src/                           # Core source packages
│   ├── contracts/                 # Typed IPC schemas and offline mock verifier
│   ├── pipeline/                  # Multi-turn rollout manager and env dispatcher
│   ├── rewards/                   # Anti-hacking guards, speedup metrics, and archive
│   ├── trainers/                  # LoRA, SinkGD, SFT, ReST, GRPO, Contrastive trainers
│   ├── verifier/                  # Sandboxed GPU compilation and profiler (Pramod)
│   ├── baselines/                 # Reference CUDA kernel baselines (Ziyadh)
│   └── analytics/                 # Telemetry analysis and fake-gain auditing (Ziyadh)
└── tests/                         # Test suites and import validation
```

## Setup and Installation

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Linting and Testing
```bash
# Run code linter
ruff check .

# Run test suite
pytest
```

## References
- Li, X., Sun, X., Wang, A., Li, J., & Shum, C. (2026). CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning. ICLR.
- Baronio, C., Marsella, P., Pan, B., & Alberti, S. (2025). Kevin-32B: Multi-Turn RL for Writing CUDA Kernels. Cognition Blog.
- Ouyang, S., et al. (2025). KernelBench: Can LLMs Write Efficient GPU Kernels? ICML.
- Terminal-Bench: Evaluating Foundation Models on Realistic Software Engineering Tasks.
