# Architecture Specification

This document details the reinforcement learning and training architecture designed for Anna Thomas's workstream.

## System Overview

The system optimizes GPU CUDA kernels using multi-turn environmental feedback and contrastive reinforcement learning. The core pipeline is divided into four primary modules:

```text
[ Task Prompt ] 
       │
       ▼
┌──────────────────────────────────────┐
│ Module 1: Multi-Turn Rollout Manager │ ◄── Thought Masking
│ (vLLM Engine, Serial Refinements)    │
└──────────────────┬───────────────────┘
                   │ Generated Kernel
                   ▼
┌──────────────────────────────────────┐
│ Module 2: Subprocess Env Dispatcher  │ ◄── Offload vLLM (sleep level=1)
│ (Sandboxed Compilation & Profiling)  │
└──────────────────┬───────────────────┘
                   │ Verifier Feedback
                   ▼
┌──────────────────────────────────────┐
│ Module 3: Reward Engine & Archive    │ ◄── 6-Layer Anti-Hacking Guard
│ (Gated Speedup + Discounted MDP)     │
└──────────────────┬───────────────────┘
                   │ Token Rewards / Pairs
                   ▼
┌──────────────────────────────────────┐
│ Module 4: Policy Trainers            │ ◄── LoRA (r=64, a=128) + SinkGD
│ (SFT / ReST / Dr. GRPO / Contrastive)│
└──────────────────────────────────────┘
```

## Module 1: Multi-Turn Trajectory Rollout Manager (`src/pipeline/rollout_manager.py`)

- **Turn Budget:** Executes between 4 and 8 serial refinement turns per task trajectory.
- **Thought Masking:** In Kevin-32B multi-turn RL, reasoning traces `<think>...</think>` from previous turns are stripped out before prompting subsequent turns. Only concise summaries and the code blocks are retained. This prevents quadratic context growth while maintaining critical debugging context.
- **State Representation:** Each turn state combines the task specification, current best code, compiler feedback, and profiler execution metrics.

## Module 2: Subprocess Environment Dispatcher (`src/pipeline/env_dispatcher.py`)

- **Subprocess Isolation:** Dispatches compilation and benchmark execution to isolated worker subprocesses to prevent CUDA runtime crashes from killing the orchestrator.
- **Memory Management:** Invokes `vLLM` sleep mode (`sleep(level=1)`) during compilation and hardware profiling to release GPU memory for kernel benchmark runs.
- **Dispatcher Protocol:** Dispatches `KernelEvalRequest` objects and ingests `KernelEvalResponse` objects conforming to `src.contracts`.

## Module 3: Reward Engine and Contrastive Archive (`src/rewards/reward_engine.py`)

### Six-Layer Anti-Reward-Hacking Guard
1. **Compilation Check:** Code must pass `nvcc` compilation without fatal errors.
2. **Correctness Verification:** Numerical outputs must match PyTorch reference tensors within strict tolerances (rtol=1e-3, atol=1e-3).
3. **AST Static Analysis:** Inspects generated C++/CUDA code to reject trivial stubs, empty bodies, and hardcoded return values.
4. **Memory Guard:** Flags and rejects out-of-bounds reads/writes and memory corruption attempts.
5. **Timer Integrity:** Validates timing measurements with CUDA Events, rejecting suspicious microsecond anomalies.
6. **Fake-Gain Filter:** Compares bandwidth against theoretical device limits to eliminate artificial timing artifacts.

### Gated Speedup Reward Function
For correct kernels, the step reward combines baseline correctness, damped speedup, and bandwidth utilization:

$$s_t = 0.30 + \phi\left(\frac{T_{\text{baseline}}}{T_{\text{cuda}}}\right) + 0.10 \cdot \Delta\text{BW}$$

where $\phi$ is a log-damped speedup scaling function, and $\Delta\text{BW}$ measures memory bandwidth improvement over the baseline. Failed compilations or incorrect outputs receive a penalized non-positive score.

### Discounted MDP Return
Returns are discounted across trajectory turns with a discount factor $\gamma = 0.4$:

$$R_t = \sum_{k=t}^{T} 0.4^{k-t} s_k$$

### Contrastive Kernel Archive
Maintains winning and losing kernel pairs across rollouts for the contrastive learning loss, enabling pairwise preference optimization over hardware performance deltas.

## Module 4: Policy Trainers (`src/trainers/`)

- **LoRA Configuration:** Rank $r=64$, scaling $\alpha=128$, targeting all projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **SinkGD Optimizer:** Uses Sinkhorn-based gradient descent (SinkGD) to reduce pre-training memory footprints and accelerate convergence alongside standard AdamW.
- **Training Branches:**
  - **Stage 1 (SFT):** Supervised fine-tuning on verified kernel demonstration pairs.
  - **Branch A (ReST):** Reinforced Self-Training with progressive speedup thresholds (1.00x, 1.10x, 1.25x).
  - **Branch B (Multi-Turn Dr. GRPO):** Group Relative Policy Optimization over multi-turn rollouts with thought masking.
  - **Branch C (Contrastive-RL):** Hybrid policy gradient with pairwise ranking loss over contrastive archive samples.
