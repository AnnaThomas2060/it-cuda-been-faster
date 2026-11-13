# Architecture Specification

This document describes the reinforcement learning pipeline for Anna Thomas's workstream. The implementation in this repository is the source of truth. Some concepts from the original research plan, such as vLLM sleep mode and SinkGD, are not required by the current training path and are marked as optional or external.

## System Overview

The pipeline turns a task prompt into a multi-turn refinement trajectory, evaluates each kernel on a separate verifier device, and trains the policy with Dr. GRPO and an optional pairwise contrastive loss.

```text
[ Task prompt and teacher context ]
                 |
                 v
[ RolloutManager ]  generates one completion per active trajectory turn
                 |
                 v
[ RewardEngine ]  sends KernelEvalRequest to the dispatcher
                 |
                 v
[ Compiler, verifier, benchmark, sanitizer ]  runs in a worker process
                 |
                 v
[ Trajectory state ]  feedback, discounted returns, group advantages
                 |
                 v
[ MultiTurnGRPOTrainer ]  Dr. GRPO loss and optional archive pairwise loss
```

## Entry Points

- `train_grpo.py`: training entrypoint. `run_training` supports `single_turn`, `multi_turn`, and `contrastive` modes. `run_grpo_training` remains as a compatibility wrapper for the original single-turn path.
- `evaluate.py`: single-turn and multi-turn evaluation with train and holdout summaries, bootstrap confidence intervals, and an optional base model control.
- `scripts/check_env.py`: preflight checks for Python, CUDA, compiler tools, configs, and imports.
- `scripts/verify_teachers.py`: measures the stored teacher kernels and writes `data/teacher_measurements.json`.
- `scripts/compare_arms.py`: paired per-task comparison of Arm A and Arm B evaluation summaries.

## Module 1: Multi-Turn Rollout Manager

Implementation: `src/pipeline/rollout_manager.py` and `src/pipeline/state.py`.

- A trajectory is created per task and per group member. `group_size` defaults to four and is configured per experiment.
- Each turn builds a chat prompt from the task description, the optional teacher context, the current best code, compiler or verifier feedback, and a short history of earlier turns.
- Reference-only prompts contain the task and interface contract. Contrastive prompts additionally contain stored teachers and their measured speedups.
- Thought text inside `<think>...</think>` is extracted into `TurnStepSample.thought` and stripped from the feedback that is fed into later prompts. This bounds context growth across turns.
- Generation uses the Hugging Face `generate` API with left padding, prompt truncation, and configurable sampling parameters. Batched generation is controlled by `rollout_batch_size`.
- Completion token ids and behavior policy log probabilities are stored per step. `src/trainers/losses.py` gathers completion log probabilities from a fixed number of final logits to keep memory bounded.
- A trajectory stops early when its best verified speedup reaches `early_stop_speedup`.
- Step rewards are discounted with `gamma`. Advantages are normalized per `(task_id, turn_index)` group, so each turn is compared against rollouts of the same task at the same depth.
- `build_training_samples` emits padded training samples with input ids, prompt and completion lengths, old log probabilities, reward, return, and advantage.

## Module 2: Environment Dispatcher and Verifier Worker

Implementation: `src/pipeline/env_dispatcher.py` and `engine/worker.py`.

- `SubprocessDispatcher` starts one or more spawn based workers on `verifier_device`. Each worker owns a pipe and evaluates one request at a time. A worker that times out or crashes is terminated and restarted.
- `InProcessDispatcher` calls the same evaluation function in the current process. It is useful on a single GPU, in a notebook, or when debugging.
- `MockDispatcher` uses `MockVerifier` and never touches CUDA. It powers CPU only tests and dry runs.
- The worker sets CUDA device selection, optional per process memory limits, CPU resource limits where available, and the PyTorch extensions directory before compiling.
- The main process keeps training memory separate from compilation and benchmarking. On a two GPU node set `device` to one GPU and `verifier_device` to the other. On a single GPU set both to the same device.
- vLLM sleep mode is not part of this implementation. Generation uses the training model directly. If a teammate provides a vLLM rollout backend, it can be integrated behind the same rollout interface.

## Module 3: Reward Engine, Guards, and Archive

Implementation: `src/rewards/reward_engine.py`, `src/rewards/anti_hack.py`, `engine/compiler.py`, `engine/verifier.py`, and `engine/benchmark.py`.

### Compilation and static guard

- The source must contain a `__global__` kernel and a `torch::Tensor run_cuda(torch::Tensor x)` entrypoint.
- A whitelist allows basic tensor allocation and construction symbols and rejects framework shortcuts such as `torch::relu`, `at::native`, tensor member calls like `.softmax()` or `.sum()`, and manual `cudaMalloc` or `cudaMemcpy`.
- Comments and string literals are removed before static inspection to avoid false positives.
- The compiler wraps the candidate in `load_inline` with `-O3` and `--use_fast_math` and returns a structured `CompileResult` with a source hash and build time.

### Numerical verification

- `verify_kernel` checks every configured shape and seed.
- Column max uses exact equality. Other tasks use the per-task relative and absolute tolerances.
- The verifier rejects input mutation, output aliasing, wrong shape, wrong dtype, non-finite values, and non-CUDA outputs.
- Seed handling uses local generators and does not mutate global RNG state.

### Timing gate

- The benchmark warms up on multiple input buffers and then measures seven chronological blocks of paired baseline and candidate runs with CUDA events.
- The candidate is rejected when block ratio variance exceeds `variance_threshold`, when the median candidate time is below `min_measurement_ms`, when achieved bandwidth exceeds the device peak by more than `fake_gain_ratio`, or when speedup exceeds `max_plausible_speedup`.
- A post-timing correctness check runs again on fresh inputs to catch caching or input specific tricks.
- Stable speedups are floored to two decimals.

### Optional memory sanitizer

`engine/sanitizer.py` can run `compute-sanitizer` memcheck when `enable_sanitizer` is true. The sanitizer is optional because it is slow and not available on every system.

### Reward shaping

For a compiled, correct, and stable kernel:

```text
reward = correctness_bonus
       + log1p(max(speedup - 1.0, 0.0))
       + bandwidth_weight * clip(bandwidth_delta, -1.0, 1.0)
```

The result is clipped to `reward_clip`. Failed or rejected candidates receive zero. `reward_mode: speedup` uses the clipped raw speedup instead.

### Discounted returns

Step rewards are discounted across turns:

```text
R_t = s_t + gamma * s_{t+1} + gamma^2 * s_{t+2} + ...
```

The default discount is 0.4. `group_normalize` converts returns into advantages inside each task and turn group.

### Contrastive archive

`ContrastiveArchive` stores correct and incorrect kernels per task with code hashes, speedups, statuses, and prompt modes. Duplicate kernels are ignored. The archive keeps a bounded mix of high and low performers and can sample winner and loser pairs with a minimum speedup margin. Pairs feed the optional DPO style pairwise loss.

## Module 4: Policy Trainers

Implementation: `src/trainers/multiturn_grpo.py` and `src/trainers/losses.py`.

- `MultiTurnGRPOTrainer` implements token level Dr. GRPO with a clipped importance ratio. The denominator is `dr_grpo_l_norm` rather than the local sequence length, which follows the Dr. GRPO correction for length bias.
- `ContrastiveRLTrainer` adds a pairwise preference loss over archive samples. The reference policy is either the frozen adapter loaded from the SFT handoff or the base model reached through `disable_adapter()`.
- The optimizer is AdamW over trainable LoRA parameters. The scheduler is cosine with warmup.
- Mixed precision uses bf16 when available and falls back to fp16 with a gradient scaler.
- Gradient accumulation, gradient clipping, periodic checkpoints, and checkpoint resume are supported. Checkpoints include optimizer, scheduler, and RNG state.
- Training metrics are appended to `train_metrics.jsonl` in the arm output directory. Run metadata, the config, the taxonomy summary, and the archive summary are written to `run_metadata.json`.

## SFT Handoff

The RL stage accepts either of these from the SFT owner:

- A PEFT adapter directory containing `adapter_config.json`. The loader freezes it as adapter `sft` and attaches a fresh trainable adapter named `rl`.
- A merged full model directory containing `config.json`. The loader uses it as the base model and attaches a fresh trainable adapter.

Set `sft_adapter_path` in the config or pass `--sft_adapter_path` on the command line. Evaluation can load a trained RL adapter directly with `--adapter_path`.

## Training Modes and Arms

- `single_turn`: one completion and one reward per task, using the TRL `GRPOTrainer` path.
- `multi_turn`: serial refinement with Dr. GRPO over turn level rewards.
- `contrastive`: multi-turn training plus the archive pairwise loss.

Arm A uses contrastive prompts. Arm B uses reference-only prompts. Both arms must use the same seed, base model, LoRA configuration, task schedule, turn budget, and benchmark gates.

## Configuration

`configs/grpo_config.py` defines `ExperimentConfig`. YAML files are loaded with `ExperimentConfig.from_yaml`. Unknown keys raise an error, and known aliases such as `gamma_discount`, `bw_bonus_weight`, `clip_eps_low`, and `clip_eps_high` are mapped to their canonical fields.

Primary example files:

- `configs/multiturn_grpo.yaml`: multi-turn Dr. GRPO.
- `configs/contrastive_rl.yaml`: contrastive variant with archive pairs.
- `configs/sft_lora.yaml` and `configs/rest.yaml`: teammate owned stage files.

## Known Limitations

- The custom trainer runs on one training GPU. Multi-GPU DDP is not implemented.
- Generation uses Hugging Face decoding rather than a paged vLLM backend.
- Teacher speedups are placeholders until `scripts/verify_teachers.py` runs on the target GPU.
- `compute-sanitizer` and cluster specific module names must be checked on the target system.
