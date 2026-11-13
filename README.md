# It CUDA Been Faster: Automating GPU Optimizations Using RL

Graduate research project by Anna Thomas, Mohamed Ziyadh Mohamed, and Pramod Goyal.

Reinforcement learning training, a two-stage reward gate, and multi-turn evaluation for training coding language models to generate high-performance custom CUDA kernels.

## Motivation

Most inference code for large language models uses significantly less compute than the theoretical limit of modern GPU hardware. Writing high-performance CUDA kernels requires specialized expertise and is hardware specific, which makes porting across architectures labor intensive. This project trains small language models with LoRA and reinforcement learning to automate CUDA kernel optimization and increase hardware efficiency on NVIDIA GPUs.

## Team Roles

| Team Member | Module Path | Primary Responsibilities |
| :--- | :--- | :--- |
| Anna Thomas | `engine/`, `rewards/`, `models/`, `configs/`, `src/pipeline/`, `src/rewards/`, `src/trainers/`, `train_grpo.py`, `evaluate.py` | RL orchestration, multi-turn rollouts, reward design, anti-hacking guards, Dr. GRPO and contrastive training, evaluation, and cluster tooling. |
| Mohamed Ziyadh Mohamed | `src/baselines/`, `src/analytics/` | Hand-written CUDA reference baselines, telemetry analysis, roofline utilization, and fake-gain auditing. |
| Pramod Goyal | `src/verifier/` | Sandboxed GPU compilation, correctness verification, and CUDA Events profiling. |

## Experimental Arms

The project asks whether conditioning a model on earlier implementations and their measured performance helps it produce faster kernels than a task specification alone.

1. **Arm A (Contrastive):** the prompt includes the task specification and stored teachers with measured speedups.
2. **Arm B (Reference only):** the prompt includes only the task specification and interface contract.

Both arms use the same base weights, LoRA configuration, seed, task schedule, turn budget, and benchmark gates. Holdout tasks now have stored teachers, so the Arm A and Arm B comparison remains valid on both training and holdout splits.

## Pipeline Overview

```text
task prompt
    |
    v
RolloutManager  ->  RewardEngine  ->  dispatcher  ->  compiler, verifier, benchmark
    ^                                                        |
    |                                                        v
    +---------------- trajectory state and rewards <---------+
                             |
                             v
              MultiTurnGRPOTrainer or ContrastiveRLTrainer
```

The rollout manager generates one completion per active turn, extracts the code block, and requests an evaluation. The reward engine applies the correctness and stability gates, records telemetry, and updates the contrastive archive. The trainer converts turn rewards into discounted returns, normalizes advantages inside each task and turn group, and performs Dr. GRPO updates. Contrastive mode adds a pairwise preference loss over archived winner and loser kernels.

## Repository Layout

```text
it-cuda-been-faster/
|-- configs/
|   |-- grpo_config.py            # ExperimentConfig dataclass
|   |-- multiturn_grpo.yaml       # Multi-turn Dr. GRPO
|   |-- contrastive_rl.yaml       # Contrastive RL
|   |-- sft_lora.yaml             # Stage 1 SFT
|   `-- rest.yaml                 # ReST branch
|-- data/
|   |-- tasks.py                  # Task registry, references, shapes, tolerances
|   |-- example_store.py          # Teacher kernels and measurement overrides
|   `-- README.md
|-- docs/
|   |-- ARCHITECTURE.md
|   `-- CONTRACTS.md
|-- engine/
|   |-- compiler.py               # Static checks and load_inline compilation
|   |-- verifier.py               # Multi-shape, multi-seed numerical verification
|   |-- benchmark.py              # Blocked CUDA Events timing and anti-cache checks
|   |-- sanitizer.py              # Optional compute-sanitizer runner
|   |-- worker.py                 # Verifier worker process
|   |-- prompt_builder.py         # Reference-only and contrastive chat prompts
|   `-- utils.py                  # Seeding, JSONL, metadata, and timing helpers
|-- models/
|   `-- loader.py                 # 4-bit base model, SFT handoff, trainable RL adapter
|-- notebooks/
|   `-- colab_rl_pipeline.ipynb
|-- rewards/
|   `-- grpo_reward.py            # TRL-compatible reward callback and taxonomy
|-- scripts/
|   |-- check_env.py
|   |-- verify_teachers.py
|   |-- compare_arms.py
|   `-- slurm/
|-- src/
|   |-- contracts/                # Pydantic request, response, and trajectory models
|   |-- pipeline/                 # Rollout manager, trajectory state, dispatchers
|   |-- rewards/                  # Reward engine, anti-hack checks, archive
|   |-- trainers/                 # Loss functions, Dr. GRPO, contrastive trainer
|   |-- verifier/                 # Teammate verifier scaffolding
|   |-- baselines/                # Teammate baseline kernels
|   `-- analytics/                # Teammate telemetry analysis
|-- tests/
|   |-- test_pipeline.py
|   |-- test_multiturn_pipeline.py
|   `-- test_smoke_imports.py
|-- train_grpo.py
|-- evaluate.py
|-- requirements.txt
`-- pyproject.toml
```

## Reward and Timing Gates

1. **Static guard.** The source must declare a `__global__` kernel and a `torch::Tensor run_cuda(torch::Tensor x)` entrypoint. A whitelist rejects PyTorch operator shortcuts, tensor member calls, and manual CUDA allocation.
2. **Compilation.** Candidate code is compiled with `torch.utils.cpp_extension.load_inline` using `-O3` and `--use_fast_math`.
3. **Correctness.** Every configured shape and seed must match the reference. Column max requires exact equality. Other tasks use `rtol=1e-4` and `atol=1e-4`. Input mutation, output aliasing, wrong shape, wrong dtype, and non-finite values are rejected.
4. **Timing.** Ten warmup iterations run on multiple input buffers, followed by seven chronological blocks of ten paired runs. Block ratio variance above 0.005 rejects the candidate. Measurements below 0.005 ms are rejected.
5. **Fake gain and cache checks.** Achieved bandwidth is compared with the device peak, implausible speedups are rejected, and a post-timing correctness check runs on fresh inputs.
6. **Optional sanitizer.** `compute-sanitizer` memcheck can be enabled for additional memory safety checks.

For a compiled, correct, and stable kernel:

```text
reward = 0.30 + log1p(max(speedup - 1.0, 0.0)) + 0.10 * bandwidth_delta
```

The reward is clipped to 2.0 by default. Failed candidates receive zero.

## Setup

Use Python 3.10 or newer with a CUDA capable PyTorch installation.

```bash
pip install -r requirements.txt
pip install -e .
```

Set the allocator flag used by the training and worker processes:

```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

Run the preflight check before a long job:

```bash
python scripts/check_env.py --config configs/multiturn_grpo.yaml
```

On a CPU only machine use `--allow_cpu` to inspect packages, configs, and imports without failing on CUDA.

## Teacher Verification

Stored teacher kernels are useful only when their speedups were measured on the target GPU. Measure them once and write the overrides:

```bash
python scripts/verify_teachers.py --config configs/multiturn_grpo.yaml --device cuda:0
```

This writes `data/teacher_measurements.json`. The prompt builder reads that file and marks measured entries as verified. Unverified placeholders remain visible in the prompt with an `(unverified)` note.

## Training

Train both matched arms with the multi-turn config:

```bash
python train_grpo.py \
  --arm both \
  --mode multi_turn \
  --config configs/multiturn_grpo.yaml \
  --dispatcher subprocess
```

Useful options:

- `--mode single_turn|multi_turn|contrastive`
- `--use_dev_model` or `--use_tiny_model` for smaller local tests
- `--dry_run` for one step
- `--dispatcher auto|subprocess|in_process|mock`
- `--device cuda:0` and `--verifier_device cuda:1`
- `--resume_from_checkpoint <checkpoint-dir>`
- `--sft_adapter_path <adapter-or-merged-model>`
- `--output_dir <dir>`

On a single GPU set both device fields to the same value and prefer `in_process` or a single subprocess worker. On a two GPU node the YAML defaults use `cuda:0` for training and `cuda:1` for verification.

Checkpoints contain the adapter, tokenizer, optimizer, scheduler, and RNG state. Each arm writes its own archive derived from `archive_path`, so Arm A and Arm B do not share anchors and the files survive a preemption.

## SFT Handoff

The RL stage accepts either a PEFT adapter directory or a merged full model checkpoint from the SFT owner.

```bash
python train_grpo.py \
  --arm arm_a \
  --config configs/multiturn_grpo.yaml \
  --sft_adapter_path outputs/sft/final_adapter
```

An adapter is frozen as the `sft` adapter and a fresh `rl` adapter is attached for training. A merged checkpoint is treated as the base model. The same path can be supplied through `sft_adapter_path` in a YAML file.

## Evaluation

Evaluate a trained adapter on the configured training and holdout tasks:

```bash
python evaluate.py \
  --config configs/multiturn_grpo.yaml \
  --adapter_path outputs/arm_a_contrastive/final_adapter \
  --prompt_mode contrastive \
  --dispatcher subprocess \
  --output_file outputs/eval_results/arm_a.json
```

Reports include correctness rate, attempt correctness rate, the fraction of tasks at or above 1.0x and 1.1x, median and mean speedup, interquartile speedups, a bootstrap confidence interval for the median, and per-split summaries. Add `--base_model_control` to evaluate the base model without an adapter.

Compare the two arms with paired per-task statistics:

```bash
python scripts/compare_arms.py \
  --arm_a outputs/eval_results/arm_a.json \
  --arm_b outputs/eval_results/arm_b.json \
  --split all \
  --output outputs/eval_results/comparison.json
```

The comparison reports wins, ties, losses, paired median difference, and a bootstrap interval. Use `--split holdout` for the generalization claim.

## Cluster and Container Runs

SLURM templates are in `scripts/slurm/`. The standard template runs preflight, both arms, both evaluations, and the paired comparison. The Apptainer template runs the same commands inside a container with `--nv`.

```bash
mkdir -p logs outputs/eval_results
sbatch scripts/slurm/train_grpo.slurm
```

Adjust the `#SBATCH` partition, account, GPU type, walltime, and module names for your site. See `scripts/slurm/README.md` and `scripts/README.md` for details.

## Google Colab

`notebooks/colab_rl_pipeline.ipynb` verifies the toolchain, measures teacher kernels, inspects prompts, accepts an optional SFT handoff, runs a dry run, trains both arms, evaluates them, and prints the comparison. It uses a single GPU by default.

## Tests

```bash
python -m pytest tests -q
python -m ruff check .
```

The suite covers task registration, teacher storage, prompt construction, static checks, reward shaping, the archive, dispatcher contracts, log probability gathering, GRPO and DPO losses, rollout sample construction, evaluation statistics, notebook structure, and repository text hygiene. It does not require CUDA.

## Known Limitations

- The custom trainer targets one training GPU. Multi-GPU DDP is not implemented.
- Generation uses the Hugging Face `generate` API. A paged vLLM backend is not included.
- Placeholder teacher speedups must be replaced with measurements on the target GPU before reporting results.
- `compute-sanitizer` availability and SLURM module names vary by cluster.

## References

- Li, X., Sun, X., Wang, A., Li, J., & Shum, C. (2026). CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning. ICLR.
- Baronio, C., Marsella, P., Pan, B., & Alberti, S. (2025). Kevin-32B: Multi-Turn RL for Writing CUDA Kernels. Cognition Blog.
- Ouyang, S., et al. (2025). KernelBench: Can LLMs Write Efficient GPU Kernels? ICML.
- Terminal-Bench: Evaluating Foundation Models on Realistic Software Engineering Tasks.
