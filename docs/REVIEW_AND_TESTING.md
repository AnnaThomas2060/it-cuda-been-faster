# Multi-Turn RL Pipeline — Review, Required Changes, and A100 Testing Guide

Branch: `implement/full-multiturn-rl-pipeline`
Spec reference: `docs/ARCHITECTURE.md` + `docs/CONTRACTS.md` (there is **no** `docs/IMPLEMENTATION.md` in the repo).
Audience: the person running the Colab/A100 verification pass.

This document has three parts:

1. **Verification summary** — what was checked and what passed.
2. **Changes and improvements to be done** — ordered by priority, each with file, exact location, rationale, and a concrete fix.
3. **A100 testing instructions** — a step-by-step protocol to exercise every component on a real GPU, plus an OOM-hardening checklist.

---

## Part 1 — Verification Summary

### Environment used for review
- CPU-only sandbox, **no GPU/CUDA**.
- **No network access to PyPI** (`torch`, `transformers`, `trl`, `pydantic`, `numpy`, `yaml` could not be installed). Only `pytest` and `ruff` were available.
- Therefore the full test suite and any real CUDA compile/train/benchmark **could not be executed here**. Those must run on the A100 (Part 3).

### What was checked and passed
| Check | Result |
| :--- | :--- |
| `ruff check .` (static lint over the whole repo) | **Pass — 0 issues** |
| `python -m compileall` over all `.py` (syntax, Py 3.11) | **Pass — all compile** |
| Import probe of every module without third-party deps | 11 modules import cleanly; the rest fail **only** on missing `pydantic`/`torch` (not code bugs) |
| 12 functional tests on dependency-free modules | **12/12 pass** (task registry, teacher store, config, static CUDA guards, code extraction, both prompt modes, `ContrastiveArchive` dedup/pairing/persistence, `<think>` masking) |
| Spec coverage of all 4 architecture modules | All implemented, not stubs |

### What could NOT be verified here (do on A100)
- Real `load_inline` CUDA compilation, numerical verification, the 7-block timing gate.
- GRPO / DPO training steps, checkpoint/resume, generation.
- The repo's own `pytest tests` suite (blocked on missing deps; all 3 test modules fail at import/collection only).

---

## Part 2 — Changes and Improvements To Be Done

Priorities: **P0** = correctness bug, fix before trusting results. **P1** = should fix before the real run. **P2** = cleanup / robustness.

### P0-1 — `TurnStepSample.task_id` and `prompt_mode` are never populated
**File:** `src/pipeline/state.py`, `TurnRecord.to_sample()` (around lines 38–50).

`to_sample()` builds `TurnStepSample(...)` without `task_id` or `prompt_mode`, so every serialized step in `TrajectoryRecord.steps` gets `task_id=""` and `prompt_mode="reference_only"` regardless of the real task/arm. **This silently corrupts offline analysis and makes Arm A vs Arm B indistinguishable at the step level** — directly undermining the core experiment.

**Fix:** thread the values from `Trajectory` (which already knows them) into the sample. Simplest change — pass them through `to_sample`:

```python
# src/pipeline/state.py
def to_sample(self, task_id: str = "", prompt_mode: str = "reference_only") -> TurnStepSample:
    return TurnStepSample(
        task_id=task_id,
        prompt_mode=prompt_mode,
        turn_index=self.turn_index,
        prompt=self.prompt,
        thought=self.thought,
        code=self.code,
        response=self.response,
        reward=self.reward,
        discounted_return=self.discounted_return,
        advantage=self.advantage,
        prompt_ids=self.prompt_ids,
        completion_ids=self.completion_ids,
        old_logprobs=self.old_logprobs,
    )
```

And in `Trajectory.to_record()`:

```python
steps=[step.to_sample(self.task_id, self.prompt_mode) for step in self.steps],
```

### P1-1 — Guard against silent Dr. GRPO normalization fallback
**Files:** `src/trainers/losses.py` (`grpo_loss`, the `denominator = float(l_norm) if l_norm else ...` line) and `configs/grpo_config.py` (`validate()`).

If `dr_grpo_l_norm` is ever `0`, the loss silently reverts to local sequence-length normalization — exactly the length bias Dr. GRPO is meant to remove — with no warning.

**Fix:** add to `ExperimentConfig.validate()`:

```python
if self.loss_type == "dr_grpo" and self.dr_grpo_l_norm <= 0:
    raise ValueError("dr_grpo_l_norm must be > 0 for dr_grpo loss")
```

### P1-2 — Harden completion-logprob alignment against models that ignore `num_logits_to_keep`
**Files:** `src/pipeline/rollout_manager.py` (`_completion_logprobs`) and `src/trainers/multiturn_grpo.py` (`_compute_loss`, `_forward_sequence_logprob`).

The offset is inferred as `logit_offset = output_ids.shape[1] - logits.shape[1]`. This is correct when the model honors `num_logits_to_keep`, but some custom/remote-code models accept the kwarg and still return full logits (no `TypeError` raised, so the fallback branch is not taken). The math still works out in the common case, but `gather_completion_logprobs` will raise `ValueError` if the inferred span runs past `sequence_length`.

**Fix:** after computing `logit_offset`, assert the invariant and fail loudly with a clear message, or clamp. Minimum: wrap the `gather_completion_logprobs` call and re-raise with the tensor shapes and `keep_last` in the message so a shape mismatch is diagnosable on the A100 rather than cryptic.

### P1-3 — Expand `validate()` coverage
**File:** `configs/grpo_config.py`, `validate()`.

Add cheap guards (each is a one-liner) so a bad YAML fails fast instead of mid-run on the GPU:
- `reward_clip > 0`
- `rollout_batch_size >= 1`, `verifier_workers >= 1`
- `pairwise_margin_weight >= 0`, `pairwise_beta >= 0`
- `max_prompt_length >= 1`, `max_teacher_examples >= 0`
- Validate that every id in `training_tasks` / `holdout_tasks` is actually registered in `data.tasks` (currently only cross-membership of `eval_tasks` is checked, not registration). Example:
  ```python
  from data.tasks import list_task_ids
  registered = set(list_task_ids(training_only=False))
  bad = [t for t in self.training_tasks + self.holdout_tasks if t not in registered]
  if bad:
      raise ValueError(f"Unregistered tasks in config: {bad}")
  ```

### P1-4 — Document / reconcile the contracts location mismatch
**Files:** `docs/CONTRACTS.md` §6 vs `src/pipeline/env_dispatcher.py`.

`CONTRACTS.md` states "all models live in `src/contracts/__init__.py`" and that `MockDispatcher`/`InProcessDispatcher` implement the contract there. In reality only `MockVerifier` is in `src/contracts`; `MockDispatcher` and `InProcessDispatcher` live in `env_dispatcher.py`. Either re-export them from `src/contracts/__init__.py` (and add to `__all__`) or correct the doc. No runtime impact, but it misleads the analytics/verifier teammates.

### P2-1 — Remove dead cross-GPU config
**File:** `configs/grpo_config.py` (`cross_gpu_validation`, `cross_gpu_ratio_threshold`, `cross_gpu_tolerance`).
No code reads these. Either implement the cross-GPU re-timing described in `PULL_REQUEST.md` item 3, or delete the fields so readers are not misled.

### P2-2 — Minor cleanups
- `src/rewards/reward_engine.py` `evaluate_code`: remove the no-op `if ... status == "ok": status = "ok"` branch.
- The benchmark status rejection set in `compute_shaped_reward` is redundant with the `not stable` early return — keep one, comment why, or drop the set.
- `docs/CONTRACTS.md` §2 status example list is missing `implausible_speedup`, `measurement_too_small`, `cache_or_incorrect`, and `empty_code` which the code actually emits. Update the example list (statuses are free strings, so not a breaking issue).

---

## Part 2b — OOM Optimization (do before the 30B run)

These target the real memory-spike paths found in the code. They matter most for `Qwen3-Coder-30B-A3B-Instruct`; the dev 7B model mostly fits an A100-40GB but benefits too.

### OOM-1 (biggest win) — Capture generation logprobs during `generate`, skip the extra full forward pass
**File:** `src/pipeline/rollout_manager.py` — `_generate` and `_completion_logprobs`.

**Problem:** after generation, `_completion_logprobs` runs a **second full forward pass over `output_ids`** (prompt + completion, up to `max_prompt_length + max_completion_length` ≈ 4096 + 1536 = 5632 tokens) to recompute the behavior-policy logprobs. This produces a `[batch, keep_last, vocab]` logits tensor (vocab ≈ 150k for Qwen3) and is the single largest rollout-time allocation — on top of whatever generation left resident.

**Fix:** ask `generate` to return scores and compute old-logprobs from them, eliminating the second forward entirely:

```python
# in _generate: request scores
generation_kwargs["return_dict_in_generate"] = True
generation_kwargs["output_scores"] = True
with torch.no_grad():
    gen = self.model.generate(**generation_kwargs)
output_ids = gen.sequences
# gen.scores is a tuple (len = num generated steps) of [batch, vocab] logits
```
Then gather per-token logprobs by indexing each step's score at the chosen token. This replaces the whole `_completion_logprobs` forward pass with cheap indexing over tensors you already paid for during generation. If you keep the separate forward for exactness, at minimum **chunk the batch to size 1** there and `del logits; torch.cuda.empty_cache()` immediately after extracting logprobs.

### OOM-2 — Free the generation KV cache before the logprob forward pass
**File:** `src/pipeline/rollout_manager.py`, between `_generate` and `_completion_logprobs`.
Insert `del output_ids_kv / torch.cuda.empty_cache()` is not enough by itself; the key is **don't hold the generation output and run a second forward at the same time**. If you keep the two-pass design, move the logprob pass into its own function scope and delete the generate outputs you no longer need first.

### OOM-3 — Cast logits to float lazily / in chunks
**File:** `src/trainers/losses.py`, `gather_completion_logprobs`.
`chunk_logits = logits[...].float()` already chunks over the time dimension (good). Make sure the **vocab-wide** `logsumexp` is also done in the loop (it is) and that the input `logits` is freed by the caller right after. Consider computing logprobs with `torch.log_softmax` on bf16 then casting the gathered scalars, instead of upcasting the full `[chunk, vocab]` slice to fp32.

### OOM-4 — Keep `per_device_train_batch_size = 1`, raise `gradient_accumulation_steps`
**File:** `configs/*.yaml`.
The trainer already loops micro-batches and accumulates. For 30B keep `per_device_train_batch_size: 1`, `rollout_batch_size: 1`, and increase `gradient_accumulation_steps` to hit your effective batch. This bounds the peak activation/logit memory.

### OOM-5 — 4-bit loader: add `max_memory` headroom and keep gradient checkpointing on
**File:** `models/loader.py`, `model_kwargs`.
`prepare_model_for_kbit_training(..., use_gradient_checkpointing=cfg.gradient_checkpointing)` is already wired — keep `gradient_checkpointing: true`. For the 30B MoE on a single 40GB A100, add a `max_memory` cap to leave room for activations and the verifier when co-resident:
```python
if has_cuda:
    model_kwargs["max_memory"] = {0: "36GiB", "cpu": "80GiB"}  # tune per card
```
Prefer an **80GB A100** for the 30B model; 40GB is tight once the verifier shares the card.

### OOM-6 — Separate verifier device when possible
On a 2-GPU node set `device: cuda:0` and `verifier_device: cuda:1` and use `--dispatcher subprocess` so compilation/benchmark memory never overlaps training. On single-GPU Colab the notebook sets `verifier_device = device` and `use_verifier_worker = False` (in-process) — correct, but then **generation, training, and kernel compilation share one card**, so keep batch sizes at 1 and turn budget modest.

### OOM-7 — Shorten the KV footprint during generation
- `max_prompt_length: 4096` with 4 teacher examples is large for contrastive prompts. Consider `max_teacher_examples: 2` and `max_prompt_length: 2048–3072` for the single-GPU run.
- `max_completion_length: 1536` is the dominant generation KV cost; 1024 is a reasonable first-pass value to confirm stability, then raise.

### OOM-8 — Always set the allocator flag
Already in notebook cell 2 and README:
```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```
Keep it; it meaningfully reduces fragmentation OOMs across the generate → forward → backward cycle.

---

## Part 3 — Thorough A100 Testing Instructions

Run these in order. Steps A–C are cheap and catch most problems before you spend GPU-hours.

### Prerequisites
- A100 (80GB strongly preferred for the 30B model; 40GB OK for the 7B dev model).
- CUDA toolkit with `nvcc` on PATH (needed by `load_inline`). Verify: `nvcc --version`.
- Python 3.10+.

### Step A — Install and preflight (no training)
```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
pip install -r requirements.txt
pip install -e .

# Static + unit checks (these FAILED to even collect in the review sandbox only
# because deps were missing — they should now run):
python -m pytest tests -q
python -m ruff check .

# Preflight: python, CUDA, compiler tools, configs, imports
python scripts/check_env.py --config configs/multiturn_grpo.yaml
```
**Expect:** pytest green (task registry, prompt building, reward shaping, archive, losses, rollout sample construction, eval stats, notebook structure, text hygiene). `check_env.py` reports CUDA available and `nvcc` found.

### Step B — Single-kernel compile + verify + benchmark sanity
This isolates the engine (compiler → verifier → benchmark) from RL.
```bash
# Measures stored teacher kernels and writes data/teacher_measurements.json
python scripts/verify_teachers.py --config configs/multiturn_grpo.yaml --device cuda:0
```
**Expect:** each teacher compiles, verifies correct, and reports a floored speedup; `data/teacher_measurements.json` is written; prompts later show measured (not `(unverified)`) speedups. If a teacher fails to compile here, the toolchain/headers are the problem, not the RL code.

### Step C — Dry run (1 step, 1 turn, in-process dispatcher)
```bash
python train_grpo.py \
  --arm arm_a --mode multi_turn \
  --config configs/multiturn_grpo.yaml \
  --use_dev_model --dry_run --dispatcher in_process \
  --device cuda:0 --verifier_device cuda:0
```
**Expect:** one rollout, at least one candidate compiled/verified/benchmarked, a non-crashing loss, and an adapter saved. Watch `nvidia-smi` — note peak memory for scaling decisions.

### Step D — Short real training, both arms (dev model first)
```bash
# Arm A (contrastive) and Arm B (reference-only), matched settings
python train_grpo.py --arm both --mode multi_turn \
  --config configs/multiturn_grpo.yaml \
  --use_dev_model --dispatcher auto \
  --device cuda:0 --verifier_device cuda:0
```
Set `max_steps` small (e.g. 10–20) for the first pass. **Expect:** `train_metrics.jsonl` written per arm, `run_metadata.json` with config + taxonomy + archive summaries, checkpoints under each arm dir, and **separate archive files per arm** (`..._arm_a.jsonl`, `..._arm_b.jsonl`).

**Verify P0-1 is fixed:** open a `TrajectoryRecord` / telemetry row and confirm each step's `task_id` is the real task and `prompt_mode` is `contrastive` for Arm A and `reference_only` for Arm B. Before the fix these are `""` / `reference_only` everywhere.

### Step E — Checkpoint resume
```bash
python train_grpo.py --arm arm_a --mode multi_turn \
  --config configs/multiturn_grpo.yaml --use_dev_model \
  --resume_from_checkpoint outputs/arm_a_contrastive/checkpoint-<N>
```
**Expect:** training resumes at the saved `global_step`; optimizer/scheduler/RNG restored (check the printed resume line).

### Step F — Evaluation + arm comparison
```bash
python evaluate.py --config configs/multiturn_grpo.yaml \
  --adapter_path outputs/arm_a_contrastive/final_adapter \
  --prompt_mode contrastive --dispatcher auto \
  --output_file outputs/eval_results/arm_a.json

python evaluate.py --config configs/multiturn_grpo.yaml \
  --adapter_path outputs/arm_b_reference_only/final_adapter \
  --prompt_mode reference_only --dispatcher auto \
  --output_file outputs/eval_results/arm_b.json

python scripts/compare_arms.py \
  --arm_a outputs/eval_results/arm_a.json \
  --arm_b outputs/eval_results/arm_b.json \
  --split all --output outputs/eval_results/comparison.json
```
**Expect:** per-eval JSON with correctness rate, attempt-correctness, fraction ≥1.0x / ≥1.1x, median/mean speedup, IQR, and a bootstrap CI; comparison reports wins/ties/losses, paired median diff, and a bootstrap interval. Also run `--split holdout` for the generalization claim and optionally `--base_model_control`.

### Step G — Contrastive mode (archive pairwise loss)
```bash
python train_grpo.py --arm arm_a --mode contrastive \
  --config configs/contrastive_rl.yaml --use_dev_model --dispatcher auto
```
**Expect:** the DPO pairwise term engages (requires a reference policy: either `--sft_adapter_path`, the frozen `sft` adapter, or `disable_adapter()`), and `all_pairs` yields winner/loser pairs once the archive has correct+incorrect kernels.

### Step H — Scale to the 30B model
Only after A–G pass on the dev model. Prefer an 80GB A100, two GPUs if available:
```bash
python train_grpo.py --arm both --mode multi_turn \
  --config configs/multiturn_grpo.yaml \
  --dispatcher subprocess --device cuda:0 --verifier_device cuda:1 \
  --sft_adapter_path <sft-adapter-or-merged-model>
```
Apply the Part 2b OOM items first (especially OOM-1, OOM-4, OOM-5). Watch `nvidia-smi -l 1` during the first rollout/backward to confirm peak memory has headroom.

### Component coverage checklist
- [ ] Static guard accepts valid kernels, rejects shortcuts/member-calls/manual CUDA alloc (unit tests + Step B).
- [ ] Compilation via `load_inline` with `-O3 --use_fast_math` (Step B/C).
- [ ] Numerical verification across all shapes/seeds; exact match for `column_max`, rtol/atol otherwise; mutation/alias/shape/dtype/non-finite rejection (Step B).
- [ ] Timing gate: 7 blocks, variance ≤ 0.005, min-measurement, fake-gain, implausible-speedup, post-timing re-check, 2-decimal floor (Step B).
- [ ] Reward shaping formula + clip; zero for failed candidates (metrics in Step D).
- [ ] Discounted returns + per-(task,turn) group advantages (inspect training samples / metrics).
- [ ] `<think>` extraction and masking from later prompts (inspect a multi-turn trajectory's prompts).
- [ ] Early stop at `early_stop_speedup` (trajectory metadata `early_stopped`).
- [ ] Dispatcher: in_process (Colab), subprocess (2-GPU), mock (CPU tests).
- [ ] Dr. GRPO loss with fixed `dr_grpo_l_norm` denominator; contrastive DPO loss (Steps D, G).
- [ ] Checkpoint save/resume incl. optimizer/scheduler/RNG (Step E).
- [ ] Evaluation stats + bootstrap CI + paired comparison (Step F).
- [ ] **P0-1 fix verified:** step-level `task_id`/`prompt_mode` correct in telemetry (Step D).

---

## Notebook note (`notebooks/colab_rl_pipeline.ipynb`)
The notebook already: sets the allocator flag (cell 2), pins deps (cell 4), runs an inline-compile sanity check (cell 6), configures single-GPU in-process dispatch (cell 8), verifies teachers (cell 10), inspects prompts (cell 12), supports SFT handoff (cell 14), does a dry run (cell 16), trains both arms (cell 18), evaluates (cell 20), and compares (cell 22). It uses the **dev (7B) model** by default. Before switching cell 8 to the 30B base model, apply OOM-1/4/5/7 and confirm Steps A–G on the dev model first. The per-cell `gc.collect()` + `torch.cuda.empty_cache()` pattern is already present between heavy cells — keep it.
