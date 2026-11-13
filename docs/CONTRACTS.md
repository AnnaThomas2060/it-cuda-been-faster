# Inter-Module Data Contracts

This document describes the typed contracts that connect the RL pipeline, the verifier, and the analytics workstream. All models live in `src/contracts/__init__.py` and are Pydantic models.

## Overview

`src.contracts` is the shared boundary between:

- Anna Thomas's pipeline (`src/pipeline/`, `src/rewards/`, `src/trainers/`, `engine/`).
- Pramod Goyal's verifier harness (`src/verifier/`).
- Mohamed Ziyadh Mohamed's baseline and analytics modules (`src/baselines/`, `src/analytics/`).

The dispatcher serializes a `KernelEvalRequest`, the worker returns a `KernelEvalResponse`, and the rollout manager stores per-turn samples and full trajectories.

## 1. Kernel Evaluation Request (`KernelEvalRequest`)

Sent from `src/pipeline/env_dispatcher.py` to the verifier worker.

| Field | Type | Description |
| :--- | :--- | :--- |
| `task_id` | `str` | Registered task identifier. |
| `code` | `str` | Candidate CUDA C++ source. |
| `baseline_code` | `Optional[str]` | Optional reference kernel code for external verifiers. |
| `turn_index` | `int` | Zero based turn position in the trajectory. |
| `timeout_seconds` | `float` | Compile, verify, and benchmark budget. Default 30.0. |
| `device_id` | `int` | Target GPU index. Default 0. |
| `run_sanitizer` | `bool` | Run `compute-sanitizer` memcheck for this request. Default false. |
| `metadata` | `Dict[str, Any]` | Free form request metadata such as prompt mode. |

## 2. Kernel Evaluation Response (`KernelEvalResponse`)

Returned by the verifier worker to the pipeline.

| Field | Type | Description |
| :--- | :--- | :--- |
| `task_id` | `str` | Registered task identifier. |
| `compiled` | `bool` | True when compilation succeeded. |
| `correct` | `bool` | True when numerical verification succeeded. |
| `execution_time_ms` | `Optional[float]` | Median candidate time. |
| `baseline_time_ms` | `Optional[float]` | Median baseline time. |
| `speedup` | `Optional[float]` | Floored baseline to candidate ratio. |
| `memory_bandwidth_gbps` | `Optional[float]` | Achieved candidate bandwidth. |
| `baseline_bandwidth_gbps` | `Optional[float]` | Achieved baseline bandwidth. |
| `bandwidth_delta` | `float` | Relative bandwidth change versus baseline. |
| `compiler_output` | `str` | Compiler diagnostics. |
| `error_message` | `Optional[str]` | Failure message. |
| `status` | `str` | Machine readable outcome such as `ok`, `compile_error`, `incorrect`, `unstable`, `fake_gain`, or `timeout`. |
| `stable` | `bool` | True when the timing gate passed. |
| `variance` | `float` | Population variance of block speedup ratios. |
| `compile_time_sec` | `float` | Compilation wall time. |
| `benchmark_time_sec` | `float` | Benchmark wall time. |
| `candidate_code_hash` | `str` | SHA256 hash of the sanitized candidate source. |
| `device_name` | `str` | Device used by the verifier. |
| `metadata` | `Dict[str, Any]` | Verification details, benchmark details, and sanitizer output. |

Status values are intentionally strings rather than an enum so teammate modules can add outcomes without a schema change.

## 3. Trajectory Step Sample (`TurnStepSample`)

Stores one turn of one trajectory and the tensors needed for training.

| Field | Type | Description |
| :--- | :--- | :--- |
| `task_id` | `str` | Registered task identifier. |
| `turn_index` | `int` | Zero based turn position. |
| `prompt` | `str` | Formatted prompt for this turn. |
| `thought` | `str` | Extracted `<think>` content, masked from later prompts. |
| `code` | `str` | Extracted CUDA source. |
| `response` | `Optional[KernelEvalResponse]` | Verifier response for this turn. |
| `reward` | `float` | Shaped step reward. |
| `discounted_return` | `float` | Discounted return from this turn. |
| `advantage` | `float` | Group normalized advantage. |
| `prompt_ids` | `List[int]` | Tokenized prompt ids. |
| `completion_ids` | `List[int]` | Tokenized completion ids up to the first EOS. |
| `old_logprobs` | `List[float]` | Behavior policy log probabilities for completion tokens. |
| `prompt_mode` | `str` | `reference_only` or `contrastive`. |

## 4. Trajectory Record (`TrajectoryRecord`)

Serializable form of a full multi-turn rollout, used for telemetry and offline analysis.

| Field | Type | Description |
| :--- | :--- | :--- |
| `trajectory_id` | `str` | Unique trajectory identifier. |
| `task_id` | `str` | Registered task identifier. |
| `prompt_mode` | `str` | Prompt arm. |
| `steps` | `List[TurnStepSample]` | Ordered turn samples. |
| `best_code` | `str` | Best verified kernel source. |
| `best_speedup` | `float` | Best verified speedup. |
| `best_status` | `str` | Status of the best turn. |
| `final_reward` | `float` | Discounted return of the first turn. |
| `metadata` | `Dict[str, Any]` | Early stop and other run flags. |

## 5. Telemetry Record (`TelemetryRecord`)

Appended to `logs/trajectory_telemetry.jsonl` and consumed by the analytics workstream.

| Field | Type | Description |
| :--- | :--- | :--- |
| `trajectory_id` | `str` | Unique trajectory identifier. |
| `task_id` | `str` | Registered task identifier. |
| `turn_index` | `int` | Turn index. |
| `speedup` | `Optional[float]` | Measured speedup. |
| `reward` | `float` | Scalar reward. |
| `compiled` | `bool` | Compilation flag. |
| `correct` | `bool` | Correctness flag. |
| `memory_bandwidth_gbps` | `Optional[float]` | Achieved bandwidth. |
| `timestamp` | `str` | ISO 8601 UTC timestamp. |
| `extra` | `Dict[str, Any]` | Supplementary metrics. |

The reward engine also writes a richer JSONL stream with code hashes, stability, variance, bandwidth delta, timings, device name, and error messages. Field additions are backward compatible for readers that ignore unknown keys.

## 6. Offline Testing

`MockVerifier` lives in `src/contracts/__init__.py` and implements the request and response contract without CUDA. The dispatchers that exercise this contract live in `src/pipeline/env_dispatcher.py`: `MockDispatcher` returns canned responses for unit tests, notebook dry runs, and CI, while `InProcessDispatcher` runs the real evaluation path without a worker process and is useful on a single GPU. (`SubprocessDispatcher` and the `build_dispatcher` factory also live there.)

## 7. Compatibility Notes

- New response fields have defaults, so older code that constructs `KernelEvalResponse` without them continues to work.
- The reward engine treats any candidate without `compiled`, `correct`, and `stable` set to true as a zero reward.
- Teammates may extend `metadata` freely. The pipeline only relies on documented top level fields.
