# Verifier Vertical

Owner: Pramod Goyal

Compiles, validates and times generated CUDA kernels in an isolated child process.
Implements the `KernelEvalRequest -> KernelEvalResponse` contract from `src.contracts`.

```python
from src.contracts import KernelEvalRequest
from src.verifier import KernelVerifier, VerifierConfig

verifier = KernelVerifier(VerifierConfig.training())      # used by rewards/grpo_reward.py
strict = KernelVerifier(VerifierConfig.evaluation())      # used by evaluate.py
response = verifier.evaluate(KernelEvalRequest(task_id="swish", code=completion))
response.correct, response.speedup, response.failure_stage
```

## Pipeline

```
parent (training process)                 child: python -m src.verifier.worker <job>
-------------------------                 -------------------------------------------
static checks (no GPU)  --fail--> reward 0
write job/request.json  --spawn-->        compile (load_inline, -O3)
  own process group, timeout              self-check: 5 known-bad callables must be rejected
  [compute-sanitizer memcheck]            correctness suite (below)
                                          paired timing + block-variance denoise
read job/result_<mode>.json  <--          result written after every stage
map to KernelEvalResponse
```

A candidate that segfaults, hangs or corrupts its CUDA context ends only the child process.
Because the result file is rewritten after each stage, the parent still knows where the child
stopped.

## Correctness suite

| Check | Catches |
|---|---|
| N seeded random cases over benchmark shapes plus odd shapes `(1,1)`, `(3,33)`, `(7,257)` | missing tail / bounds handling |
| Same-shape cases reuse one input buffer with new values | output caching keyed on the input pointer |
| Fresh allocations are NaN-filled (`fill_uninitialized_memory`) | unwritten output elements |
| 6 adversarial patterns (zeros, ±1000, cancellation, ties, ...) | overflow, unstable reductions |
| Input written after a delay on a non-default stream | kernels that ignore the current stream |
| Input compared after every call | in-place mutation |
| Compute Sanitizer memcheck (evaluation profile) | out-of-bounds or misaligned accesses |

`column_max` requires bitwise equality. All other tasks use `atol = rtol = 1e-4`.

## Timing protocol

Reference and candidate are timed back to back in a seeded random order. Each sample covers
`repeats` invocations and ends with a device sync, so launch overhead is included (a
conservative measurement). Sampling continues for the timing window, with a minimum of 7 pairs.
The pairs are then split chronologically into 7 blocks:

* score = median of block-mean speedups, rounded to two decimals toward 1.0;
* rejected (reward 0) if the population variance of block means is above 0.005;
* scores above 3x must be reproduced within 10% on `verification_device_id`.

Bandwidth = (input bytes + output bytes) / median latency: the minimum DRAM traffic.

| Profile | random cases | window | repeats | memcheck | timeout |
|---|---|---|---|---|---|
| `training()` | 64 | 3 s | 20 | off | 180 s |
| `evaluation()` | 1000 | 30 s | 100 | on | 600 s |

## failure_stage values

`static_check`, `compile`, `incorrect`, `memcheck`, `runtime_error`, `worker_crash`, `timeout`,
`inter_block_variance_exceeded`, `speedup_above_suspicious_threshold`, `cross_gpu_mismatch`.
`None` means a verified speedup.

## Tests

```bash
pytest tests/test_verifier.py                          # CPU: timing stats, schedules, mapping
RUN_GPU_TESTS=1 pytest tests/test_verifier.py -k gpu   # teacher kernels end to end
```

On a two-GPU machine, train on GPU 0 and set `VERIFIER_DEVICE=1` so reward timing is not
disturbed by the policy's own kernels.
