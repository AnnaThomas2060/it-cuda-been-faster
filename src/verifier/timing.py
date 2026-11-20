"""Paired latency sampling and block-variance denoising.

Pure Python: the GPU measurement is injected as `measure(name)`, so the
stopping rule and statistics are unit-testable without CUDA.

Protocol (adapted from CUDA-L1, Sec. 3.3):
  * each pair times the reference and the candidate back to back, in a
    seeded random order, so slow drift affects both sides equally;
  * pairs are split chronologically into `block_count` blocks;
  * the score is the median of block-mean speedups, rejected when the
    population variance of block means exceeds `variance_limit`;
  * accepted scores are rounded to two decimals toward 1.0, so timing noise
    cannot create a speedup.
"""

import math
import random
import statistics
import time
from typing import Callable, Dict, List

ACCEPTED = "accepted"
REJECTED = "rejected"
VERIFICATION_REQUIRED = "verification_required"


def _positive_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def collect_timing_pairs(
    measure: Callable[[str], Dict[str, float]],
    *,
    seed: int,
    window_seconds: float,
    min_pairs: int,
    max_pairs: int,
    clock: Callable[[], float] = time.monotonic,
) -> Dict[str, object]:
    """Sample (reference, candidate) pairs until the window and minimum are met.

    `measure(name)` returns `{"event_ms": ..., "wall_ms": ...}` per invocation
    for `name` in {"reference", "candidate"}. Warm-up is the caller's job.
    """
    if not 1 <= min_pairs <= max_pairs:
        raise ValueError("require 1 <= min_pairs <= max_pairs")
    rng = random.Random(seed)
    started = clock()
    pairs: List[Dict[str, object]] = []
    while len(pairs) < max_pairs:
        if len(pairs) >= min_pairs and clock() - started >= window_seconds:
            break
        order = ["reference", "candidate"]
        rng.shuffle(order)
        pair: Dict[str, object] = {"index": len(pairs), "order": order}
        for name in order:
            sample = measure(name)
            for key in ("event_ms", "wall_ms"):
                _positive_finite(sample.get(key), f"{name} {key}")
            pair[f"{name}_ms"] = float(sample["event_ms"])
            pair[f"{name}_wall_ms"] = float(sample["wall_ms"])
        pairs.append(pair)
    elapsed = clock() - started
    return {
        "pairs": pairs,
        "elapsed_seconds": elapsed,
        "window_complete": elapsed >= window_seconds and len(pairs) >= min_pairs,
    }


def round_toward_one(value: float) -> float:
    """Two-decimal rounding that never moves a score away from 1.0."""
    scaled = round(value * 100.0, 6)  # absorb float artefacts such as 114.99999
    return (math.floor(scaled) if value >= 1.0 else math.ceil(scaled)) / 100.0


def denoise_pairs(
    pairs: List[Dict[str, object]],
    *,
    block_count: int = 7,
    variance_limit: float = 0.005,
    suspicious_speedup: float = 3.0,
) -> Dict[str, object]:
    """Turn raw pairs into a speedup decision. Never raises on noisy data."""
    if block_count < 1:
        raise ValueError("block_count must be positive")
    if len(pairs) < block_count:
        raise ValueError(f"need at least {block_count} pairs, got {len(pairs)}")
    ratios = [
        _positive_finite(p["reference_ms"], "reference_ms")
        / _positive_finite(p["candidate_ms"], "candidate_ms")
        for p in pairs
    ]

    # Chronological blocks; the remainder goes one-per-block to the earliest.
    size, extra = divmod(len(ratios), block_count)
    block_means, offset = [], 0
    for index in range(block_count):
        count = size + (index < extra)
        block_means.append(sum(ratios[offset:offset + count]) / count)
        offset += count

    mean = sum(block_means) / block_count
    variance = sum((m - mean) ** 2 for m in block_means) / block_count
    raw = statistics.median(block_means)

    if variance > variance_limit:
        status, reason = REJECTED, "inter_block_variance_exceeded"
    elif raw > suspicious_speedup:
        status, reason = VERIFICATION_REQUIRED, "speedup_above_suspicious_threshold"
    else:
        status, reason = ACCEPTED, None

    return {
        "status": status,
        "reason": reason,
        "raw_speedup": raw,
        "speedup": round_toward_one(raw) if status == ACCEPTED else None,
        "variance": variance,
        "block_means": block_means,
        "pair_count": len(pairs),
        "median_reference_ms": statistics.median(p["reference_ms"] for p in pairs),
        "median_candidate_ms": statistics.median(p["candidate_ms"] for p in pairs),
    }


def confirm_on_second_gpu(
    primary: Dict[str, object], secondary: Dict[str, object], tolerance: float
) -> Dict[str, object]:
    """Accept a suspicious speedup only if a second GPU reproduces it."""
    a, b = primary["raw_speedup"], secondary["raw_speedup"]
    agrees = secondary["status"] != REJECTED and abs(a - b) / a <= tolerance
    return {
        **primary,
        "status": ACCEPTED if agrees else REJECTED,
        "reason": None if agrees else "cross_gpu_mismatch",
        "speedup": round_toward_one(min(a, b)) if agrees else None,
        "secondary_raw_speedup": b,
    }


def measure_callable(torch_module, fn: Callable, inputs: tuple, repeats: int) -> Dict[str, float]:
    """Per-invocation latency of `fn(*inputs)` over `repeats` back-to-back calls.

    The interval ends after a full device sync, so it includes launch and
    allocation overhead: a conservative invocation-level measurement, not
    isolated kernel time. Both CUDA-event and wall-clock values are returned.
    """
    cuda = torch_module.cuda
    device = inputs[0].device
    start = cuda.Event(enable_timing=True)
    end = cuda.Event(enable_timing=True)
    cuda.synchronize(device)
    wall_start = time.perf_counter_ns()
    start.record()
    for _ in range(repeats):
        fn(*inputs)
    cuda.synchronize(device)
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter_ns() - wall_start) / 1e6 / repeats
    return {"event_ms": start.elapsed_time(end) / repeats, "wall_ms": wall_ms}
