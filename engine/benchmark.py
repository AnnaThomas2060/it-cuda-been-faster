import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from data.tasks import get_task, task_io_bytes


@dataclass
class BenchmarkResult:
    speedup: float
    stable: bool
    status: str
    message: str
    variance: float = 0.0
    median_ref_ms: float = 0.0
    median_cand_ms: float = 0.0
    candidate_bandwidth_gbps: float = 0.0
    baseline_bandwidth_gbps: float = 0.0
    bandwidth_delta: float = 0.0
    block_ratios: List[float] = field(default_factory=list)
    device_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_variance(values: List[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    return sum((value - mean_value) ** 2 for value in values) / len(values)


def floor_speedup(ratio: float) -> float:
    return math.floor(round(ratio * 100.0, 6)) / 100.0


def peak_memory_bandwidth_gbps(device: str) -> Optional[float]:
    try:
        import torch

        props = torch.cuda.get_device_properties(device)
        clock_hz = float(props.memory_clock_rate) * 1000.0
        bus_bytes = float(props.memory_bus_width) / 8.0
        return 2.0 * clock_hz * bus_bytes / 1e9
    except Exception:
        return None


def _sanity_check(
    module: Any, task: Dict[str, Any], inputs: List[Any], device: str
) -> Tuple[bool, str]:
    import torch

    rtol = float(task.get("rtol", 1e-4))
    atol = float(task.get("atol", 1e-4))
    exact = bool(task.get("exact_match", False))
    for inp in inputs:
        ref_out = task["fn"](inp)
        cand_out = module.run_cuda(inp)
        torch.cuda.synchronize()
        if not isinstance(cand_out, torch.Tensor):
            return False, "Candidate did not return a tensor during timing sanity check."
        if cand_out.shape != ref_out.shape:
            return False, "Candidate output shape changed during timing sanity check."
        if exact:
            if not torch.equal(cand_out, ref_out):
                return False, "Candidate output changed during timing sanity check."
        elif not torch.allclose(cand_out, ref_out, rtol=rtol, atol=atol):
            return False, "Candidate output failed the post-timing correctness check."
    return True, "Post-timing sanity check passed."


def benchmark_kernel(
    module: Any,
    task_id: str,
    device: str = "cuda",
    warmup_iters: int = 10,
    num_blocks: int = 7,
    runs_per_block: int = 10,
    variance_threshold: float = 0.005,
    min_measurement_ms: float = 0.005,
    use_bandwidth: bool = True,
    fake_gain_ratio: float = 1.20,
    max_plausible_speedup: float = 10.0,
) -> BenchmarkResult:
    try:
        import torch
    except ImportError:
        return BenchmarkResult(
            0.0, False, "runtime_unavailable", "PyTorch is required for benchmarking."
        )

    if not torch.cuda.is_available():
        return BenchmarkResult(0.0, False, "runtime_unavailable", "CUDA device is not available.")

    task = get_task(task_id)
    bench_shape = task["shapes"][-1]
    ref_fn = task["fn"]
    input_gen = task["input_gen"]

    try:
        with torch.cuda.device(device):
            device_name = torch.cuda.get_device_name(device)
            inputs = [input_gen(bench_shape, device=device, seed=42 + idx) for idx in range(3)]

            for _ in range(warmup_iters):
                for inp in inputs:
                    _ = ref_fn(inp)
                    _ = module.run_cuda(inp)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            block_ratios: List[float] = []
            ref_samples: List[float] = []
            cand_samples: List[float] = []

            for block_index in range(num_blocks):
                ref_times: List[float] = []
                cand_times: List[float] = []
                for run_index in range(runs_per_block):
                    inp = inputs[(block_index * runs_per_block + run_index) % len(inputs)]
                    torch.cuda.synchronize()
                    start_event.record()
                    _ = ref_fn(inp)
                    end_event.record()
                    end_event.synchronize()
                    ref_times.append(start_event.elapsed_time(end_event))

                    torch.cuda.synchronize()
                    start_event.record()
                    _ = module.run_cuda(inp)
                    end_event.record()
                    end_event.synchronize()
                    cand_times.append(start_event.elapsed_time(end_event))

                ref_times.sort()
                cand_times.sort()
                med_ref = ref_times[len(ref_times) // 2]
                med_cand = cand_times[len(cand_times) // 2]
                if med_cand <= 0.0:
                    return BenchmarkResult(
                        0.0, False, "invalid_timing", "Candidate timing is not positive."
                    )
                ref_samples.append(med_ref)
                cand_samples.append(med_cand)
                block_ratios.append(med_ref / med_cand)

            variance = compute_variance(block_ratios)
            ref_samples.sort()
            cand_samples.sort()
            median_ref = ref_samples[len(ref_samples) // 2]
            median_cand = cand_samples[len(cand_samples) // 2]
            median_ratio = median_ref / median_cand
            speedup = floor_speedup(median_ratio)

            io_bytes = task_io_bytes(task_id, bench_shape) if use_bandwidth else 0
            candidate_bw = io_bytes / (median_cand * 1e-3) / 1e9 if median_cand > 0 else 0.0
            baseline_bw = io_bytes / (median_ref * 1e-3) / 1e9 if median_ref > 0 else 0.0
            bw_delta = (candidate_bw - baseline_bw) / baseline_bw if baseline_bw > 0 else 0.0

            metadata = {
                "bench_shape": list(bench_shape),
                "io_bytes": io_bytes,
                "peak_bandwidth_gbps": peak_memory_bandwidth_gbps(device),
            }

            if variance > variance_threshold:
                return BenchmarkResult(
                    0.0,
                    False,
                    "unstable",
                    f"Measurement variance {variance:.6f} exceeded {variance_threshold:.6f}.",
                    variance=variance,
                    median_ref_ms=median_ref,
                    median_cand_ms=median_cand,
                    candidate_bandwidth_gbps=candidate_bw,
                    baseline_bandwidth_gbps=baseline_bw,
                    bandwidth_delta=bw_delta,
                    block_ratios=block_ratios,
                    device_name=device_name,
                    metadata=metadata,
                )

            if median_cand < min_measurement_ms:
                return BenchmarkResult(
                    0.0,
                    False,
                    "measurement_too_small",
                    f"Median candidate time {median_cand:.6f} ms is below the timer floor.",
                    variance=variance,
                    median_ref_ms=median_ref,
                    median_cand_ms=median_cand,
                    candidate_bandwidth_gbps=candidate_bw,
                    baseline_bandwidth_gbps=baseline_bw,
                    bandwidth_delta=bw_delta,
                    block_ratios=block_ratios,
                    device_name=device_name,
                    metadata=metadata,
                )

            peak_bw = metadata["peak_bandwidth_gbps"]
            if use_bandwidth and peak_bw and candidate_bw > peak_bw * fake_gain_ratio:
                return BenchmarkResult(
                    0.0,
                    False,
                    "fake_gain",
                    f"Candidate bandwidth {candidate_bw:.1f} GB/s exceeds the device peak.",
                    variance=variance,
                    median_ref_ms=median_ref,
                    median_cand_ms=median_cand,
                    candidate_bandwidth_gbps=candidate_bw,
                    baseline_bandwidth_gbps=baseline_bw,
                    bandwidth_delta=bw_delta,
                    block_ratios=block_ratios,
                    device_name=device_name,
                    metadata=metadata,
                )

            if speedup > max_plausible_speedup:
                return BenchmarkResult(
                    0.0,
                    False,
                    "implausible_speedup",
                    f"Speedup {speedup:.2f}x exceeds the plausibility limit.",
                    variance=variance,
                    median_ref_ms=median_ref,
                    median_cand_ms=median_cand,
                    candidate_bandwidth_gbps=candidate_bw,
                    baseline_bandwidth_gbps=baseline_bw,
                    bandwidth_delta=bw_delta,
                    block_ratios=block_ratios,
                    device_name=device_name,
                    metadata=metadata,
                )

            passed, reason = _sanity_check(module, task, inputs, device)
            if not passed:
                return BenchmarkResult(
                    0.0,
                    False,
                    "cache_or_incorrect",
                    reason,
                    variance=variance,
                    median_ref_ms=median_ref,
                    median_cand_ms=median_cand,
                    candidate_bandwidth_gbps=candidate_bw,
                    baseline_bandwidth_gbps=baseline_bw,
                    bandwidth_delta=bw_delta,
                    block_ratios=block_ratios,
                    device_name=device_name,
                    metadata=metadata,
                )

            return BenchmarkResult(
                speedup,
                True,
                "ok",
                "Benchmark passed.",
                variance=variance,
                median_ref_ms=median_ref,
                median_cand_ms=median_cand,
                candidate_bandwidth_gbps=candidate_bw,
                baseline_bandwidth_gbps=baseline_bw,
                bandwidth_delta=bw_delta,
                block_ratios=block_ratios,
                device_name=device_name,
                metadata=metadata,
            )
    except Exception as exc:
        return BenchmarkResult(
            0.0, False, "runtime_error", f"Benchmark runtime error: {str(exc)[:400]}"
        )


def measure_speedup(
    module: Any,
    task_id: str,
    device: str = "cuda",
    warmup_iters: int = 10,
    num_blocks: int = 7,
    runs_per_block: int = 10,
    variance_threshold: float = 0.005,
) -> Tuple[float, bool]:
    result = benchmark_kernel(
        module,
        task_id,
        device=device,
        warmup_iters=warmup_iters,
        num_blocks=num_blocks,
        runs_per_block=runs_per_block,
        variance_threshold=variance_threshold,
    )
    return result.speedup, result.stable
