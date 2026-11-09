import math
from typing import Any, List, Tuple

from data.tasks import get_task

# Latency benchmarking engine using high precision CUDA events and chronological blocks.
# Evaluates execution time over seven blocks and rejects measurements exceeding the variance gate.
# Final speedup ratios are floored to two decimal places to penalize noise artifacts.

def measure_speedup(
    module: Any,
    task_id: str,
    device: str = "cuda",
    warmup_iters: int = 10,
    num_blocks: int = 7,
    runs_per_block: int = 10,
    variance_threshold: float = 0.005,
) -> Tuple[float, bool]:
    try:
        import torch
    except ImportError:
        return 0.0, False

    if not torch.cuda.is_available():
        return 0.0, False

    task = get_task(task_id)
    # Select largest shape to ensure execution time dominates launch overhead.
    bench_shape = task["shapes"][-1]
    ref_fn = task["fn"]
    input_gen = task["input_gen"]

    try:
        with torch.cuda.device(device):
            inp = input_gen(bench_shape, device=device, seed=42)

            # Warmup phase for caches and clock stabilization
            for _ in range(warmup_iters):
                _ = ref_fn(inp)
                _ = module.run_cuda(inp)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            block_ratios: List[float] = []

            for _ in range(num_blocks):
                ref_times: List[float] = []
                cand_times: List[float] = []

                for _ in range(runs_per_block):
                    # Measure PyTorch reference
                    torch.cuda.synchronize()
                    start_event.record()
                    _ = ref_fn(inp)
                    end_event.record()
                    end_event.synchronize()
                    ref_times.append(start_event.elapsed_time(end_event))

                    # Measure candidate CUDA kernel
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
                    return 0.0, False

                ratio = med_ref / med_cand
                block_ratios.append(ratio)

            # Population variance calculation: sigma^2 = (1/N) * sum((x - mean)^2)
            mean_ratio = sum(block_ratios) / len(block_ratios)
            variance = sum((r - mean_ratio) ** 2 for r in block_ratios) / len(block_ratios)

            if variance > variance_threshold:
                return 0.0, False

            block_ratios.sort()
            median_ratio = block_ratios[len(block_ratios) // 2]
            # Prevent IEEE 754 precision artifacts (e.g., 1.15 * 100 -> 114.999...)
            floored_speedup = math.floor(round(median_ratio * 100.0, 6)) / 100.0

            return floored_speedup, True

    except Exception:
        return 0.0, False
