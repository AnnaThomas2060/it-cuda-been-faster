"""Roofline bounds for memory-bound kernels.

A kernel must read its input and write its output at least once, and it cannot
move those bytes faster than the GPU's memory bandwidth. That gives a hard floor
on runtime. A measured time below the floor means the kernel skipped the work.
"""

import math
from typing import Dict, Sequence

from data.tasks import get_task

# Published peak memory bandwidth in GB/s. Real cards reach somewhat less,
# so pass a measured value to the functions below when you have one.
PEAK_BANDWIDTH_GBPS: Dict[str, float] = {
    "RTX 4050": 192.0,
    "T4": 320.0,
    "L4": 300.0,
    "RTX 3090": 936.2,
    "V100": 900.0,
    "A100-40GB": 1555.0,
    "A100-80GB": 2039.0,
}

BYTES_PER_FLOAT32 = 4


def min_bytes_moved(task_id: str, shape: Sequence[int]) -> int:
    """Bytes for one read of the input plus one write of the output."""
    import torch

    output = get_task(task_id)["fn"](torch.zeros(tuple(shape)))
    return BYTES_PER_FLOAT32 * (math.prod(shape) + output.numel())


def achieved_bandwidth_gbps(bytes_moved: int, time_ms: float) -> float:
    return bytes_moved / 1e9 / (time_ms / 1e3)


def min_time_ms(task_id: str, shape: Sequence[int], peak_gbps: float) -> float:
    """Fastest physically possible runtime for this task and shape."""
    return min_bytes_moved(task_id, shape) / 1e9 / peak_gbps * 1e3


def utilization(task_id: str, shape: Sequence[int], time_ms: float, peak_gbps: float) -> float:
    """Fraction of peak bandwidth the kernel reached. Above 1.0 is not physically possible."""
    return achieved_bandwidth_gbps(min_bytes_moved(task_id, shape), time_ms) / peak_gbps
