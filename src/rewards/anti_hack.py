import os
import shutil
from typing import Optional, Tuple

from engine.compiler import run_static_checks
from src.contracts import KernelEvalResponse


def static_audit(code: str) -> Tuple[bool, str]:
    return run_static_checks(code)


def timer_is_trustworthy(
    variance: float,
    median_time_ms: float,
    variance_threshold: float,
    min_measurement_ms: float,
) -> Tuple[bool, str]:
    if median_time_ms <= 0.0:
        return False, "non_positive_timing"
    if median_time_ms < min_measurement_ms:
        return False, "measurement_too_small"
    if variance > variance_threshold:
        return False, "unstable_timing"
    return True, "ok"


def is_fake_gain(
    response: KernelEvalResponse,
    peak_bandwidth_gbps: Optional[float],
    tolerance_ratio: float,
    max_plausible_speedup: float,
) -> Tuple[bool, str]:
    if response.speedup is not None and response.speedup > max_plausible_speedup:
        return True, "implausible_speedup"
    if (
        peak_bandwidth_gbps
        and response.memory_bandwidth_gbps
        and response.memory_bandwidth_gbps > peak_bandwidth_gbps * tolerance_ratio
    ):
        return True, "bandwidth_exceeds_peak"
    return False, "ok"


def sanitizer_available() -> bool:
    return shutil.which("compute-sanitizer") is not None


def sanitizer_enabled_by_env() -> bool:
    return os.environ.get("ENABLE_CUDA_SANITIZER", "0") == "1"
