"""Verifier settings with two named profiles.

`training` keeps a GRPO reward call to a few seconds after compilation.
`evaluation` is the strict protocol used for reported numbers: 1000 random
cases, a 30 s timing window and Compute Sanitizer memcheck.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class VerifierConfig:
    # Correctness
    random_cases: int = 64
    adversarial_cases: bool = True
    nondefault_stream_check: bool = True
    seed: int = 635

    # Timing
    warmup_calls: int = 10
    repeats_per_sample: int = 20
    timing_window_seconds: float = 3.0
    min_timing_pairs: int = 7
    max_timing_pairs: int = 10_000
    block_count: int = 7
    variance_limit: float = 0.005
    # Speedups above this need confirmation on `verification_device_id`.
    suspicious_speedup: float = 3.0
    cross_gpu_tolerance: float = 0.10

    # Isolation
    timeout_seconds: float = 180.0
    memcheck: bool = False
    compute_sanitizer: str = "compute-sanitizer"
    verification_device_id: Optional[int] = None

    @classmethod
    def training(cls, **overrides: Any) -> "VerifierConfig":
        return cls(**overrides)

    @classmethod
    def evaluation(cls, **overrides: Any) -> "VerifierConfig":
        strict = dict(
            random_cases=1000,
            repeats_per_sample=100,
            timing_window_seconds=30.0,
            timeout_seconds=600.0,
            memcheck=True,
        )
        strict.update(overrides)
        return cls(**strict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def suspicious_threshold(task_id: str, base: float, teacher_factor: float = 2.0) -> float:
    """Speedup above which a candidate needs second-GPU confirmation.

    `max(base, teacher_factor * best verified teacher speedup)`, after
    CUDA-L1's "more than 2x the previous best" rule. A flat 3x would flag every
    good kernel on tasks where verified teachers already exceed 3x (channel
    RMSNorm teachers measure about 5x), so those kernels would get no reward
    whenever no spare GPU is free for confirmation. Tasks without a measured
    teacher keep `base`.
    """
    try:
        from data.example_store import get_teacher_examples
    except ImportError:
        return base
    measured = [
        float(t["speedup"])
        for t in get_teacher_examples(task_id)
        if t.get("verified") and t.get("speedup")
    ]
    return max([base] + [teacher_factor * s for s in measured])
