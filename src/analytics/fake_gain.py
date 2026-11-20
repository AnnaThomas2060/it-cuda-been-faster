"""Flag telemetry records whose reward or speedup cannot be trusted."""

from typing import Iterable, List, Optional, Tuple

from src.contracts import TelemetryRecord


def audit_record(
    record: TelemetryRecord,
    peak_gbps: Optional[float] = None,
    max_speedup: float = 10.0,
) -> List[str]:
    """Return the reasons this record looks like a fake gain (empty if clean)."""
    flags: List[str] = []
    if record.reward > 0 and not (record.compiled and record.correct):
        flags.append("reward_without_correctness")
    if record.correct and not record.compiled:
        flags.append("correct_without_compile")
    # ponytail: one fixed ceiling for all tasks, use a per-task roofline ceiling
    # once baseline timings are logged
    if record.speedup is not None and record.speedup > max_speedup:
        flags.append("implausible_speedup")
    bandwidth = record.memory_bandwidth_gbps
    if peak_gbps is not None and bandwidth is not None and bandwidth > peak_gbps:
        flags.append("above_roofline")
    return flags


def audit(
    records: Iterable[TelemetryRecord],
    peak_gbps: Optional[float] = None,
    max_speedup: float = 10.0,
) -> List[Tuple[TelemetryRecord, List[str]]]:
    """Return only the flagged records, each with its reasons."""
    flagged = ((r, audit_record(r, peak_gbps, max_speedup)) for r in records)
    return [(r, flags) for r, flags in flagged if flags]
