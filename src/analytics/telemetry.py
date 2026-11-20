"""Load, append, and summarize trajectory telemetry records."""

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from src.contracts import TelemetryRecord

DEFAULT_LOG_PATH = Path("logs/trajectory_telemetry.jsonl")


def append_record(record: TelemetryRecord, path: Union[str, Path] = DEFAULT_LOG_PATH) -> None:
    """Append one record to the JSONL log, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def load_records(path: Union[str, Path] = DEFAULT_LOG_PATH) -> List[TelemetryRecord]:
    """Read every record from the JSONL log. A malformed line raises."""
    with open(path, encoding="utf-8") as f:
        return [TelemetryRecord.model_validate_json(line) for line in f if line.strip()]


def summarize(records: Iterable[TelemetryRecord]) -> Dict[str, Dict[str, Any]]:
    """Per-task attempt counts, pass rates, mean reward, and median/peak speedup.

    Speedups only count records that passed correctness.
    """
    by_task: Dict[str, List[TelemetryRecord]] = defaultdict(list)
    for record in records:
        by_task[record.task_id].append(record)

    summary: Dict[str, Dict[str, Any]] = {}
    for task_id, recs in by_task.items():
        speedups = [r.speedup for r in recs if r.correct and r.speedup is not None]
        summary[task_id] = {
            "attempts": len(recs),
            "compile_rate": sum(r.compiled for r in recs) / len(recs),
            "correct_rate": sum(r.correct for r in recs) / len(recs),
            "mean_reward": statistics.fmean(r.reward for r in recs),
            "median_speedup": statistics.median(speedups) if speedups else None,
            "peak_speedup": max(speedups, default=None),
        }
    return summary


def reward_curve(records: Iterable[TelemetryRecord], window: int = 10) -> List[float]:
    """Rolling mean reward in time order, for checking convergence."""
    # ponytail: sorts ISO-8601 strings as text, parse them if writers mix timezones
    rewards = [r.reward for r in sorted(records, key=lambda r: r.timestamp)]
    return [
        statistics.fmean(rewards[max(0, i - window + 1): i + 1]) for i in range(len(rewards))
    ]
