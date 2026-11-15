"""Tests for the analytics vertical. CPU only."""

import pytest

from src.analytics.fake_gain import audit, audit_record
from src.analytics.roofline import PEAK_BANDWIDTH_GBPS, min_bytes_moved, min_time_ms, utilization
from src.analytics.telemetry import append_record, load_records, reward_curve, summarize
from src.contracts import TelemetryRecord


def make(task_id="swish", i=0, **fields) -> TelemetryRecord:
    return TelemetryRecord(
        trajectory_id=f"t{i}",
        task_id=task_id,
        turn_index=0,
        timestamp=f"2026-01-01T00:00:{i:02d}Z",
        **fields,
    )


def test_log_round_trip(tmp_path) -> None:
    path = tmp_path / "logs" / "telemetry.jsonl"
    records = [make(i=0, reward=1.2, compiled=True, correct=True, speedup=1.2), make(i=1)]
    for record in records:
        append_record(record, path)
    assert load_records(path) == records


def test_summarize_ignores_incorrect_speedups() -> None:
    records = [
        make(i=0, compiled=True, correct=True, speedup=1.2, reward=1.2),
        make(i=1, compiled=True, correct=True, speedup=1.6, reward=1.6),
        make(i=2, compiled=True, correct=False, speedup=50.0),
        make(i=3),
        make("column_max", i=4),
    ]
    summary = summarize(records)
    assert summary["swish"]["attempts"] == 4
    assert summary["swish"]["compile_rate"] == 0.75
    assert summary["swish"]["correct_rate"] == 0.5
    assert summary["swish"]["median_speedup"] == pytest.approx(1.4)
    assert summary["swish"]["peak_speedup"] == 1.6
    assert summary["column_max"]["median_speedup"] is None


def test_reward_curve_is_time_ordered_rolling_mean() -> None:
    records = [make(i=2, reward=3.0), make(i=0, reward=0.0), make(i=1, reward=1.0)]
    assert reward_curve(records, window=2) == [0.0, 0.5, 2.0]


def test_roofline_bounds() -> None:
    # swish: read 8 floats, write 8 floats
    assert min_bytes_moved("swish", (2, 4)) == 64
    # column max over dim 0: read 8 floats, write 4
    assert min_bytes_moved("column_max", (2, 4)) == 48

    peak = PEAK_BANDWIDTH_GBPS["T4"]
    shape = (1024, 4096)
    floor_ms = min_time_ms("swish", shape, peak)
    assert utilization("swish", shape, floor_ms, peak) == pytest.approx(1.0)
    assert utilization("swish", shape, floor_ms * 2, peak) == pytest.approx(0.5)


def test_audit_flags() -> None:
    clean = make(i=0, compiled=True, correct=True, speedup=1.3, reward=1.3)
    assert audit_record(clean, peak_gbps=320.0) == []

    assert audit_record(make(reward=1.0, compiled=True)) == ["reward_without_correctness"]
    assert "correct_without_compile" in audit_record(make(correct=True))
    assert audit_record(make(compiled=True, correct=True, speedup=40.0)) == [
        "implausible_speedup"
    ]
    fast = make(compiled=True, correct=True, memory_bandwidth_gbps=900.0)
    assert audit_record(fast, peak_gbps=320.0) == ["above_roofline"]
    assert audit_record(fast) == []

    assert [r for r, _ in audit([clean, fast], peak_gbps=320.0)] == [fast]
