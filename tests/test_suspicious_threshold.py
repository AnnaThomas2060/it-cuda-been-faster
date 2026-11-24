"""The >3x suspicion threshold scales with each task's best verified teacher."""

import json

import pytest

from src.verifier.config import suspicious_threshold


@pytest.fixture
def measurements(tmp_path, monkeypatch):
    path = tmp_path / "teacher_measurements.json"
    path.write_text(json.dumps({
        "channel_rmsnorm": [{"name": "warp_reduced_rmsnorm", "speedup": 4.96}],
        "swish": [{"name": "basic_elementwise_swish", "speedup": 1.2}],
    }))
    monkeypatch.setenv("TEACHER_MEASUREMENTS_PATH", str(path))


def test_fast_teacher_raises_threshold(measurements):
    assert suspicious_threshold("channel_rmsnorm", 3.0) == pytest.approx(9.92)


def test_slow_teacher_keeps_base(measurements):
    assert suspicious_threshold("swish", 3.0) == 3.0


def test_unmeasured_task_keeps_base(tmp_path, monkeypatch):
    monkeypatch.setenv("TEACHER_MEASUREMENTS_PATH", str(tmp_path / "missing.json"))
    assert suspicious_threshold("channel_rmsnorm", 3.0) == 3.0
