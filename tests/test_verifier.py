"""CPU tests for the isolated verifier; one GPU end-to-end test is opt-in."""

import itertools
import os

import pytest

from data.tasks import get_task
from rewards import grpo_reward
from src.contracts import KernelEvalRequest, KernelEvalResponse, MockVerifier
from src.verifier import KernelVerifier, VerifierConfig
from src.verifier.correctness import (
    random_case_schedule,
    tolerances,
    validation_shapes,
)
from src.verifier.harness import physical_device, sanitizer_passed
from src.verifier.timing import (
    ACCEPTED,
    REJECTED,
    VERIFICATION_REQUIRED,
    collect_timing_pairs,
    confirm_on_second_gpu,
    denoise_pairs,
    round_toward_one,
)

# --------------------------------------------------------------------- timing


def fake_clock(step):
    ticks = itertools.count()
    return lambda: next(ticks) * step


def test_collect_stops_after_window_and_minimum():
    out = collect_timing_pairs(
        lambda name: {"event_ms": 2.0 if name == "reference" else 1.0, "wall_ms": 1.0},
        seed=0, window_seconds=1.0, min_pairs=7, max_pairs=1000, clock=fake_clock(0.05),
    )
    assert out["window_complete"]
    assert len(out["pairs"]) >= 7
    assert all(p["reference_ms"] == 2.0 and p["candidate_ms"] == 1.0 for p in out["pairs"])


def test_collect_order_is_shuffled_but_seeded():
    def orders(seed):
        out = collect_timing_pairs(
            lambda name: {"event_ms": 1.0, "wall_ms": 1.0},
            seed=seed, window_seconds=0, min_pairs=20, max_pairs=20, clock=fake_clock(0),
        )
        return [tuple(p["order"]) for p in out["pairs"]]

    assert orders(1) == orders(1)
    assert len(set(orders(1))) == 2  # both orders occur


def test_collect_rejects_nonpositive_samples():
    with pytest.raises(ValueError):
        collect_timing_pairs(
            lambda name: {"event_ms": 0.0, "wall_ms": 1.0},
            seed=0, window_seconds=0, min_pairs=1, max_pairs=1, clock=fake_clock(0),
        )


def pairs_with_ratios(ratios):
    return [{"reference_ms": r, "candidate_ms": 1.0} for r in ratios]


def test_denoise_accepts_stable_speedup_and_rounds_toward_one():
    result = denoise_pairs(pairs_with_ratios([1.157] * 14))
    assert result["status"] == ACCEPTED
    assert result["speedup"] == 1.15


def test_denoise_rejects_high_block_variance():
    result = denoise_pairs(pairs_with_ratios([1.0, 1.0, 1.5, 1.5, 0.8, 0.8, 1.6] * 2))
    assert result["status"] == REJECTED
    assert result["speedup"] is None


def test_denoise_flags_suspicious_speedup():
    result = denoise_pairs(pairs_with_ratios([4.0] * 7))
    assert result["status"] == VERIFICATION_REQUIRED
    assert result["speedup"] is None


def test_round_toward_one():
    assert round_toward_one(1.15) == 1.15  # 1.15 * 100 == 114.99999999999999
    assert round_toward_one(1.159) == 1.15
    assert round_toward_one(0.981) == 0.99


def test_cross_gpu_confirmation():
    primary = denoise_pairs(pairs_with_ratios([4.0] * 7))
    agree = confirm_on_second_gpu(primary, denoise_pairs(pairs_with_ratios([3.8] * 7)), 0.10)
    assert agree["status"] == ACCEPTED and agree["speedup"] == 3.8
    disagree = confirm_on_second_gpu(primary, denoise_pairs(pairs_with_ratios([2.0] * 7)), 0.10)
    assert disagree["status"] == REJECTED and disagree["reason"] == "cross_gpu_mismatch"


# ---------------------------------------------------------------- correctness


def test_validation_shapes_include_odd_tails():
    shapes = validation_shapes(get_task("swish"))
    assert (3, 33) in shapes and (1024, 4096) in shapes
    assert all(len(s) == 4 for s in validation_shapes(get_task("average_pooling_2d")))


def test_schedule_is_balanced_grouped_and_deterministic():
    task = get_task("softmax")
    schedule = list(random_case_schedule(task, 1000, seed=635))
    assert len(schedule) == 1000
    assert schedule == list(random_case_schedule(task, 1000, seed=635))
    groups = [s for s, _ in itertools.groupby(schedule, key=lambda c: c["shape"])]
    assert groups == validation_shapes(task)  # each shape appears in one block


def test_column_max_requires_exact_match():
    assert tolerances(get_task("column_max")) == {"atol": 0.0, "rtol": 0.0}
    assert tolerances(get_task("swish")) == {"atol": 1e-4, "rtol": 1e-4}


# -------------------------------------------------------------------- harness


def test_sanitizer_parsing():
    assert sanitizer_passed(0, "========= ERROR SUMMARY: 0 errors")
    assert not sanitizer_passed(0, "========= ERROR SUMMARY: 2 errors")
    assert not sanitizer_passed(86, "========= ERROR SUMMARY: 0 errors")
    assert not sanitizer_passed(0, "no summary at all")


def test_device_mapping_respects_parent_visibility():
    assert physical_device(1, None) == "1"
    assert physical_device(0, "2,3") == "2"
    assert physical_device(1, "GPU-a, GPU-b") == "GPU-b"
    with pytest.raises(ValueError):
        physical_device(2, "0,1")


def test_static_failures_never_start_a_worker():
    verifier = KernelVerifier(python="/nonexistent/python")  # would fail if launched
    response = verifier.evaluate(KernelEvalRequest(task_id="swish", code="no kernel here"))
    assert response.failure_stage == "static_check"
    assert not response.compiled and not response.correct


INFO = {"returncode": 0, "timed_out": False, "seconds": 1.0, "log_tail": "", "mode": "full"}
REQ = KernelEvalRequest(task_id="swish", code="")


def response_for(result, **info):
    return KernelVerifier._response(REQ, result, {**INFO, **info}, {})


def test_response_for_compile_failure():
    r = response_for({"stage": "compile", "status": "failed", "message": "nvcc error"})
    assert (r.compiled, r.correct, r.failure_stage) == (False, False, "compile")


def test_response_for_incorrect_output():
    r = response_for({
        "stage": "correctness", "status": "failed", "message": "Values differ",
        "correctness": {"correct": False, "failed_stage": "random"},
    })
    assert (r.compiled, r.correct, r.failure_stage) == (True, False, "incorrect")


def test_response_for_timeout_and_crash():
    timeout = response_for({"stage": "correctness"}, timed_out=True, returncode=-9)
    assert timeout.failure_stage == "timeout"
    crash = response_for({"stage": "correctness"}, returncode=-11)
    assert crash.failure_stage == "runtime_error" and not crash.correct


def test_response_for_accepted_timing():
    timing = denoise_pairs(pairs_with_ratios([1.3] * 7))
    timing.update(candidate_bandwidth_gbps=800.0, reference_bandwidth_gbps=600.0)
    r = response_for({"stage": "done", "status": "ok", "correctness": {"correct": True},
                      "timing": timing})
    assert r.failure_stage is None and r.correct and r.speedup == 1.3
    assert r.memory_bandwidth_gbps == 800.0


def test_response_for_unstable_timing():
    timing = denoise_pairs(pairs_with_ratios([1.0, 1.0, 1.5, 1.5, 0.8, 0.8, 1.6]))
    r = response_for({"stage": "done", "status": "ok", "correctness": {"correct": True},
                      "timing": timing})
    assert r.correct and r.speedup is None
    assert r.failure_stage == "inter_block_variance_exceeded"


def test_evaluation_profile_is_strict():
    cfg = VerifierConfig.evaluation()
    assert cfg.random_cases == 1000 and cfg.memcheck and cfg.timing_window_seconds == 30.0
    assert VerifierConfig.evaluation(memcheck=False).memcheck is False


# --------------------------------------------------------------------- reward


class StubVerifier:
    def __init__(self, **fields):
        self.fields = fields

    def evaluate(self, request):
        return KernelEvalResponse(task_id=request.task_id, **self.fields)


@pytest.fixture
def reward_env(tmp_path):
    grpo_reward.reset_taxonomy_stats()
    yield tmp_path / "telemetry.jsonl"
    grpo_reward.configure_verifier(None, telemetry_path="")


def test_reward_uses_verifier_speedup_and_writes_telemetry(reward_env):
    grpo_reward.configure_verifier(MockVerifier(simulated_speedup=1.4), str(reward_env))
    assert grpo_reward.evaluate_single_completion("code", "swish") == 1.4
    assert grpo_reward.get_taxonomy_summary()["error_taxonomy"]["success"] == 1
    assert '"task_id": "swish"' in reward_env.read_text()


def test_reward_maps_failures_to_taxonomy(reward_env):
    grpo_reward.configure_verifier(
        StubVerifier(compiled=True, correct=True, speedup=None,
                     failure_stage="inter_block_variance_exceeded")
    )
    assert grpo_reward.evaluate_single_completion("code", "swish") == 0.0
    assert grpo_reward.get_taxonomy_summary()["error_taxonomy"]["timing_unstable"] == 1


# ----------------------------------------------------------------- GPU (opt-in)

gpu = pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS") != "1", reason="set RUN_GPU_TESTS=1 on a CUDA machine"
)


@gpu
@pytest.mark.parametrize("task_id", ["swish", "column_max", "channel_rmsnorm"])
def test_teacher_kernels_pass_end_to_end(task_id):
    from data.example_store import get_teacher_examples

    verifier = KernelVerifier(VerifierConfig.training(timing_window_seconds=1.0))
    for example in get_teacher_examples(task_id):
        response = verifier.evaluate(KernelEvalRequest(task_id=task_id, code=example["code"]))
        assert response.compiled, response.error_message
        assert response.correct, f"{example['name']}: {response.error_message}"
        assert response.execution_time_ms and response.baseline_time_ms


@gpu
def test_missing_bounds_check_is_caught_by_memcheck():
    # Out-of-bounds writes can leave every in-range output correct, so only
    # Compute Sanitizer reliably catches this class of bug.
    from data.example_store import get_teacher_examples

    code = get_teacher_examples("swish")[0]["code"].replace("if (idx < total_elements)", "")
    verifier = KernelVerifier(VerifierConfig.training(memcheck=True, random_cases=16))
    response = verifier.evaluate(KernelEvalRequest(task_id="swish", code=code))
    assert response.compiled and not response.correct
    assert response.failure_stage == "memcheck"
