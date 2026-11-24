"""Copying a prompt example earns no reward and is never verified."""

import re

import pytest

from configs.grpo_config import ExperimentConfig
from data.example_store import get_teacher_examples
from engine.prompt_builder import build_turn_messages
from src.contracts import KernelEvalResponse
from src.rewards.novelty import max_example_similarity, prompt_examples, token_similarity
from src.rewards.reward_engine import RewardEngine

TEACHER = get_teacher_examples("channel_rmsnorm")[0]["code"]
OWN_BEST = (
    "__global__ void mine(float* y) {}\n"
    "torch::Tensor run_cuda(torch::Tensor x) { return x; }"
)


def prompt(mode="contrastive", turn_index=1):
    messages = build_turn_messages(
        "channel_rmsnorm", mode=mode, turn_index=turn_index, best_code=OWN_BEST, speedup=1.1
    )
    return messages[-1]["content"]


def renamed(code):
    """Rename identifiers and reformat: the copies the tiny model produced."""
    code = re.sub(r"\brmsnorm_warp_kernel\b", "my_norm_kernel", code)
    code = re.sub(r"\bsum_sq\b", "acc", code)
    return "// my own kernel\n" + code.replace("    ", "  ")


def test_examples_exclude_the_models_own_best_code():
    examples = prompt_examples(prompt())
    assert examples, "contrastive prompt should contain teacher examples"
    assert all("void mine" not in block for block in examples)


def test_reference_only_prompt_has_no_examples():
    assert prompt_examples(prompt(mode="reference_only")) == []


def test_renamed_copy_is_detected_but_new_code_is_not():
    assert token_similarity(renamed(TEACHER), TEACHER) >= 0.9
    assert max_example_similarity(renamed(TEACHER), prompt()) >= 0.9
    assert max_example_similarity(OWN_BEST, prompt()) < 0.5


class CountingDispatcher:
    def __init__(self):
        self.calls = 0

    def evaluate(self, request):
        self.calls += 1
        return KernelEvalResponse(
            task_id=request.task_id, compiled=True, correct=True, stable=True,
            status="ok", speedup=4.5,
        )


@pytest.fixture
def engine(tmp_path):
    config = ExperimentConfig(telemetry_path=str(tmp_path / "telemetry.jsonl"))
    dispatcher = CountingDispatcher()
    return RewardEngine(config, dispatcher), dispatcher


def test_copy_gets_zero_reward_without_verification(engine):
    reward_engine, dispatcher = engine
    result = reward_engine.evaluate_code(
        renamed(TEACHER), "channel_rmsnorm", prompt_text=prompt(), prompt_mode="contrastive"
    )
    assert result.status == "copied_example"
    assert result.reward == 0.0 and result.speedup == 0.0
    assert dispatcher.calls == 0


def test_original_code_is_verified_normally(engine):
    reward_engine, dispatcher = engine
    result = reward_engine.evaluate_code(
        OWN_BEST, "channel_rmsnorm", prompt_text=prompt(), prompt_mode="contrastive"
    )
    assert dispatcher.calls == 1
    assert result.status == "ok" and result.reward > 0


def test_gate_can_be_disabled(engine):
    reward_engine, dispatcher = engine
    reward_engine.config.novelty_threshold = None
    reward_engine.evaluate_code(renamed(TEACHER), "channel_rmsnorm", prompt_text=prompt())
    assert dispatcher.calls == 1
