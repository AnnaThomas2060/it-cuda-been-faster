"""Unit tests for the multi-turn RL pipeline.

These tests run on CPU only. They cover configuration handling, reward shaping,
archive bookkeeping, loss math, rollout sample construction, teacher storage,
and evaluation statistics without requiring a CUDA device.
"""

import json
import math
from pathlib import Path

import pytest
import torch

from configs.grpo_config import ExperimentConfig
from data.example_store import get_teacher_examples
from engine.prompt_builder import build_turn_messages, format_chat_prompt
from src.contracts import KernelEvalRequest, KernelEvalResponse, MockVerifier
from src.pipeline.env_dispatcher import MockDispatcher
from src.pipeline.rollout_manager import assign_group_advantages, build_training_samples
from src.pipeline.state import Trajectory, TurnRecord, build_feedback
from src.rewards.anti_hack import is_fake_gain, static_audit, timer_is_trustworthy
from src.rewards.archive import ContrastiveArchive
from src.rewards.reward_engine import (
    assign_discounted_returns,
    compute_shaped_reward,
    group_normalize,
)
from src.trainers.losses import (
    dpo_loss,
    gather_completion_logprobs,
    grpo_loss,
    masked_sequence_logprob,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _response(**overrides) -> KernelEvalResponse:
    payload = {
        "task_id": "swish",
        "compiled": True,
        "correct": True,
        "stable": True,
        "status": "ok",
        "speedup": 1.4,
        "bandwidth_delta": 0.2,
        "variance": 0.001,
    }
    payload.update(overrides)
    return KernelEvalResponse(**payload)


def _config(**overrides) -> ExperimentConfig:
    config = ExperimentConfig()
    config.archive_path = str(REPO_ROOT / "outputs" / "test_archive.jsonl")
    config.telemetry_path = str(REPO_ROOT / "outputs" / "test_telemetry.jsonl")
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_yaml_configs_load_into_experiment_config() -> None:
    multiturn = ExperimentConfig.from_yaml(str(REPO_ROOT / "configs" / "multiturn_grpo.yaml"))
    contrastive = ExperimentConfig.from_yaml(str(REPO_ROOT / "configs" / "contrastive_rl.yaml"))

    assert multiturn.mode == "multi_turn"
    assert multiturn.group_size == 16
    assert multiturn.gamma == pytest.approx(0.4)
    assert multiturn.tasks_per_step == 8

    assert contrastive.mode == "contrastive"
    assert contrastive.enable_cross_trajectory_anchors is True
    assert contrastive.pairwise_margin_weight == pytest.approx(0.20)

    multiturn.validate()
    contrastive.validate()


def test_config_aliases_and_unknown_keys() -> None:
    config = ExperimentConfig.from_dict(
        {
            "gamma_discount": 0.5,
            "bw_bonus_weight": 0.25,
            "clip_eps_low": 0.1,
            "clip_eps_high": 0.3,
            "kl_coeff": 0.05,
            "batch_tasks": 2,
            "train_turns": 3,
        }
    )
    assert config.gamma == pytest.approx(0.5)
    assert config.bandwidth_weight == pytest.approx(0.25)
    assert config.epsilon == pytest.approx(0.1)
    assert config.epsilon_high == pytest.approx(0.3)
    assert config.beta == pytest.approx(0.05)
    assert config.tasks_per_step == 2
    assert config.num_turns == 3

    with pytest.raises(ValueError, match="Unknown config keys"):
        ExperimentConfig.from_dict({"not_a_real_key": 1})


def test_config_validation_rejects_invalid_values() -> None:
    config = ExperimentConfig()
    config.mode = "invalid"
    with pytest.raises(ValueError, match="Unsupported mode"):
        config.validate()

    config = ExperimentConfig()
    config.gamma = 2.0
    with pytest.raises(ValueError, match="gamma"):
        config.validate()

    config = ExperimentConfig()
    config.bf16 = True
    config.fp16 = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        config.validate()

    config = ExperimentConfig()
    config.eval_tasks = ["missing_task"]
    with pytest.raises(ValueError, match="unregistered tasks"):
        config.validate()

    config = ExperimentConfig()
    config.max_completion_length = 8
    with pytest.raises(ValueError, match="too small"):
        config.validate()


def test_config_arm_helpers() -> None:
    config = ExperimentConfig()
    assert config.prompt_mode_for_arm("arm_a") == "contrastive"
    assert config.prompt_mode_for_arm("arm_b") == "reference_only"
    assert config.output_dir_for_arm("arm_a") == config.output_dir_arm_a
    assert config.output_dir_for_arm("arm_b") == config.output_dir_arm_b
    with pytest.raises(ValueError):
        config.prompt_mode_for_arm("arm_c")

    config.group_size = 0
    config.num_generations = 6
    assert config.effective_group_size() == 6

    config.archive_path = "./outputs/contrastive_archive.jsonl"
    assert config.archive_path_for_arm("arm_a").endswith("contrastive_archive_arm_a.jsonl")
    assert config.archive_path_for_arm("arm_b").endswith("contrastive_archive_arm_b.jsonl")
    with pytest.raises(ValueError):
        config.archive_path_for_arm("arm_c")


def test_prompt_turns_include_state_and_teacher_context() -> None:
    first_turn = build_turn_messages("swish", mode="reference_only", turn_index=0)
    assert first_turn[0]["role"] == "system"
    assert "torch::Tensor run_cuda" in first_turn[1]["content"]
    assert "Multi-turn refinement state" not in first_turn[1]["content"]

    second_turn = build_turn_messages(
        "swish",
        mode="reference_only",
        turn_index=1,
        best_code="__global__ void kernel() {}",
        feedback="Compilation failed.",
        speedup=1.1,
        history=[{"turn_index": 0, "status": "compile_error", "feedback": "bad"}],
    )
    content = second_turn[1]["content"]
    assert "Multi-turn refinement state (turn 2)" in content
    assert "Current best implementation" in content
    assert "Compilation failed." in content
    assert "1.10x" in content
    assert "Previous turn summaries" in content

    holdout_turn = build_turn_messages("softmax", mode="contrastive", turn_index=0)
    assert "Historical Implementations and Measured Speedups" in holdout_turn[1]["content"]

    plain = format_chat_prompt(_FallbackTokenizer(), first_turn)
    assert plain.startswith("SYSTEM:")
    assert plain.endswith("ASSISTANT:")


class _FallbackTokenizer:
    chat_template = None


def test_shaped_reward_filters_and_clips() -> None:
    config = ExperimentConfig()
    kwargs = {
        "reward_mode": "shaped",
        "correctness_bonus": config.correctness_bonus,
        "bandwidth_weight": config.bandwidth_weight,
        "reward_clip": config.reward_clip,
    }
    assert compute_shaped_reward(_response(speedup=1.5), **kwargs) > config.correctness_bonus
    assert compute_shaped_reward(_response(correct=False), **kwargs) == 0.0
    assert compute_shaped_reward(_response(stable=False), **kwargs) == 0.0
    assert compute_shaped_reward(_response(status="fake_gain"), **kwargs) == 0.0
    assert compute_shaped_reward(_response(speedup=1.0), **kwargs) == pytest.approx(
        config.correctness_bonus + config.bandwidth_weight * 0.2
    )

    speedup_reward = compute_shaped_reward(
        _response(speedup=4.0),
        reward_mode="speedup",
        correctness_bonus=0.3,
        bandwidth_weight=0.1,
        reward_clip=2.0,
    )
    assert speedup_reward == pytest.approx(2.0)


def test_returns_and_group_normalization() -> None:
    returns = assign_discounted_returns([1.0, 2.0, 3.0], gamma=0.5)
    assert returns == pytest.approx([2.75, 3.5, 3.0])

    normalized = group_normalize([1.0, 2.0, 3.0])
    assert sum(normalized) == pytest.approx(0.0, abs=1e-6)
    assert group_normalize([2.0, 2.0]) == [0.0, 0.0]


def test_anti_hack_checks() -> None:
    valid = """
    __global__ void kernel(float* x) {}
    torch::Tensor run_cuda(torch::Tensor x) { return x; }
    """
    assert static_audit(valid)[0] is True

    shortcut = """
    __global__ void kernel(float* x) {}
    torch::Tensor run_cuda(torch::Tensor x) { return x.relu(); }
    """
    assert static_audit(shortcut)[0] is False

    assert timer_is_trustworthy(0.001, 0.5, 0.005, 0.005) == (True, "ok")
    assert timer_is_trustworthy(0.001, 0.001, 0.005, 0.005)[1] == "measurement_too_small"
    assert timer_is_trustworthy(0.01, 0.5, 0.005, 0.005)[1] == "unstable_timing"
    assert timer_is_trustworthy(0.0, 0.0, 0.005, 0.005)[1] == "non_positive_timing"

    assert is_fake_gain(_response(speedup=20.0), 1000.0, 1.2, 10.0)[1] == "implausible_speedup"
    bandwidth = _response(speedup=1.2, memory_bandwidth_gbps=5000.0)
    assert is_fake_gain(bandwidth, 1000.0, 1.2, 10.0)[1] == "bandwidth_exceeds_peak"
    assert is_fake_gain(_response(speedup=1.2), 1000.0, 1.2, 10.0) == (False, "ok")


def test_reward_engine_with_mock_dispatcher(tmp_path: Path) -> None:
    from src.rewards.reward_engine import RewardEngine

    config = _config()
    config.telemetry_path = str(tmp_path / "telemetry.jsonl")
    engine = RewardEngine(config, MockDispatcher(simulated_speedup=1.5))
    result = engine.evaluate_code("```cpp\n__global__ void k() {}\n```", "swish")

    assert result.status == "ok"
    assert result.reward > 0.0
    assert result.response.compiled is True
    assert result.response.speedup == pytest.approx(1.5)
    assert (tmp_path / "telemetry.jsonl").is_file()

    record = json.loads((tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["task_id"] == "swish"
    assert record["correct"] is True


def test_archive_dedupes_ranks_and_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "archive.jsonl"
    archive = ContrastiveArchive(str(path), capacity_per_task=8, min_speedup_margin_delta=0.1)

    archive.add("swish", "kernel-a", speedup=1.5, correct=True, status="ok")
    archive.add("swish", "kernel-a", speedup=1.5, correct=True, status="ok")
    archive.add("swish", "kernel-b", speedup=1.1, correct=True, status="ok")
    archive.add("swish", "kernel-c", speedup=0.0, correct=False, status="compile_error")

    assert len(archive.entries["swish"]) == 3
    assert archive.winners("swish")[0].code == "kernel-a"
    assert archive.losers("swish")[0].code == "kernel-c"

    pair = archive.sample_pair("swish")
    assert pair is not None
    assert pair[0].code != pair[1].code
    assert pair[0].correct is True

    pairs = archive.all_pairs(max_pairs=2, seed=0)
    assert len(pairs) == 1
    assert pairs[0][0] == "swish"

    archive.save()
    reloaded = ContrastiveArchive(str(path), capacity_per_task=8)
    assert len(reloaded.entries["swish"]) == 3
    assert reloaded.summary()["swish"]["count"] == 3


def test_gather_completion_logprobs_matches_manual_math() -> None:
    torch.manual_seed(0)
    batch_size, sequence_length, vocab = 2, 8, 11
    logits = torch.randn(batch_size, sequence_length, vocab)
    input_ids = torch.randint(0, vocab, (batch_size, sequence_length))
    prompt_lengths = [3, 4]
    completion_lengths = [2, 1]

    token_logprobs, mask = gather_completion_logprobs(
        logits,
        input_ids,
        prompt_lengths,
        completion_lengths,
    )

    assert token_logprobs.shape == (batch_size, 2)
    assert mask.shape == (batch_size, 2)
    assert mask[0].tolist() == [1.0, 1.0]
    assert mask[1].tolist() == [1.0, 0.0]

    reference = torch.log_softmax(logits.float(), dim=-1)
    expected = reference[0, 2, int(input_ids[0, 3])]
    assert token_logprobs[0, 0] == pytest.approx(float(expected), abs=1e-6)
    expected_last = reference[1, 3, int(input_ids[1, 4])]
    assert token_logprobs[1, 0] == pytest.approx(float(expected_last), abs=1e-6)

    sequence = masked_sequence_logprob(
        logits,
        input_ids,
        prompt_lengths,
        completion_lengths,
    )
    assert sequence.shape == (batch_size,)
    assert sequence[0] == pytest.approx(float(token_logprobs[0, 0] + token_logprobs[0, 1]))


def test_gather_logprobs_respects_logit_offset() -> None:
    torch.manual_seed(1)
    vocab = 7
    full_logits = torch.randn(1, 6, vocab)
    input_ids = torch.randint(0, vocab, (1, 6))
    keep_last = 4
    trimmed = full_logits[:, -keep_last:, :]

    full_logprobs, _ = gather_completion_logprobs(full_logits, input_ids, [3], [3])
    trimmed_logprobs, _ = gather_completion_logprobs(
        trimmed,
        input_ids,
        [3],
        [3],
        logit_offset=2,
    )
    assert torch.allclose(full_logprobs, trimmed_logprobs, atol=1e-6)


def test_grpo_and_dpo_losses() -> None:
    new_logprobs = torch.zeros(2, 3)
    old_logprobs = torch.zeros(2, 3)
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.ones(2, 3)

    loss = grpo_loss(
        new_logprobs,
        old_logprobs,
        advantages,
        mask,
        loss_type="dr_grpo",
        l_norm=6.0,
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-6)

    shifted = new_logprobs.clone()
    shifted[0, :] = math.log(1.5)
    loss_with_ratio = grpo_loss(
        shifted,
        old_logprobs,
        advantages,
        mask,
        epsilon_low=0.2,
        epsilon_high=0.28,
        l_norm=6.0,
    )
    capped_ratio = 1.28
    expected = (-capped_ratio * 3 + 1.0 * 3) / 6.0
    assert float(loss_with_ratio) == pytest.approx(expected, abs=1e-6)

    better = dpo_loss(
        torch.tensor(-1.0),
        torch.tensor(-3.0),
        torch.tensor(-1.2),
        torch.tensor(-2.8),
    )
    worse = dpo_loss(
        torch.tensor(-3.0),
        torch.tensor(-1.0),
        torch.tensor(-1.2),
        torch.tensor(-2.8),
    )
    assert float(better) < float(worse)


def _make_trajectory(task_id: str = "swish") -> Trajectory:
    trajectory = Trajectory(trajectory_id="traj", task_id=task_id, prompt_mode="reference_only")
    trajectory.add_step(
        TurnRecord(
            turn_index=0,
            prompt="prompt",
            prompt_ids=[1, 2],
            completion_ids=[3, 4],
            completion_text="```cpp\ncode\n```",
            old_logprobs=[-0.1, -0.2],
            response=_response(speedup=1.0),
            reward=0.5,
            code="code",
            status="ok",
            speedup=1.0,
            feedback="ok",
        )
    )
    trajectory.add_step(
        TurnRecord(
            turn_index=1,
            prompt="prompt",
            prompt_ids=[1, 2, 5],
            completion_ids=[6],
            completion_text="```cpp\nbetter\n```",
            old_logprobs=[-0.3],
            response=_response(speedup=1.6),
            reward=0.9,
            code="better",
            status="ok",
            speedup=1.6,
            feedback="ok",
        )
    )
    return trajectory


def test_trajectory_best_step_and_returns() -> None:
    trajectory = _make_trajectory()
    assert trajectory.best_code == "better"
    assert trajectory.best_speedup == pytest.approx(1.6)
    assert trajectory.to_record().steps[0].turn_index == 0

    trajectory.compute_returns(gamma=0.5)
    assert trajectory.steps[1].discounted_return == pytest.approx(0.9)
    assert trajectory.steps[0].discounted_return == pytest.approx(0.5 + 0.5 * 0.9)
    assert trajectory.final_reward == pytest.approx(0.95)

    assert "Measured speedup 1.60x" in build_feedback(_response(speedup=1.6))
    assert "Compilation failed" in build_feedback(_response(compiled=False, status="compile_error"))


def test_training_sample_construction_and_advantages() -> None:
    first = _make_trajectory()
    second = _make_trajectory()
    second.steps[0].discounted_return = 2.0
    second.steps[1].discounted_return = 2.0
    first.steps[0].discounted_return = 0.0
    first.steps[1].discounted_return = 0.0

    samples = build_training_samples([first, second])
    assert len(samples) == 4
    assert samples[0]["input_ids"] == [1, 2, 3, 4]
    assert samples[0]["prompt_length"] == 2
    assert samples[0]["completion_length"] == 2

    assign_group_advantages([first, second])
    grouped = [first.steps[0].advantage, second.steps[0].advantage]
    assert sum(grouped) == pytest.approx(0.0, abs=1e-6)
    assert first.steps[0].advantage < second.steps[0].advantage


def test_training_samples_skip_mismatched_logprobs() -> None:
    trajectory = _make_trajectory()
    trajectory.steps[0].old_logprobs = []
    samples = build_training_samples([trajectory])
    assert len(samples) == 1
    assert samples[0]["turn_index"] == 1


def test_holdout_teachers_and_measurement_override(monkeypatch, tmp_path: Path) -> None:
    for task_id in ("average_pooling_2d", "softmax", "cumsum"):
        teachers = get_teacher_examples(task_id)
        assert teachers, f"Missing teacher example for {task_id}"
        assert teachers[0]["code"].strip()

    override = tmp_path / "measurements.json"
    override.write_text(
        json.dumps({"softmax": [{"name": "block_reduced_softmax", "speedup": 1.73}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TEACHER_MEASUREMENTS_PATH", str(override))
    softmax_teacher = get_teacher_examples("softmax")[0]
    assert softmax_teacher["speedup"] == pytest.approx(1.73)
    assert softmax_teacher["verified"] is True


def test_mock_dispatcher_contract() -> None:
    dispatcher = MockDispatcher(simulated_speedup=1.25)
    request = KernelEvalRequest(task_id="swish", code="__global__ void k() {}")
    response = dispatcher.evaluate(request)
    assert isinstance(response, KernelEvalResponse)
    assert response.status == "ok"
    assert response.speedup == pytest.approx(1.25)
    assert response.stable is True

    verifier = MockVerifier(simulated_speedup=1.1)
    assert verifier.evaluate(request).bandwidth_delta == pytest.approx(0.11)


def test_evaluation_statistics() -> None:
    from evaluate import _summarize_records, bootstrap_median_ci

    lower, upper = bootstrap_median_ci([1.0, 1.2, 1.4, 1.6, 1.8], samples=200, seed=0)
    assert 1.0 <= lower <= upper <= 1.8

    records = [
        {"task_id": "swish", "numerically_correct": True, "is_stable": True, "speedup": 1.2},
        {"task_id": "swish", "numerically_correct": True, "is_stable": True, "speedup": 1.6},
        {"task_id": "softmax", "numerically_correct": False, "is_stable": False, "speedup": 0.0},
    ]
    summary = _summarize_records(records, ["swish", "softmax"])
    assert summary["total_tasks"] == 2
    assert summary["correct_count"] == 1
    assert summary["correctness_rate"] == pytest.approx(0.5)
    assert summary["median_speedup"] == pytest.approx(1.6)
    assert summary["best_speedup"] == pytest.approx(1.6)


def test_compare_arms_reports_paired_differences() -> None:
    from scripts.compare_arms import compare

    arm_a = {
        "task_best": [
            {"task_id": "swish", "speedup": 1.5, "numerically_correct": True},
            {"task_id": "softmax", "speedup": 1.1, "numerically_correct": True},
        ]
    }
    arm_b = {
        "task_best": [
            {"task_id": "swish", "speedup": 1.2, "numerically_correct": True},
            {"task_id": "softmax", "speedup": 1.3, "numerically_correct": False},
        ]
    }
    result = compare(arm_a, arm_b, bootstrap_samples=100, seed=0)

    assert result["num_tasks"] == 2
    assert result["median_difference"] == pytest.approx(0.05)
    assert result["wins_a"] == 1
    assert result["wins_b"] == 1
    assert result["arm_a_correctness_rate"] == pytest.approx(1.0)
    assert result["arm_b_correctness_rate"] == pytest.approx(0.5)
    assert result["per_task"][0]["task_id"] == "softmax"


class _FakeTokenizer:
    chat_template = None
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 4
    padding_side = "right"

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=None,
        truncation=None,
        max_length=None,
        add_special_tokens=None,
    ):
        rows = [[1, 2, 3] for _ in texts]
        input_ids = torch.tensor(rows, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def decode(self, token_ids, skip_special_tokens=True):
        return "```cpp\n__global__ void k() {}\n```"

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


class _FakeLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 8)
        self.head = torch.nn.Linear(8, vocab_size)
        self.config = type("Config", (), {"use_cache": False})()
        self.generation_config = type("GenerationConfig", (), {"use_cache": True})()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        use_cache=False,
        num_logits_to_keep=None,
        **kwargs,
    ):
        hidden = self.embed(input_ids)
        logits = self.head(hidden)
        if num_logits_to_keep is not None:
            logits = logits[:, -num_logits_to_keep:, :]
        return type("Output", (), {"logits": logits})()

    def generate(self, input_ids, attention_mask=None, max_new_tokens=8, **kwargs):
        batch = input_ids.shape[0]
        generated = torch.full(
            (batch, max_new_tokens),
            3,
            dtype=torch.long,
            device=input_ids.device,
        )
        generated[:, -1] = 4
        return torch.cat([input_ids, generated], dim=1)

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


def test_cpu_multi_turn_training_step(tmp_path: Path) -> None:
    from src.pipeline.rollout_manager import RolloutManager
    from src.rewards.reward_engine import RewardEngine
    from src.trainers.multiturn_grpo import MultiTurnGRPOTrainer

    config = ExperimentConfig()
    config.mode = "multi_turn"
    config.bf16 = False
    config.fp16 = False
    config.max_steps = 1
    config.num_turns = 1
    config.group_size = 2
    config.num_generations = 2
    config.tasks_per_step = 1
    config.per_device_train_batch_size = 1
    config.gradient_accumulation_steps = 1
    config.max_prompt_length = 32
    config.max_completion_length = 16
    config.training_tasks = ["swish"]
    config.eval_tasks = ["swish"]
    config.save_steps = 0
    config.early_stop_speedup = 99.0
    config.telemetry_path = str(tmp_path / "telemetry.jsonl")
    config.archive_path = str(tmp_path / "archive.jsonl")
    config.validate()

    model = _FakeLM()
    tokenizer = _FakeTokenizer()
    reward_engine = RewardEngine(config, MockDispatcher(simulated_speedup=1.3))
    rollout_manager = RolloutManager(model, tokenizer, reward_engine, config)
    trainer = MultiTurnGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        rollout_manager=rollout_manager,
        config=config,
        output_dir=str(tmp_path / "run"),
        prompt_mode="reference_only",
    )

    metrics = trainer.train()
    assert len(metrics) == 1
    assert metrics[0].num_samples == 2
    assert metrics[0].mean_reward > 0.0
    assert metrics[0].mean_speedup == pytest.approx(1.3)
    assert (tmp_path / "run" / "train_metrics.jsonl").is_file()
    assert (tmp_path / "run" / "checkpoint-1" / "config.yaml").is_file()
    assert trainer.global_step == 1


def test_notebook_is_valid_json_and_uses_multiturn_entrypoint() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "colab_rl_pipeline.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert isinstance(notebook["cells"], list)
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        assert isinstance(cell["source"], list)

    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "run_training" in source
    assert "multi_turn" in source
    assert "sft_adapter_path" in source


def test_repository_text_has_no_dash_or_emoji_violations() -> None:
    forbidden = {"\u2014": "em dash", "\u2013": "en dash"}
    suffixes = {".py", ".md", ".yaml", ".yml", ".ipynb", ".txt"}
    offenders = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for character, name in forbidden.items():
            if character in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains an {name}")
        for character in text:
            code_point = ord(character)
            if 0x1F300 <= code_point <= 0x1FAFF:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains an emoji")
                break
    assert not offenders, "\n".join(offenders)
