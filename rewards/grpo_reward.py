import json
import logging
import os
import uuid
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from data.tasks import TASKS
from engine.utils import now_iso
from src.contracts import KernelEvalRequest, KernelEvalResponse, TelemetryRecord
from src.rewards.reward_engine import RewardEngine

logger = logging.getLogger(__name__)

ERROR_TAXONOMY: Dict[str, int] = defaultdict(int)
GROUP_STATS: Dict[str, int] = defaultdict(int)

FAILURE_TAXONOMY = {
    "static_check": "static_check_failed",
    "compile": "compilation_failed",
    "incorrect": "numerical_divergence",
    "memcheck": "memcheck_failed",
    "runtime_error": "runtime_crash",
    "worker_crash": "runtime_crash",
    "timeout": "timeout",
    "inter_block_variance_exceeded": "timing_unstable",
    "speedup_above_suspicious_threshold": "unverified_large_speedup",
    "cross_gpu_mismatch": "cross_gpu_mismatch",
}

_VERIFIER: Optional[Any] = None
_TELEMETRY_PATH: Optional[str] = os.environ.get("TELEMETRY_PATH")


def configure_verifier(
    verifier: Optional[Any] = None, telemetry_path: Optional[str] = None
) -> None:
    """Install the verifier (anything with `evaluate(request)`) and telemetry sink."""
    global _VERIFIER, _TELEMETRY_PATH
    _VERIFIER = verifier
    if telemetry_path is not None:
        _TELEMETRY_PATH = telemetry_path


def get_verifier() -> Any:
    global _VERIFIER
    if _VERIFIER is None:
        from src.verifier import KernelVerifier, VerifierConfig

        _VERIFIER = KernelVerifier(VerifierConfig.training())
    return _VERIFIER


def _emit_telemetry(response: KernelEvalResponse, reward: float) -> None:
    if not _TELEMETRY_PATH:
        return
    record = TelemetryRecord(
        trajectory_id=str(uuid.uuid4()),
        task_id=response.task_id,
        turn_index=0,
        speedup=response.speedup,
        reward=reward,
        compiled=response.compiled,
        correct=response.correct,
        memory_bandwidth_gbps=response.memory_bandwidth_gbps,
        timestamp=now_iso(),
        extra={
            "failure_stage": response.failure_stage,
            "execution_time_ms": response.execution_time_ms,
            "baseline_time_ms": response.baseline_time_ms,
            "baseline_bandwidth_gbps": response.baseline_bandwidth_gbps,
            "raw_speedup": (response.extra.get("timing") or {}).get("raw_speedup"),
        },
    )
    os.makedirs(os.path.dirname(_TELEMETRY_PATH) or ".", exist_ok=True)
    payload = record.model_dump() if hasattr(record, "model_dump") else record.dict()
    with open(_TELEMETRY_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def infer_task_id_from_prompt(prompt: str) -> str:
    prompt_lower = prompt.lower()
    for task_id, data in TASKS.items():
        if data["name"].lower() in prompt_lower or task_id in prompt_lower:
            return task_id
    return "swish"


def _taxonomy_for_status(status: str) -> str:
    if status == "static_failed":
        return "static_check_failed"
    if status in {"compile_error", "runtime_unavailable"}:
        return "compilation_failed"
    if status in {"incorrect", "cache_or_incorrect"}:
        return "numerical_divergence"
    if status == "runtime_error":
        return "runtime_crash"
    if status in {
        "unstable",
        "timeout",
        "fake_gain",
        "implausible_speedup",
        "measurement_too_small",
    }:
        return "timing_unstable"
    if status == "sanitizer_error":
        return "memory_safety"
    if status == "copied_example":
        return "copied_example"
    if status == "ok":
        return "success"
    return "unknown"


def evaluate_single_completion(
    completion: str,
    task_id: str,
    config: Any = None,
    dispatcher: Any = None,
    device_id: Optional[int] = None,
) -> float:
    if dispatcher is not None and config is not None:
        engine = RewardEngine(config, dispatcher)
        result = engine.evaluate_code(completion, task_id)
        ERROR_TAXONOMY[_taxonomy_for_status(result.status)] += 1
        return float(result.reward)

    if device_id is None:
        device_id = int(os.environ.get("VERIFIER_DEVICE", "0"))
    response = get_verifier().evaluate(
        KernelEvalRequest(task_id=task_id, code=completion, device_id=device_id)
    )
    if response.failure_stage is None and response.speedup:
        reward = float(response.speedup)
        ERROR_TAXONOMY["success"] += 1
    else:
        reward = 0.0
        ERROR_TAXONOMY[FAILURE_TAXONOMY.get(response.failure_stage or "", "unknown_failure")] += 1
    _emit_telemetry(response, reward)
    return reward


def _flatten_completions(completions: List[Any]) -> List[str]:
    flat: List[str] = []
    for completion in completions:
        if isinstance(completion, list) and completion and isinstance(completion[0], dict):
            flat.append(completion[0].get("content", ""))
        elif isinstance(completion, dict):
            flat.append(completion.get("content", ""))
        else:
            flat.append(str(completion))
    return flat


def _resolve_task_ids(
    prompts: List[str],
    num_samples: int,
    task_id_arg: Any,
    fallback_prompt: str,
) -> List[str]:
    task_ids: Optional[List[str]] = None
    if isinstance(task_id_arg, list):
        if len(task_id_arg) == num_samples:
            task_ids = [str(value) for value in task_id_arg]
        elif task_id_arg and num_samples % len(task_id_arg) == 0:
            repeat = num_samples // len(task_id_arg)
            task_ids = [str(value) for value in task_id_arg for _ in range(repeat)]
    elif isinstance(task_id_arg, str):
        task_ids = [task_id_arg] * num_samples

    if task_ids is None:
        task_ids = []
        for prompt in prompts:
            inferred = infer_task_id_from_prompt(prompt or fallback_prompt)
            task_ids.append(inferred)
    return task_ids


def compute_grpo_rewards(
    prompts: Optional[List[str]] = None,
    completions: Optional[List[Any]] = None,
    **kwargs: Any,
) -> List[float]:
    config = kwargs.pop("config", None)
    dispatcher = kwargs.pop("dispatcher", None)
    if completions is None:
        completions = kwargs.get("candidate_completions", [])

    flat_completions = _flatten_completions(completions)
    num_samples = len(flat_completions)

    if prompts is not None and len(prompts) > 0:
        if len(prompts) != num_samples and num_samples % len(prompts) == 0:
            repeat = num_samples // len(prompts)
            prompts = [prompt for prompt in prompts for _ in range(repeat)]
        elif len(prompts) != num_samples:
            prompts = [kwargs.get("prompt", "")] * num_samples
    else:
        prompts = [kwargs.get("prompt", "")] * num_samples

    task_ids = _resolve_task_ids(
        prompts, num_samples, kwargs.get("task_id"), kwargs.get("prompt", "")
    )

    rewards: List[float] = []
    for prompt, completion, task_id in zip(prompts, flat_completions, task_ids):
        reward = evaluate_single_completion(
            completion,
            task_id,
            config=config,
            dispatcher=dispatcher,
        )
        rewards.append(reward)

    group_size = int(kwargs.get("group_size", 2))
    if group_size > 1:
        for index in range(0, len(rewards), group_size):
            group = rewards[index : index + group_size]
            if len(group) == group_size:
                GROUP_STATS["total_groups"] += 1
                if all(reward == 0.0 for reward in group):
                    GROUP_STATS["zero_reward_groups"] += 1

    return rewards


def make_reward_function(config: Any, dispatcher: Any) -> Callable[..., List[float]]:
    def kernel_reward(prompts=None, completions=None, **kwargs):
        kwargs["config"] = config
        kwargs["dispatcher"] = dispatcher
        kwargs.setdefault("group_size", config.effective_group_size())
        return compute_grpo_rewards(prompts=prompts, completions=completions, **kwargs)

    kernel_reward.__name__ = "kernel_reward"
    return kernel_reward


def get_taxonomy_summary() -> Dict[str, Any]:
    return {
        "error_taxonomy": dict(ERROR_TAXONOMY),
        "group_stats": dict(GROUP_STATS),
    }


def reset_taxonomy_stats() -> None:
    ERROR_TAXONOMY.clear()
    GROUP_STATS.clear()
