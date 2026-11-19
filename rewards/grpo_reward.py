import json
import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from data.tasks import TASKS
from src.contracts import KernelEvalRequest, KernelEvalResponse, TelemetryRecord

# Reward computation for Group Relative Policy Optimization training.
# Every completion is sent to the process-isolated verifier (src/verifier),
# which runs static checks, compilation, the correctness suite and paired
# timing. The reward is the denoised speedup, or 0.0 on any failure.

logger = logging.getLogger(__name__)

# Global registry tracking failure taxonomy and zero reward group occurrences.
ERROR_TAXONOMY: Dict[str, int] = defaultdict(int)
GROUP_STATS: Dict[str, int] = defaultdict(int)

# Verifier failure_stage -> taxonomy key (kept stable for existing logs).
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


def infer_task_id_from_prompt(prompt: str) -> str:
    prompt_lower = prompt.lower()
    for task_id, data in TASKS.items():
        if data["name"].lower() in prompt_lower or task_id in prompt_lower:
            return task_id
    return "swish"


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
        timestamp=datetime.now(timezone.utc).isoformat(),
        extra={
            "failure_stage": response.failure_stage,
            "execution_time_ms": response.execution_time_ms,
            "baseline_time_ms": response.baseline_time_ms,
            "baseline_bandwidth_gbps": response.baseline_bandwidth_gbps,
            "raw_speedup": (response.extra.get("timing") or {}).get("raw_speedup"),
        },
    )
    os.makedirs(os.path.dirname(_TELEMETRY_PATH) or ".", exist_ok=True)
    with open(_TELEMETRY_PATH, "a") as handle:
        handle.write(json.dumps(record.model_dump()) + "\n")


def evaluate_single_completion(
    completion: str, task_id: str, device_id: Optional[int] = None
) -> float:
    # On a 2-GPU box, train on GPU 0 and set VERIFIER_DEVICE=1 so timing is
    # not disturbed by the policy's own kernels.
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
        ERROR_TAXONOMY[FAILURE_TAXONOMY.get(response.failure_stage, "unknown_failure")] += 1
        logger.debug("task=%s failure=%s: %s", task_id, response.failure_stage,
                     (response.error_message or "")[:200])
    _emit_telemetry(response, reward)
    return reward


def compute_grpo_rewards(
    prompts: Optional[List[str]] = None,
    completions: Optional[List[Any]] = None,
    **kwargs: Any
) -> List[float]:
    # Handle variations in TRL completion argument formats
    if completions is None:
        completions = kwargs.get("candidate_completions", [])

    flat_completions: List[str] = []
    for comp in completions:
        if isinstance(comp, list) and len(comp) > 0 and isinstance(comp[0], dict):
            flat_completions.append(comp[0].get("content", ""))
        elif isinstance(comp, dict):
            flat_completions.append(comp.get("content", ""))
        else:
            flat_completions.append(str(comp))

    num_samples = len(flat_completions)

    # Replicate prompts if TRL provides 1 prompt per batch with G completions
    if prompts is not None and len(prompts) > 0:
        if len(prompts) != num_samples and num_samples % len(prompts) == 0:
            repeat = num_samples // len(prompts)
            prompts = [p for p in prompts for _ in range(repeat)]
        elif len(prompts) != num_samples:
            default_prompt = kwargs.get("prompt", "")
            prompts = [default_prompt] * num_samples
    else:
        fallback_prompt = kwargs.get("prompt", "")
        prompts = [fallback_prompt] * num_samples

    # Check if task_id was passed directly via dataset column
    task_ids: Optional[List[str]] = None
    if "task_id" in kwargs:
        tid_arg = kwargs["task_id"]
        if isinstance(tid_arg, list):
            if len(tid_arg) == num_samples:
                task_ids = tid_arg
            elif len(tid_arg) > 0 and num_samples % len(tid_arg) == 0:
                repeat = num_samples // len(tid_arg)
                task_ids = [t for t in tid_arg for _ in range(repeat)]
        elif isinstance(tid_arg, str):
            task_ids = [tid_arg] * num_samples

    rewards: List[float] = []
    for idx, (prompt, comp) in enumerate(zip(prompts, flat_completions)):
        if task_ids and idx < len(task_ids):
            task_id = task_ids[idx]
        else:
            task_id = infer_task_id_from_prompt(prompt)
        reward = evaluate_single_completion(comp, task_id)
        rewards.append(reward)

    # Detect and record zero reward collapse within groups of size G=2
    if len(rewards) >= 2:
        for i in range(0, len(rewards), 2):
            group = rewards[i:i + 2]
            if len(group) == 2:
                GROUP_STATS["total_groups"] += 1
                if all(r == 0.0 for r in group):
                    GROUP_STATS["zero_reward_groups"] += 1

    return rewards

def get_taxonomy_summary() -> Dict[str, Any]:
    return {
        "error_taxonomy": dict(ERROR_TAXONOMY),
        "group_stats": dict(GROUP_STATS),
    }

def reset_taxonomy_stats() -> None:
    ERROR_TAXONOMY.clear()
    GROUP_STATS.clear()
