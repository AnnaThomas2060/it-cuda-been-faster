import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from data.tasks import TASKS
from engine.benchmark import benchmark_kernel
from engine.compiler import compile_kernel_ex
from engine.verifier import verify_kernel
from src.rewards.reward_engine import RewardEngine

logger = logging.getLogger(__name__)

ERROR_TAXONOMY: Dict[str, int] = defaultdict(int)
GROUP_STATS: Dict[str, int] = defaultdict(int)


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
    if status == "ok":
        return "success"
    return "unknown"


def _evaluate_direct(completion: str, task_id: str, config: Any = None) -> float:
    compiled = compile_kernel_ex(completion)
    if compiled.module is None:
        ERROR_TAXONOMY[_taxonomy_for_status(compiled.status)] += 1
        return 0.0

    verification = verify_kernel(
        compiled.module,
        task_id,
        device=getattr(config, "verifier_device", "cuda:0") if config else "cuda:0",
        seeds=getattr(config, "verification_seeds", None) if config else None,
        rtol=getattr(config, "numerical_rtol", None) if config else None,
        atol=getattr(config, "numerical_atol", None) if config else None,
    )
    if not verification.correct:
        ERROR_TAXONOMY[_taxonomy_for_status(verification.status)] += 1
        return 0.0

    benchmark = benchmark_kernel(
        compiled.module,
        task_id,
        device=getattr(config, "verifier_device", "cuda:0") if config else "cuda:0",
        warmup_iters=getattr(config, "benchmark_warmup_iters", 10) if config else 10,
        num_blocks=getattr(config, "benchmark_num_blocks", 7) if config else 7,
        runs_per_block=getattr(config, "benchmark_runs_per_block", 10) if config else 10,
        variance_threshold=getattr(config, "variance_threshold", 0.005) if config else 0.005,
        min_measurement_ms=getattr(config, "min_measurement_ms", 0.005) if config else 0.005,
        use_bandwidth=getattr(config, "use_bandwidth", True) if config else True,
        fake_gain_ratio=getattr(config, "fake_gain_ratio", 1.20) if config else 1.20,
        max_plausible_speedup=getattr(config, "max_plausible_speedup", 10.0) if config else 10.0,
    )
    if benchmark.status != "ok" or not benchmark.stable:
        ERROR_TAXONOMY[_taxonomy_for_status(benchmark.status)] += 1
        return 0.0

    ERROR_TAXONOMY["success"] += 1
    return float(benchmark.speedup)


def evaluate_single_completion(
    completion: str,
    task_id: str,
    cross_gpu_validation: bool = False,
    config: Any = None,
    dispatcher: Any = None,
) -> float:
    if dispatcher is not None and config is not None:
        engine = RewardEngine(config, dispatcher)
        result = engine.evaluate_code(completion, task_id)
        ERROR_TAXONOMY[_taxonomy_for_status(result.status)] += 1
        return float(result.reward)
    return _evaluate_direct(completion, task_id, config=config)


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
