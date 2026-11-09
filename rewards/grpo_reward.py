import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from data.tasks import TASKS
from engine.benchmark import measure_speedup
from engine.compiler import compile_kernel
from engine.verifier import verify_numerical_correctness

# Reward computation engine for Group Relative Policy Optimization training.
# Executes two stage verification covering compilation, numerical equivalence, and latency.
# Detailed taxonomy logs record contract failures and identify zero reward groups.

logger = logging.getLogger(__name__)

# Global registry tracking failure taxonomy and zero reward group occurrences.
ERROR_TAXONOMY: Dict[str, int] = defaultdict(int)
GROUP_STATS: Dict[str, int] = defaultdict(int)

def infer_task_id_from_prompt(prompt: str) -> str:
    prompt_lower = prompt.lower()
    for task_id, data in TASKS.items():
        if data["name"].lower() in prompt_lower or task_id in prompt_lower:
            return task_id
    return "swish"

def evaluate_single_completion(
    completion: str,
    task_id: str,
    cross_gpu_validation: bool = True
) -> float:
    # Stage 1: Static inspection and compilation
    module, compile_status = compile_kernel(completion)
    if module is None:
        is_static = any(
            kw in compile_status for kw in ("Static inspection", "Missing", "Forbidden")
        )
        if is_static:
            ERROR_TAXONOMY["static_check_failed"] += 1
        else:
            ERROR_TAXONOMY["compilation_failed"] += 1
        return 0.0

    # Stage 2: Numerical validation
    passed_num, num_status = verify_numerical_correctness(module, task_id)
    if not passed_num:
        if "Runtime error" in num_status:
            ERROR_TAXONOMY["runtime_crash"] += 1
        else:
            ERROR_TAXONOMY["numerical_divergence"] += 1
        return 0.0

    # Stage 3: Latency measurement and stability gate
    speedup, is_stable = measure_speedup(module, task_id, device="cuda")
    if not is_stable or speedup <= 0.0:
        ERROR_TAXONOMY["timing_unstable"] += 1
        return 0.0

    # Cross-GPU validation gate if multiple GPUs are detected
    if cross_gpu_validation:
        try:
            import torch
            if torch.cuda.device_count() > 1 and speedup > 3.0:
                speedup_gpu1, stable_gpu1 = measure_speedup(module, task_id, device="cuda:1")
                if not stable_gpu1 or abs(speedup - speedup_gpu1) / speedup > 0.10:
                    ERROR_TAXONOMY["cross_gpu_mismatch"] += 1
                    return 0.0
        except Exception:
            pass

    ERROR_TAXONOMY["success"] += 1
    return float(speedup)

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
