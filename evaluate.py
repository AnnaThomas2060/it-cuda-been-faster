import argparse
import json
import os
import random
import statistics
import time
from typing import Any, Dict, List, Optional

from configs.grpo_config import ExperimentConfig
from engine.compiler import extract_code_block
from engine.prompt_builder import build_turn_messages, format_chat_prompt
from engine.utils import environment_metadata, set_seed
from models.loader import load_model_and_tokenizer
from src.pipeline.env_dispatcher import build_dispatcher
from src.pipeline.rollout_manager import RolloutManager
from src.rewards.reward_engine import RewardEngine


def bootstrap_median_ci(values: List[float], samples: int = 1000, seed: int = 0) -> List[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    medians = []
    for _ in range(samples):
        resample = [rng.choice(values) for _ in values]
        medians.append(statistics.median(resample))
    medians.sort()
    lower = medians[int(0.025 * len(medians))]
    upper = medians[int(0.975 * len(medians)) - 1]
    return [round(lower, 4), round(upper, 4)]


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _single_turn_attempt(
    model: Any,
    tokenizer: Any,
    task_id: str,
    prompt_mode: str,
    reward_engine: RewardEngine,
    config: ExperimentConfig,
    do_sample: bool,
    temperature: Optional[float],
) -> Dict[str, Any]:
    import torch

    messages = build_turn_messages(task_id, mode=prompt_mode)
    prompt_text = format_chat_prompt(tokenizer, messages)
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(_model_device(model))
    attention_mask = encoded["attention_mask"].to(_model_device(model))
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": config.max_completion_length,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature or config.temperature
    start_time = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)
    generation_time = time.perf_counter() - start_time
    generated = output_ids[0][input_ids.shape[1] :]
    completion_text = tokenizer.decode(generated, skip_special_tokens=True)
    code = extract_code_block(completion_text)
    result = reward_engine.evaluate_code(
        code,
        task_id,
        prompt_mode=prompt_mode,
        raw_completion=completion_text,
    )
    response = result.response
    return {
        "task_id": task_id,
        "mode": "single_turn",
        "compiled": response.compiled,
        "numerically_correct": response.correct,
        "is_stable": response.stable,
        "speedup": result.speedup if response.correct and response.stable else 0.0,
        "status": result.status,
        "error_message": response.error_message,
        "completion": completion_text,
        "tokens_generated": int(generated.shape[0]),
        "generation_time_sec": round(generation_time, 4),
        "tokens_per_sec": round(int(generated.shape[0]) / max(generation_time, 1e-6), 2),
        "compiler_output": response.compiler_output,
    }


def _multi_turn_attempt(
    model: Any,
    tokenizer: Any,
    task_id: str,
    prompt_mode: str,
    config: ExperimentConfig,
    reward_engine: RewardEngine,
    do_sample: bool,
    temperature: Optional[float],
) -> Dict[str, Any]:
    original_temperature = config.temperature
    if temperature is not None:
        config.temperature = temperature
    rollout_manager = RolloutManager(model, tokenizer, reward_engine, config)
    start_time = time.perf_counter()
    try:
        trajectories = rollout_manager.collect(
            [task_id],
            prompt_mode=prompt_mode,
            group_size=1,
            num_turns=config.eval_turns,
            do_sample=do_sample,
        )
    finally:
        config.temperature = original_temperature
    generation_time = time.perf_counter() - start_time
    trajectory = trajectories[0]
    best_response = trajectory.best_response
    best_code = trajectory.best_code or trajectory.last_code
    if best_response is None:
        last_step = trajectory.steps[-1] if trajectory.steps else None
        status = last_step.status if last_step else "no_attempt"
        compiled = any(step.response.compiled for step in trajectory.steps)
        correct = False
        stable = False
        speedup = 0.0
        error_message = last_step.response.error_message if last_step else "No turns executed."
    else:
        status = "ok"
        compiled = True
        correct = True
        stable = True
        speedup = trajectory.best_speedup
        error_message = None
    return {
        "task_id": task_id,
        "mode": "multi_turn",
        "compiled": compiled,
        "numerically_correct": correct,
        "is_stable": stable,
        "speedup": speedup if correct and stable else 0.0,
        "status": status,
        "error_message": error_message,
        "completion": best_code,
        "turns_executed": len(trajectory.steps),
        "tokens_generated": sum(len(step.completion_ids) for step in trajectory.steps),
        "generation_time_sec": round(generation_time, 4),
        "turn_records": [
            {
                "turn_index": step.turn_index,
                "status": step.status,
                "speedup": step.speedup,
                "reward": step.reward,
                "feedback": step.feedback,
                "code": step.code,
            }
            for step in trajectory.steps
        ],
    }


def _summarize_records(records: List[Dict[str, Any]], task_keys: List[str]) -> Dict[str, Any]:
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        by_task.setdefault(record["task_id"], []).append(record)

    best_records = []
    for task_id in task_keys:
        attempts = by_task.get(task_id, [])
        correct_attempts = [
            attempt
            for attempt in attempts
            if attempt["numerically_correct"] and attempt["is_stable"]
        ]
        if correct_attempts:
            best_records.append(max(correct_attempts, key=lambda attempt: attempt["speedup"]))
        elif attempts:
            best_records.append(
                max(
                    attempts,
                    key=lambda attempt: (
                        attempt["numerically_correct"],
                        attempt.get("compiled", False),
                    ),
                )
            )
        else:
            best_records.append(
                {
                    "task_id": task_id,
                    "speedup": 0.0,
                    "numerically_correct": False,
                    "compiled": False,
                    "is_stable": False,
                }
            )

    correct_count = sum(1 for record in best_records if record["numerically_correct"])
    stable_speedups = [
        float(record["speedup"])
        for record in best_records
        if record["numerically_correct"] and record["is_stable"] and record["speedup"] > 0
    ]
    total_tasks = len(task_keys)
    total_attempts = len(records)
    attempt_correct = sum(1 for record in records if record["numerically_correct"])
    return {
        "total_tasks": total_tasks,
        "total_attempts": total_attempts,
        "correct_count": correct_count,
        "correctness_rate": round(correct_count / total_tasks, 4) if total_tasks else 0.0,
        "attempt_correctness_rate": round(attempt_correct / total_attempts, 4)
        if total_attempts
        else 0.0,
        "fast_1_0_rate": round(sum(1 for value in stable_speedups if value >= 1.0) / total_tasks, 4)
        if total_tasks
        else 0.0,
        "fast_1_1_rate": round(sum(1 for value in stable_speedups if value >= 1.1) / total_tasks, 4)
        if total_tasks
        else 0.0,
        "median_speedup": round(statistics.median(stable_speedups), 4) if stable_speedups else 0.0,
        "mean_speedup": round(statistics.mean(stable_speedups), 4) if stable_speedups else 0.0,
        "p25_speedup": round(statistics.quantiles(stable_speedups, n=4)[0], 4)
        if len(stable_speedups) >= 2
        else 0.0,
        "p75_speedup": round(statistics.quantiles(stable_speedups, n=4)[2], 4)
        if len(stable_speedups) >= 2
        else 0.0,
        "median_speedup_ci": bootstrap_median_ci(stable_speedups),
        "best_speedup": round(max(stable_speedups), 4) if stable_speedups else 0.0,
        "tasks": records,
        "task_best": best_records,
    }


def evaluate_checkpoint(
    model: Any,
    tokenizer: Any,
    task_keys: List[str],
    prompt_mode: str = "reference_only",
    config: Optional[ExperimentConfig] = None,
    dispatcher: Optional[Any] = None,
    mode: Optional[str] = None,
    do_sample: bool = False,
    temperature: Optional[float] = None,
    num_samples: int = 1,
) -> Dict[str, Any]:
    cfg = config or ExperimentConfig()
    cfg.validate()
    mode = mode or cfg.mode
    if dispatcher is None:
        dispatcher = build_dispatcher(cfg, kind="in_process")
    reward_engine = RewardEngine(cfg, dispatcher)
    model.eval()
    set_seed(cfg.seed)
    records: List[Dict[str, Any]] = []
    for task_id in task_keys:
        for _ in range(max(1, num_samples)):
            if mode in {"multi_turn", "contrastive"}:
                record = _multi_turn_attempt(
                    model,
                    tokenizer,
                    task_id,
                    prompt_mode,
                    cfg,
                    reward_engine,
                    do_sample=do_sample,
                    temperature=temperature,
                )
            else:
                record = _single_turn_attempt(
                    model,
                    tokenizer,
                    task_id,
                    prompt_mode,
                    reward_engine,
                    cfg,
                    do_sample=do_sample,
                    temperature=temperature,
                )
            records.append(record)

    summary = _summarize_records(records, task_keys)
    split = {}
    for split_name, split_tasks in (
        ("training", cfg.training_tasks),
        ("holdout", cfg.holdout_tasks),
    ):
        selected = [record for record in records if record["task_id"] in split_tasks]
        if selected:
            split[split_name] = _summarize_records(
                selected, [task for task in task_keys if task in split_tasks]
            )
    summary["by_split"] = split
    summary["metadata"] = {
        "mode": mode,
        "prompt_mode": prompt_mode,
        "num_samples": num_samples,
        "do_sample": do_sample,
        "temperature": temperature,
        "config": cfg.to_dict(),
        "environment": environment_metadata(),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CUDA generation policy.")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--sft_adapter_path", default=None)
    parser.add_argument(
        "--prompt_mode", default="reference_only", choices=["reference_only", "contrastive"]
    )
    parser.add_argument(
        "--mode", default=None, choices=["single_turn", "multi_turn", "contrastive"]
    )
    parser.add_argument("--use_dev_model", action="store_true")
    parser.add_argument("--use_tiny_model", action="store_true")
    parser.add_argument("--base_model_control", action="store_true")
    parser.add_argument("--tasks", default=None, help="Comma separated task ids.")
    parser.add_argument("--num_turns", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--verifier_device", default=None)
    parser.add_argument(
        "--dispatcher", default="auto", choices=["auto", "subprocess", "in_process", "mock"]
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_file", default="./outputs/eval_results/eval_summary.json")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    config.use_dev_model = config.use_dev_model or args.use_dev_model
    config.use_tiny_model = config.use_tiny_model or args.use_tiny_model
    if args.mode:
        config.mode = args.mode
    if args.num_turns:
        config.eval_turns = args.num_turns
    if args.device:
        config.device = args.device
    if args.verifier_device:
        config.verifier_device = args.verifier_device
    if args.sft_adapter_path:
        config.sft_adapter_path = args.sft_adapter_path
    config.resolve_paths()
    set_seed(config.seed)

    adapter_path = None if args.base_model_control else args.adapter_path
    if args.base_model_control:
        config.sft_adapter_path = None
    model, tokenizer = load_model_and_tokenizer(
        config=config,
        is_trainable=False,
        adapter_path=adapter_path,
    )
    dispatcher = build_dispatcher(config, kind=args.dispatcher)
    task_keys = args.tasks.split(",") if args.tasks else list(config.eval_tasks)
    try:
        summary = evaluate_checkpoint(
            model=model,
            tokenizer=tokenizer,
            task_keys=task_keys,
            prompt_mode=args.prompt_mode,
            config=config,
            dispatcher=dispatcher,
            mode=config.mode,
            do_sample=args.do_sample,
            temperature=args.temperature,
            num_samples=args.num_samples,
        )
    finally:
        close = getattr(dispatcher, "close", None)
        if close is not None:
            close()

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nEvaluation Summary")
    print(f"Total tasks:        {summary['total_tasks']}")
    print(f"Correctness rate:   {summary['correctness_rate'] * 100:.2f}%")
    print(f"Fast >= 1.0x rate:  {summary['fast_1_0_rate'] * 100:.2f}%")
    print(f"Fast >= 1.1x rate:  {summary['fast_1_1_rate'] * 100:.2f}%")
    print(f"Median speedup:     {summary['median_speedup']:.2f}x")
    print(f"Median 95% CI:      {summary['median_speedup_ci']}")
    print(f"Results saved to:   {args.output_file}")


if __name__ == "__main__":
    main()
