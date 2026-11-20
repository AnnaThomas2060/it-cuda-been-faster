import argparse
import contextlib
import gc
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from transformers import get_cosine_schedule_with_warmup

from configs.grpo_config import ExperimentConfig
from engine.utils import environment_metadata, set_seed, sha256_text
from models.loader import load_model_and_tokenizer
from src.pipeline.env_dispatcher import build_dispatcher
from src.pipeline.rollout_manager import RolloutManager
from src.rewards.reward_engine import RewardEngine
from train_sft import _causal_lm_loss, _collate_sft, _save_adapter, _to_id_list

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_rest_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def select_accepted_samples(
    trajectories: List[Any],
    tokenizer: Any,
    speedup_threshold: float,
    max_kept_per_task: int,
    max_seq_length: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    eos = getattr(tokenizer, "eos_token", None) or ""
    status_counts: Dict[str, int] = {}
    by_task: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for trajectory in trajectories:
        for step in trajectory.steps:
            status_counts[step.status] = status_counts.get(step.status, 0) + 1
            if not (step.response and step.response.compiled and step.response.correct):
                continue
            if not step.response.stable or step.speedup < speedup_threshold:
                continue
            code = step.code.strip()
            if not code:
                continue

            code_hash = sha256_text(code)
            completion_text = f"```cpp\n{code}\n```{eos}"
            completion_ids = _to_id_list(
                tokenizer(
                    completion_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max(64, max_seq_length - len(step.prompt_ids)),
                )["input_ids"]
            )
            if not step.prompt_ids or not completion_ids:
                continue

            input_ids = list(step.prompt_ids) + list(completion_ids)
            labels = ([-100] * len(step.prompt_ids)) + list(completion_ids)
            candidate = {
                "task_id": trajectory.task_id,
                "code_hash": code_hash,
                "speedup": float(step.speedup),
                "input_ids": input_ids,
                "labels": labels,
            }
            bucket = by_task.setdefault(trajectory.task_id, {})
            existing = bucket.get(code_hash)
            if existing is None or candidate["speedup"] > existing["speedup"]:
                bucket[code_hash] = candidate

    selected: List[Dict[str, Any]] = []
    for task_id in sorted(by_task):
        ranked = sorted(
            by_task[task_id].values(),
            key=lambda item: item["speedup"],
            reverse=True,
        )
        selected.extend(ranked[: max(1, max_kept_per_task)])
    return selected, status_counts


def run_rest_training(
    rest_cfg: Dict[str, Any],
    sft_adapter_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    use_dev_model: bool = False,
    use_tiny_model: bool = False,
    dry_run: bool = False,
    dispatcher_kind: str = "auto",
    device: Optional[str] = None,
    verifier_device: Optional[str] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Tuple[str, Dict[str, Any]]:
    seed = int(rest_cfg.get("seed", 635))
    set_seed(seed)

    exp_cfg = ExperimentConfig()
    exp_cfg.base_model_name = str(rest_cfg.get("model_name_or_path", exp_cfg.base_model_name))
    exp_cfg.use_dev_model = bool(use_dev_model or rest_cfg.get("use_dev_model", False))
    exp_cfg.use_tiny_model = bool(use_tiny_model or rest_cfg.get("use_tiny_model", False))
    exp_cfg.lora_r = int(rest_cfg.get("lora_r", 8))
    exp_cfg.lora_alpha = int(rest_cfg.get("lora_alpha", 16))
    exp_cfg.sft_adapter_path = sft_adapter_path or rest_cfg.get("sft_adapter_path")
    exp_cfg.temperature = float(rest_cfg.get("temperature", 0.8))
    exp_cfg.top_p = float(rest_cfg.get("top_p", 0.95))
    exp_cfg.max_prompt_length = int(rest_cfg.get("max_prompt_length", 3072))
    exp_cfg.max_completion_length = int(rest_cfg.get("max_completion_length", 1024))
    exp_cfg.rollout_batch_size = int(rest_cfg.get("rollout_batch_size", 1))
    exp_cfg.num_turns = int(rest_cfg.get("refinement_turns", 2))
    exp_cfg.seed = seed
    if device:
        exp_cfg.device = device
    elif rest_cfg.get("device"):
        exp_cfg.device = str(rest_cfg["device"])
    if verifier_device:
        exp_cfg.verifier_device = verifier_device
    elif rest_cfg.get("verifier_device"):
        exp_cfg.verifier_device = str(rest_cfg["verifier_device"])

    resolved_output = Path(output_dir or rest_cfg.get("output_dir", "./outputs/rest_selftrain"))
    resolved_output.mkdir(parents=True, exist_ok=True)
    exp_cfg.telemetry_path = str(resolved_output / "rest_telemetry.jsonl")

    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(config=exp_cfg, is_trainable=True)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters found for ReST.")

    num_iterations = 1 if dry_run else int(rest_cfg.get("num_iterations", 3))
    rollouts_per_task = 2 if dry_run else int(rest_cfg.get("rollouts_per_task", 4))
    refinement_turns = 1 if dry_run else int(rest_cfg.get("refinement_turns", 2))
    thresholds = list(rest_cfg.get("speedup_thresholds", [1.0]))
    max_kept = int(rest_cfg.get("max_kept_per_task", 4))
    max_seq_length = int(rest_cfg.get("max_seq_length", 4096))
    lr = float(rest_cfg.get("learning_rate", 1e-6))
    batch_size = max(1, int(rest_cfg.get("batch_size", 1)))
    task_ids = list(rest_cfg.get("training_tasks", exp_cfg.training_tasks))

    dispatcher = build_dispatcher(exp_cfg, kind=dispatcher_kind)
    reward_engine = RewardEngine(exp_cfg, dispatcher)
    rollout_manager = RolloutManager(model, tokenizer, reward_engine, exp_cfg)

    model_device = next(model.parameters()).device
    has_cuda = model_device.type == "cuda"
    amp_dtype = None
    if has_cuda:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    iteration_logs: List[Dict[str, Any]] = []
    rng = random.Random(seed)

    try:
        for iteration in range(num_iterations):
            threshold = float(thresholds[min(iteration, len(thresholds) - 1)])
            trajectories = rollout_manager.collect(
                task_ids=task_ids,
                prompt_mode="reference_only",
                group_size=rollouts_per_task,
                num_turns=refinement_turns,
                do_sample=True,
            )
            accepted, status_counts = select_accepted_samples(
                trajectories,
                tokenizer,
                speedup_threshold=threshold,
                max_kept_per_task=max_kept,
                max_seq_length=max_seq_length,
            )

            if not accepted:
                logger.warning(
                    "ReST iteration %d yielded 0 accepted samples (threshold=%.2f). "
                    "Skipping update. Statuses: %s",
                    iteration,
                    threshold,
                    status_counts,
                )
                iteration_logs.append(
                    {
                        "iteration": iteration,
                        "threshold": threshold,
                        "accepted_count": 0,
                        "status_counts": status_counts,
                        "mean_loss": None,
                        "updated": False,
                    }
                )
                continue

            rng.shuffle(accepted)
            optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
            total_steps = max(1, (len(accepted) + batch_size - 1) // batch_size)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses: List[float] = []

            for idx in range(0, len(accepted), batch_size):
                batch_items = accepted[idx : idx + batch_size]
                batch = _collate_sft(batch_items, int(tokenizer.pad_token_id), model_device)
                ctx = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if amp_dtype is not None
                    else contextlib.nullcontext()
                )
                with ctx:
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    )
                    loss = _causal_lm_loss(outputs.logits, batch["labels"])
                losses.append(float(loss.detach().item()))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            mean_loss = sum(losses) / len(losses)
            logger.info(
                "ReST iteration %d accepted=%d mean_loss=%.4f",
                iteration,
                len(accepted),
                mean_loss,
            )
            iteration_logs.append(
                {
                    "iteration": iteration,
                    "threshold": threshold,
                    "accepted_count": len(accepted),
                    "status_counts": status_counts,
                    "mean_loss": round(mean_loss, 6),
                    "updated": True,
                }
            )
    finally:
        reward_engine.close()

    adapter_dir = resolved_output / "final_adapter"
    _save_adapter(model, adapter_dir)
    tokenizer.save_pretrained(str(adapter_dir))

    metadata = {
        "model_name": exp_cfg.get_active_model_name(),
        "seed": seed,
        "sft_adapter_path": exp_cfg.sft_adapter_path,
        "iterations": iteration_logs,
        "environment": environment_metadata(),
    }
    with open(resolved_output / "rest_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(adapter_dir), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 ReST correctness-filtered training.")
    parser.add_argument("--config", default="configs/rest.yaml")
    parser.add_argument("--sft_adapter_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--verifier_device", default=None)
    parser.add_argument(
        "--dispatcher", default="auto", choices=["auto", "subprocess", "in_process", "mock"]
    )
    parser.add_argument("--use_dev_model", action="store_true")
    parser.add_argument("--use_tiny_model", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    rest_cfg = load_rest_yaml(args.config)
    adapter_path, _ = run_rest_training(
        rest_cfg,
        sft_adapter_path=args.sft_adapter_path,
        output_dir=args.output_dir,
        use_dev_model=args.use_dev_model,
        use_tiny_model=args.use_tiny_model,
        dry_run=args.dry_run,
        dispatcher_kind=args.dispatcher,
        device=args.device,
        verifier_device=args.verifier_device,
    )
    logger.info("Saved ReST adapter to %s", adapter_path)


if __name__ == "__main__":
    main()
