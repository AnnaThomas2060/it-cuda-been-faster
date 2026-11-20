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
from data.example_store import get_teacher_examples
from engine.prompt_builder import build_turn_messages, format_chat_prompt
from engine.utils import environment_metadata, set_seed
from models.loader import load_model_and_tokenizer, verify_trainable_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_sft_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _to_id_list(ids: Any) -> List[int]:
    if torch.is_tensor(ids):
        return [int(x) for x in ids.flatten().tolist()]
    if ids and isinstance(ids[0], list):
        return [int(x) for x in ids[0]]
    return [int(x) for x in ids]


def build_sft_examples(
    task_ids: List[str],
    tokenizer: Any,
    max_seq_length: int = 4096,
) -> List[Dict[str, Any]]:
    eos = getattr(tokenizer, "eos_token", None) or ""
    samples: List[Dict[str, Any]] = []
    for task_id in task_ids:
        teachers = get_teacher_examples(task_id)
        for teacher in teachers:
            code = str(teacher.get("code", "")).strip()
            if not code:
                continue
            messages = build_turn_messages(task_id, mode="reference_only", turn_index=0)
            prompt_text = format_chat_prompt(tokenizer, messages)
            completion_text = f"```cpp\n{code}\n```{eos}"

            prompt_ids = _to_id_list(
                tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max(64, max_seq_length // 2),
                )["input_ids"]
            )
            completion_ids = _to_id_list(
                tokenizer(
                    completion_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max(64, max_seq_length - len(prompt_ids)),
                )["input_ids"]
            )
            if not prompt_ids or not completion_ids:
                continue

            input_ids = list(prompt_ids) + list(completion_ids)
            labels = ([-100] * len(prompt_ids)) + list(completion_ids)
            samples.append(
                {
                    "task_id": task_id,
                    "teacher_name": teacher.get("name", "unknown"),
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": [1] * len(input_ids),
                }
            )
    return samples


def _collate_sft(
    batch: List[Dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for row, item in enumerate(batch):
        length = len(item["input_ids"])
        input_ids[row, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[row, :length] = torch.tensor(item["labels"], dtype=torch.long)
        attention_mask[row, :length] = 1

    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": attention_mask.to(device),
    }


def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _save_adapter(model: Any, path: Path) -> None:
    peft_config = getattr(model, "peft_config", None)
    path.mkdir(parents=True, exist_ok=True)
    if peft_config and "rl" in peft_config:
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        rl_state = get_peft_model_state_dict(model, adapter_name="rl")
        save_file(rl_state, path / "adapter_model.safetensors")
        peft_config["rl"].save_pretrained(str(path))
    else:
        model.save_pretrained(str(path))


def run_sft_training(
    sft_cfg: Dict[str, Any],
    output_dir: Optional[str] = None,
    use_dev_model: bool = False,
    use_tiny_model: bool = False,
    dry_run: bool = False,
    device: Optional[str] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Tuple[str, Dict[str, Any]]:
    seed = int(sft_cfg.get("seed", 635))
    set_seed(seed)

    exp_cfg = ExperimentConfig()
    exp_cfg.base_model_name = str(sft_cfg.get("model_name_or_path", exp_cfg.base_model_name))
    exp_cfg.use_dev_model = bool(use_dev_model or sft_cfg.get("use_dev_model", False))
    exp_cfg.use_tiny_model = bool(use_tiny_model or sft_cfg.get("use_tiny_model", False))
    exp_cfg.lora_r = int(sft_cfg.get("lora_r", 8))
    exp_cfg.lora_alpha = int(sft_cfg.get("lora_alpha", 16))
    exp_cfg.lora_dropout = float(sft_cfg.get("lora_dropout", 0.0))
    if "target_modules" in sft_cfg:
        exp_cfg.target_modules = list(sft_cfg["target_modules"])
    if device:
        exp_cfg.device = device
    elif sft_cfg.get("device"):
        exp_cfg.device = str(sft_cfg["device"])

    resolved_output = Path(output_dir or sft_cfg.get("output_dir", "./outputs/sft_warmup"))
    resolved_output.mkdir(parents=True, exist_ok=True)

    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(config=exp_cfg, is_trainable=True)

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    task_ids = list(sft_cfg.get("training_tasks", exp_cfg.training_tasks))
    max_seq_length = int(sft_cfg.get("max_seq_length", 4096))
    examples = build_sft_examples(task_ids, tokenizer, max_seq_length=max_seq_length)
    if not examples:
        raise ValueError(f"No teacher SFT examples found for tasks: {task_ids}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters found for SFT.")

    lr = float(sft_cfg.get("learning_rate", 1e-6))
    weight_decay = float(sft_cfg.get("weight_decay", 0.01))
    num_epochs = 1 if dry_run else int(sft_cfg.get("num_epochs", 1))
    batch_size = max(1, int(sft_cfg.get("per_device_train_batch_size", 1)))
    grad_accum = max(1, int(sft_cfg.get("gradient_accumulation_steps", 1)))
    warmup_ratio = float(sft_cfg.get("warmup_ratio", 0.05))
    max_grad_norm = float(sft_cfg.get("max_grad_norm", 1.0))

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    eff_batch = batch_size * grad_accum
    steps_per_epoch = max(1, (len(examples) + eff_batch - 1) // eff_batch)
    total_steps = max(1, num_epochs * steps_per_epoch)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    model_device = next(model.parameters()).device
    has_cuda = model_device.type == "cuda"
    amp_dtype = None
    if has_cuda:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model.train()
    optimizer.zero_grad(set_to_none=True)
    rng = random.Random(seed)
    step_losses: List[Dict[str, Any]] = []
    global_step = 0

    for epoch in range(num_epochs):
        order = list(examples)
        rng.shuffle(order)
        if dry_run:
            order = order[:batch_size]
        for idx in range(0, len(order), batch_size):
            batch_items = order[idx : idx + batch_size]
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
            loss_val = float(loss.detach().item())
            (loss / grad_accum).backward()

            should_step = ((idx // batch_size) + 1) % grad_accum == 0
            is_last = idx + batch_size >= len(order)
            if should_step or is_last:
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            step_losses.append(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "tasks": [item["task_id"] for item in batch_items],
                    "teachers": [item["teacher_name"] for item in batch_items],
                    "loss": round(loss_val, 6),
                }
            )
            logger.info(
                "SFT epoch=%d step=%d loss=%.4f teachers=%s",
                epoch,
                global_step,
                loss_val,
                [item["teacher_name"] for item in batch_items],
            )

    adapter_dir = resolved_output / "final_adapter"
    _save_adapter(model, adapter_dir)
    tokenizer.save_pretrained(str(adapter_dir))

    trainable_params, all_params, pct = verify_trainable_parameters(
        model, exp_cfg.get_active_model_name()
    )
    metadata = {
        "model_name": exp_cfg.get_active_model_name(),
        "seed": seed,
        "num_examples": len(examples),
        "num_epochs": num_epochs,
        "global_steps": global_step,
        "learning_rate": lr,
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_pct": round(pct, 6),
        "step_losses": step_losses,
        "environment": environment_metadata(),
    }
    with open(resolved_output / "sft_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(adapter_dir), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 SFT LoRA warm-up on teacher kernels.")
    parser.add_argument("--config", default="configs/sft_lora.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--use_dev_model", action="store_true")
    parser.add_argument("--use_tiny_model", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    sft_cfg = load_sft_yaml(args.config)
    if args.seed is not None:
        sft_cfg["seed"] = args.seed
    if args.num_epochs is not None:
        sft_cfg["num_epochs"] = args.num_epochs
    if args.learning_rate is not None:
        sft_cfg["learning_rate"] = args.learning_rate

    adapter_path, _ = run_sft_training(
        sft_cfg,
        output_dir=args.output_dir,
        use_dev_model=args.use_dev_model,
        use_tiny_model=args.use_tiny_model,
        dry_run=args.dry_run,
        device=args.device,
    )
    logger.info("Saved SFT warm-up adapter to %s", adapter_path)


if __name__ == "__main__":
    main()
