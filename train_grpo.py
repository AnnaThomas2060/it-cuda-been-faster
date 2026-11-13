import argparse
import gc
import json
import logging
import os
from typing import Any, Dict, List, Optional

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from configs.grpo_config import ExperimentConfig
from engine.prompt_builder import build_turn_messages, format_chat_prompt
from engine.utils import environment_metadata, set_seed
from models.loader import load_model_and_tokenizer
from rewards.grpo_reward import (
    get_taxonomy_summary,
    make_reward_function,
    reset_taxonomy_stats,
)
from src.pipeline.env_dispatcher import build_dispatcher
from src.pipeline.rollout_manager import RolloutManager
from src.rewards.archive import ContrastiveArchive
from src.rewards.reward_engine import RewardEngine
from src.trainers.multiturn_grpo import ContrastiveRLTrainer, MultiTurnGRPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_training_dataset(task_ids: List[str], num_passes: int, prompt_mode: str):
    from datasets import Dataset

    from engine.prompt_builder import build_prompt

    prompts: List[Dict[str, Any]] = []
    for pass_index in range(num_passes):
        for task_id in task_ids:
            prompts.append(
                {
                    "prompt": build_prompt(task_id, mode=prompt_mode),
                    "task_id": task_id,
                    "pass_idx": pass_index,
                }
            )
    return Dataset.from_list(prompts)


def build_chat_dataset(
    task_ids: List[str],
    num_passes: int,
    prompt_mode: str,
    tokenizer: Any,
):
    from datasets import Dataset

    prompts: List[Dict[str, Any]] = []
    for pass_index in range(num_passes):
        for task_id in task_ids:
            messages = build_turn_messages(task_id, mode=prompt_mode)
            prompts.append(
                {
                    "prompt": format_chat_prompt(tokenizer, messages),
                    "task_id": task_id,
                    "pass_idx": pass_index,
                }
            )
    return Dataset.from_list(prompts)


def _resync_lora_dtype(model: Any, target_dtype: Any) -> None:
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(target_dtype)


def _save_adapter(model: Any, path: str) -> None:
    peft_config = getattr(model, "peft_config", None)
    if peft_config and "rl" in peft_config:
        # Save only the rl adapter at the root of path.
        # Extract rl adapter weights and config, save as standalone adapter.
        from pathlib import Path

        from safetensors.torch import save_file

        Path(path).mkdir(parents=True, exist_ok=True)
        # Get rl adapter state dict (keys have "rl." prefix)
        state_dict = model.state_dict()
        rl_state = {k.replace("rl.", ""): v for k, v in state_dict.items() if k.startswith("rl.")}
        # Save weights
        save_file(rl_state, Path(path) / "adapter_model.safetensors")
        # Save config
        rl_config = peft_config["rl"]
        rl_config.save_pretrained(path)
    else:
        model.save_pretrained(path)


def run_single_turn_training(
    arm: str,
    config: ExperimentConfig,
    dry_run: bool = False,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    dispatcher: Optional[Any] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> str:
    import torch
    from trl import GRPOConfig, GRPOTrainer

    prompt_mode = config.prompt_mode_for_arm(arm)
    output_dir = config.output_dir_for_arm(arm)
    os.makedirs(output_dir, exist_ok=True)
    reset_taxonomy_stats()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(config=config, is_trainable=True)

    train_dataset = build_chat_dataset(
        task_ids=config.training_tasks,
        num_passes=config.training_passes,
        prompt_mode=prompt_mode,
        tokenizer=tokenizer,
    )
    has_cuda = torch.cuda.is_available()
    is_bf16 = has_cuda and torch.cuda.is_bf16_supported() and config.bf16
    is_fp16 = has_cuda and not is_bf16
    generation_batch_size = config.num_generations * max(1, config.gradient_accumulation_steps)

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        beta=config.beta,
        num_generations=config.num_generations,
        generation_batch_size=generation_batch_size,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        seed=config.seed,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=is_bf16,
        fp16=is_fp16,
        logging_steps=config.logging_steps,
        save_strategy="no" if dry_run else "steps",
        save_steps=max(1, config.save_steps),
        save_total_limit=config.save_total_limit,
        max_steps=1 if dry_run else config.max_steps,
        num_train_epochs=config.num_train_epochs,
        report_to=[] if config.report_to == "none" else config.report_to,
        run_name=f"{config.run_name}_{arm}",
    )

    reward_func = make_reward_function(config, dispatcher)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    target_lora_dtype = torch.bfloat16 if is_bf16 else torch.float32
    _resync_lora_dtype(trainer.model, target_lora_dtype)
    if hasattr(trainer, "model_wrapped") and trainer.model_wrapped is not None:
        _resync_lora_dtype(trainer.model_wrapped, target_lora_dtype)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    adapter_save_path = os.path.join(output_dir, "final_adapter")
    _save_adapter(trainer.model, adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)
    logger.info("Saved single-turn adapter to %s", adapter_save_path)

    summary = get_taxonomy_summary()
    logger.info("Reward taxonomy summary: %s", summary)

    del trainer
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return adapter_save_path


def run_multi_turn_training(
    arm: str,
    config: ExperimentConfig,
    dry_run: bool = False,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    dispatcher: Optional[Any] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> str:
    import torch

    prompt_mode = config.prompt_mode_for_arm(arm)
    output_dir = config.output_dir_for_arm(arm)
    os.makedirs(output_dir, exist_ok=True)
    reset_taxonomy_stats()

    if model is None or tokenizer is None:
        adapter_path = resume_from_checkpoint if resume_from_checkpoint else None
        model, tokenizer = load_model_and_tokenizer(
            config=config,
            is_trainable=True,
            adapter_path=adapter_path,
        )

    archive = ContrastiveArchive(
        path=config.archive_path_for_arm(arm),
        capacity_per_task=config.archive_capacity_per_task,
        min_speedup_margin_delta=config.min_speedup_margin_delta,
    )
    reward_engine = RewardEngine(config, dispatcher, archive=archive)
    rollout_manager = RolloutManager(model, tokenizer, reward_engine, config)
    trainer_class = ContrastiveRLTrainer if config.mode == "contrastive" else MultiTurnGRPOTrainer
    trainer = trainer_class(
        model=model,
        tokenizer=tokenizer,
        rollout_manager=rollout_manager,
        config=config,
        output_dir=output_dir,
        archive=archive,
        prompt_mode=prompt_mode,
        enable_contrastive=config.mode == "contrastive",
    )
    if resume_from_checkpoint:
        trainer.resume_from_checkpoint(resume_from_checkpoint)
    metrics = trainer.train()

    adapter_save_path = os.path.join(output_dir, "final_adapter")
    _save_adapter(model, adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)
    archive.save()

    metadata = {
        "arm": arm,
        "mode": config.mode,
        "prompt_mode": prompt_mode,
        "config": config.to_dict(),
        "environment": environment_metadata(),
        "taxonomy": get_taxonomy_summary(),
        "archive": archive.summary(),
        "metrics": [metric.to_dict() for metric in metrics],
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Saved multi-turn adapter to %s", adapter_save_path)

    reward_engine.close()
    del trainer
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return adapter_save_path


def run_training(
    arm: str,
    config: ExperimentConfig,
    dry_run: bool = False,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    dispatcher_kind: str = "auto",
    resume_from_checkpoint: Optional[str] = None,
) -> str:
    config.validate()
    config.resolve_paths()
    set_seed(config.seed)
    if dry_run:
        config.max_steps = 1
        config.save_steps = 0
        config.logging_steps = 1

    logger.info("Starting %s training for %s.", config.mode, arm)
    logger.info("Model: %s", config.get_active_model_name())
    logger.info("Verifier device: %s | Training device: %s", config.verifier_device, config.device)

    dispatcher = build_dispatcher(config, kind=dispatcher_kind)
    try:
        if config.mode in {"multi_turn", "contrastive"}:
            return run_multi_turn_training(
                arm,
                config,
                dry_run=dry_run,
                model=model,
                tokenizer=tokenizer,
                dispatcher=dispatcher,
                resume_from_checkpoint=resume_from_checkpoint,
            )
        return run_single_turn_training(
            arm,
            config,
            dry_run=dry_run,
            model=model,
            tokenizer=tokenizer,
            dispatcher=dispatcher,
            resume_from_checkpoint=resume_from_checkpoint,
        )
    finally:
        close = getattr(dispatcher, "close", None)
        if close is not None:
            close()


def run_grpo_training(
    arm: str,
    config: ExperimentConfig,
    dry_run: bool = False,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> str:
    return run_training(
        arm,
        config,
        dry_run=dry_run,
        model=model,
        tokenizer=tokenizer,
        dispatcher_kind="auto",
    )


def _apply_cli_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.mode:
        config.mode = args.mode
    if args.use_dev_model:
        config.use_dev_model = True
    if args.use_tiny_model:
        config.use_tiny_model = True
    if args.seed is not None:
        config.seed = args.seed
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.num_turns is not None:
        config.num_turns = args.num_turns
    if args.eval_turns is not None:
        config.eval_turns = args.eval_turns
    if args.group_size is not None:
        config.group_size = args.group_size
        config.num_generations = args.group_size
    if args.device:
        config.device = args.device
    if args.verifier_device:
        config.verifier_device = args.verifier_device
    if args.sft_adapter_path:
        config.sft_adapter_path = args.sft_adapter_path
    if args.output_dir:
        if args.arm == "both":
            config.output_dir_arm_a = os.path.join(args.output_dir, "arm_a")
            config.output_dir_arm_b = os.path.join(args.output_dir, "arm_b")
        else:
            if args.arm == "arm_a":
                config.output_dir_arm_a = args.output_dir
            else:
                config.output_dir_arm_b = args.output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched GRPO training arms.")
    parser.add_argument("--arm", default="arm_a", choices=["arm_a", "arm_b", "both"])
    parser.add_argument(
        "--mode", choices=["single_turn", "multi_turn", "contrastive"], default=None
    )
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument("--use_dev_model", action="store_true")
    parser.add_argument("--use_tiny_model", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_turns", type=int, default=None)
    parser.add_argument("--eval_turns", type=int, default=None)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--verifier_device", default=None)
    parser.add_argument(
        "--dispatcher", default="auto", choices=["auto", "subprocess", "in_process", "mock"]
    )
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--sft_adapter_path", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    _apply_cli_overrides(config, args)

    if args.arm in ("arm_a", "both"):
        run_training(
            "arm_a",
            config,
            dry_run=args.dry_run,
            dispatcher_kind=args.dispatcher,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    if args.arm in ("arm_b", "both"):
        run_training(
            "arm_b",
            config,
            dry_run=args.dry_run,
            dispatcher_kind=args.dispatcher,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )


if __name__ == "__main__":
    main()
