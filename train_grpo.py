import argparse
import gc
import logging
import os
from typing import Any, Dict, List, Optional

# Ensure memory allocator reduces fragmentation on 24 GiB and 40 GiB GPUs.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from configs.grpo_config import ExperimentConfig
from engine.prompt_builder import build_prompt
from models.loader import load_model_and_tokenizer
from rewards.grpo_reward import (
    compute_grpo_rewards,
    get_taxonomy_summary,
    reset_taxonomy_stats,
)

# Main training script executing matched baseline Group Relative Policy Optimization runs.
# Trains Arm A (contrastive) and Arm B (reference only) from identical initial states.
# Configures TRL GRPOTrainer with beta zero to eliminate value networks and conserve VRAM.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def build_training_dataset(task_ids: List[str], num_passes: int, prompt_mode: str):
    from datasets import Dataset

    prompts: List[Dict[str, str]] = []
    for pass_idx in range(num_passes):
        for tid in task_ids:
            prompt_text = build_prompt(tid, mode=prompt_mode)
            prompts.append({
                "prompt": prompt_text,
                "task_id": tid,
                "pass_idx": pass_idx,
            })
    return Dataset.from_list(prompts)

def run_grpo_training(
    arm: str,
    config: ExperimentConfig,
    dry_run: bool = False,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> str:
    import torch
    from trl import GRPOConfig, GRPOTrainer

    prompt_mode = "contrastive" if arm.lower() in ("arm_a", "contrastive") else "reference_only"
    output_dir = (
        config.output_dir_arm_a if prompt_mode == "contrastive" else config.output_dir_arm_b
    )
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Initiating GRPO training for {arm.upper()} in {prompt_mode} mode.")
    logger.info(f"Target output directory: {output_dir}")

    reset_taxonomy_stats()

    # Flush stale GPU allocations to reclaim all available VRAM
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_dataset = build_training_dataset(
        task_ids=config.training_tasks,
        num_passes=config.training_passes,
        prompt_mode=prompt_mode,
    )
    logger.info(
        f"Constructed training dataset containing {len(train_dataset)} scheduled prompt items."
    )

    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(config=config, is_trainable=True)

    max_steps = 1 if dry_run else -1
    num_epochs = 1 if dry_run else 1

    # Hardware-aware precision setup: Turing (T4) requires fp16; Ampere/Hopper (A100) supports bf16
    has_cuda = torch.cuda.is_available()
    is_bf16 = has_cuda and torch.cuda.is_bf16_supported() and config.bf16
    is_fp16 = has_cuda and not is_bf16

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        beta=config.beta,
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        seed=config.seed,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=is_bf16,
        fp16=is_fp16,
        logging_steps=config.logging_steps,
        save_strategy="no" if dry_run else "steps",
        save_steps=10,
        max_steps=max_steps,
        num_train_epochs=num_epochs,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[compute_grpo_rewards],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    logger.info("Commencing policy optimization steps.")
    trainer.train()

    adapter_save_path = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)
    logger.info(f"Successfully finalized training. Saved adapter to {adapter_save_path}.")

    summary = get_taxonomy_summary()
    logger.info(f"Post-run reward taxonomy summary: {summary}")

    # Explicitly tear down trainer and release memory
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return adapter_save_path

def main():
    parser = argparse.ArgumentParser(description="Run matched baseline GRPO training.")
    parser.add_argument(
        "--arm",
        type=str,
        default="arm_a",
        choices=["arm_a", "arm_b", "both"],
        help="Experimental arm: arm_a (contrastive), arm_b (reference_only), or both.",
    )
    parser.add_argument(
        "--use_dev_model",
        action="store_true",
        help="Use Qwen2.5-Coder-7B-Instruct for development iterations.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Execute a single step dry run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=635,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig()
    cfg.use_dev_model = args.use_dev_model
    cfg.seed = args.seed

    if args.arm in ("arm_a", "both"):
        run_grpo_training("arm_a", cfg, dry_run=args.dry_run)

    if args.arm in ("arm_b", "both"):
        run_grpo_training("arm_b", cfg, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
