import contextlib
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import get_cosine_schedule_with_warmup

from engine.prompt_builder import build_turn_messages, format_chat_prompt
from engine.utils import JsonlWriter, now_iso
from src.pipeline.rollout_manager import RolloutManager, build_training_samples
from src.rewards.archive import ArchiveEntry, ContrastiveArchive
from src.trainers.losses import (
    dpo_loss,
    gather_completion_logprobs,
    grpo_loss,
    masked_sequence_logprob,
)


@dataclass
class TrainerMetrics:
    step: int
    loss: float
    num_samples: int
    mean_reward: float
    mean_speedup: float
    best_speedup: float
    optimizer_steps: int
    status_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "loss": self.loss,
            "num_samples": self.num_samples,
            "mean_reward": self.mean_reward,
            "mean_speedup": self.mean_speedup,
            "best_speedup": self.best_speedup,
            "optimizer_steps": self.optimizer_steps,
            "status_counts": self.status_counts,
            "timestamp": now_iso(),
        }


class MultiTurnGRPOTrainer:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        rollout_manager: RolloutManager,
        config: Any,
        output_dir: str,
        archive: Optional[ContrastiveArchive] = None,
        reference_model: Optional[Any] = None,
        enable_contrastive: bool = False,
        prompt_mode: str = "reference_only",
    ) -> None:
        self.model = model
        self.prompt_mode = prompt_mode
        self.tokenizer = tokenizer
        self.rollout_manager = rollout_manager
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.archive = archive
        self.reference_model = reference_model
        self.enable_contrastive = enable_contrastive
        self.device = next(model.parameters()).device
        self.metrics_writer = JsonlWriter(str(self.output_dir / "train_metrics.jsonl"))

        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("No trainable parameters found. Check the LoRA adapter configuration.")
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        steps_per_rollout = max(
            1,
            (config.num_turns * config.effective_group_size() * config.tasks_per_step)
            // max(1, config.per_device_train_batch_size * config.gradient_accumulation_steps),
        )
        total_steps = max(1, config.max_steps * steps_per_rollout)
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        self.use_amp = bool(config.bf16 or config.fp16)
        self.amp_dtype = None
        if config.bf16:
            self.amp_dtype = torch.bfloat16
        elif config.fp16:
            self.amp_dtype = torch.float16
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=bool(config.fp16))
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=bool(config.fp16))
        self.global_step = 0
        self._contrastive_warned = False

    def _autocast(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _pad_sequences(self, sequences: List[List[int]]) -> Dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), max_length),
            pad_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, : len(sequence)] = 1
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }

    def _collate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        sequences = [sample["input_ids"] for sample in samples]
        padded = self._pad_sequences(sequences)
        max_completion = max(sample["completion_length"] for sample in samples)
        old_logprobs = torch.zeros(
            (len(samples), max_completion),
            dtype=torch.float32,
            device=self.device,
        )
        completion_mask = torch.zeros(
            (len(samples), max_completion),
            dtype=torch.float32,
            device=self.device,
        )
        for row, sample in enumerate(samples):
            length = sample["completion_length"]
            old_logprobs[row, :length] = torch.tensor(
                sample["old_logprobs"],
                dtype=torch.float32,
                device=self.device,
            )
            completion_mask[row, :length] = 1.0
        padded["prompt_lengths"] = [sample["prompt_length"] for sample in samples]
        padded["completion_lengths"] = [sample["completion_length"] for sample in samples]
        padded["old_logprobs"] = old_logprobs
        padded["completion_mask"] = completion_mask
        padded["advantages"] = torch.tensor(
            [sample["advantage"] for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        return padded

    def _model_logits(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        keep_last: int,
    ) -> torch.Tensor:
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                num_logits_to_keep=max(1, keep_last),
            )
        except TypeError:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        return outputs.logits

    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        keep_last = max(batch["completion_lengths"]) + 1
        with self._autocast():
            logits = self._model_logits(
                self.model,
                batch["input_ids"],
                batch["attention_mask"],
                keep_last,
            )
        logit_offset = int(batch["input_ids"].shape[1] - logits.shape[1])
        new_logprobs, mask = gather_completion_logprobs(
            logits,
            batch["input_ids"],
            batch["prompt_lengths"],
            batch["completion_lengths"],
            logit_offset=logit_offset,
        )
        if new_logprobs.shape != batch["old_logprobs"].shape:
            raise ValueError(
                f"Logprob shape mismatch: {new_logprobs.shape} vs {batch['old_logprobs'].shape}"
            )
        loss = grpo_loss(
            new_logprobs,
            batch["old_logprobs"],
            batch["advantages"],
            mask,
            epsilon_low=self.config.epsilon,
            epsilon_high=self.config.epsilon_high,
            loss_type=self.config.loss_type,
            l_norm=float(self.config.dr_grpo_l_norm),
        )
        if (
            self.enable_contrastive
            and self.config.enable_cross_trajectory_anchors
            and self.archive is not None
            and self.config.pairwise_margin_weight > 0
        ):
            contrastive = self._contrastive_loss()
            if contrastive is not None:
                loss = loss + self.config.pairwise_margin_weight * contrastive
        return loss

    def _code_completion(self, entry: ArchiveEntry) -> str:
        raw = str(entry.metadata.get("raw_completion", "")).strip()
        if raw:
            return raw
        return f"```cpp\n{entry.code}\n```"

    def _encode_pair(
        self, task_id: str, winner: ArchiveEntry, loser: ArchiveEntry
    ) -> Optional[Dict[str, Any]]:
        prompt_text = format_chat_prompt(
            self.tokenizer,
            build_turn_messages(task_id, mode=self.prompt_mode),
        )
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_prompt_length,
        )["input_ids"]
        winner_ids = self.tokenizer(
            self._code_completion(winner),
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_completion_length,
        )["input_ids"]
        loser_ids = self.tokenizer(
            self._code_completion(loser),
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_completion_length,
        )["input_ids"]
        if not prompt_ids or not winner_ids or not loser_ids:
            return None
        sequences = [list(prompt_ids) + list(winner_ids), list(prompt_ids) + list(loser_ids)]
        padded = self._pad_sequences(sequences)
        padded["prompt_lengths"] = [len(prompt_ids), len(prompt_ids)]
        padded["completion_lengths"] = [len(winner_ids), len(loser_ids)]
        return padded

    @contextlib.contextmanager
    def _reference_context(self):
        if self.reference_model is not None:
            yield
            return
        peft_config = getattr(self.model, "peft_config", {})
        if "sft" in peft_config and hasattr(self.model, "set_adapter"):
            active = self.model.active_adapter
            self.model.set_adapter("sft")
            try:
                yield
            finally:
                self.model.set_adapter(active)
            return
        disable = getattr(self.model, "disable_adapter", None)
        if disable is not None:
            with disable():
                yield
            return
        raise RuntimeError("No reference policy is available for contrastive training.")

    def _forward_sequence_logprob(self, batch: Dict[str, Any], use_reference: bool) -> torch.Tensor:
        model = (
            self.reference_model
            if use_reference and self.reference_model is not None
            else self.model
        )
        context = self._reference_context() if use_reference else contextlib.nullcontext()
        was_training = model.training
        if use_reference:
            model.eval()
        try:
            with torch.set_grad_enabled(not use_reference):
                with context:
                    keep_last = max(batch["completion_lengths"]) + 1
                    with self._autocast():
                        logits = self._model_logits(
                            model,
                            batch["input_ids"],
                            batch["attention_mask"],
                            keep_last,
                        )
                    logit_offset = int(batch["input_ids"].shape[1] - logits.shape[1])
                    logprob = masked_sequence_logprob(
                        logits,
                        batch["input_ids"],
                        batch["prompt_lengths"],
                        batch["completion_lengths"],
                        logit_offset=logit_offset,
                    )
        finally:
            if use_reference:
                model.train(was_training)
        return logprob

    def _contrastive_loss(self) -> Optional[torch.Tensor]:
        if self.reference_model is None and not hasattr(self.model, "disable_adapter"):
            if not self._contrastive_warned:
                print("Contrastive loss disabled: no reference policy or PEFT adapter.")
                self._contrastive_warned = True
            return None
        pairs = self.archive.all_pairs(
            max_pairs=self.config.contrastive_pairs_per_step,
            seed=self.global_step,
        )
        losses = []
        for task_id, winner, loser in pairs:
            batch = self._encode_pair(task_id, winner, loser)
            if batch is None:
                continue
            policy_logp = self._forward_sequence_logprob(batch, use_reference=False)
            reference_logp = self._forward_sequence_logprob(batch, use_reference=True)
            losses.append(
                dpo_loss(
                    policy_logp[0],
                    policy_logp[1],
                    reference_logp[0].detach(),
                    reference_logp[1].detach(),
                    beta=self.config.pairwise_beta,
                )
            )
        if not losses:
            return None
        return torch.stack(losses).mean()

    def _train_samples(self, samples: List[Dict[str, Any]], step: int) -> Tuple[float, int]:
        if not samples:
            return 0.0, 0
        rng = random.Random(self.config.seed + step)
        rng.shuffle(samples)
        batch_size = max(1, int(self.config.per_device_train_batch_size))
        accumulation = max(1, int(self.config.gradient_accumulation_steps))
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        loss_count = 0
        optimizer_steps = 0
        for index in range(0, len(samples), batch_size):
            batch_samples = samples[index : index + batch_size]
            batch = self._collate(batch_samples)
            loss = self._compute_loss(batch)
            loss_value = float(loss.detach().item())
            total_loss += loss_value
            loss_count += 1
            scaled_loss = loss / accumulation
            if self.scaler.is_enabled():
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            should_step = ((index // batch_size) + 1) % accumulation == 0
            is_last = index + batch_size >= len(samples)
            if should_step or is_last:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                    self.config.max_grad_norm,
                )
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        return total_loss / max(1, loss_count), optimizer_steps

    def save_checkpoint(self, step: int) -> None:
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Save only the RL adapter if we have multiple adapters (e.g., SFT + RL)
        if hasattr(self.model, "peft_config") and "rl" in self.model.peft_config:
            self.model.save_pretrained(checkpoint_dir, selected_adapters=["rl"])
        else:
            self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        torch.save(
            {
                "step": step,
                "global_step": self.global_step,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            checkpoint_dir / "trainer_state.pt",
        )
        self.config.to_yaml(str(checkpoint_dir / "config.yaml"))
        checkpoints = sorted(
            [path for path in self.output_dir.glob("checkpoint-*") if path.is_dir()],
            key=lambda path: int(path.name.split("-")[-1]),
        )
        while len(checkpoints) > self.config.save_total_limit:
            shutil.rmtree(checkpoints.pop(0), ignore_errors=True)
        print(f"Saved checkpoint to {checkpoint_dir}")

    def resume_from_checkpoint(self, checkpoint_dir: str) -> None:
        state_path = Path(checkpoint_dir) / "trainer_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing trainer state: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self.global_step = int(state.get("global_step", state.get("step", 0)))
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if state.get("torch_rng") is not None:
            torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resumed optimizer state from {checkpoint_dir} at step {self.global_step}")

    def _summarize(
        self,
        trajectories: List[Any],
        sample_count: int,
        loss: float,
        optimizer_steps: int,
        step: int,
    ) -> TrainerMetrics:
        rewards = [step.reward for trajectory in trajectories for step in trajectory.steps]
        speedups = [
            step.speedup
            for trajectory in trajectories
            for step in trajectory.steps
            if step.response.correct and step.response.stable
        ]
        status_counts: Dict[str, int] = {}
        for trajectory in trajectories:
            for turn in trajectory.steps:
                status_counts[turn.status] = status_counts.get(turn.status, 0) + 1
        return TrainerMetrics(
            step=step,
            loss=loss,
            num_samples=sample_count,
            mean_reward=sum(rewards) / len(rewards) if rewards else 0.0,
            mean_speedup=sum(speedups) / len(speedups) if speedups else 0.0,
            best_speedup=max(speedups) if speedups else 0.0,
            optimizer_steps=optimizer_steps,
            status_counts=status_counts,
        )

    def train(self) -> List[TrainerMetrics]:
        self.config.validate()
        self.model.config.use_cache = False
        metrics_history: List[TrainerMetrics] = []
        if self.config.max_steps <= 0:
            return metrics_history
        tasks = list(self.config.training_tasks)
        for step in range(self.global_step, self.config.max_steps):
            task_ids = [
                tasks[(step * self.config.tasks_per_step + index) % len(tasks)]
                for index in range(self.config.tasks_per_step)
            ]
            trajectories = self.rollout_manager.collect(
                task_ids,
                prompt_mode=self.prompt_mode,
                group_size=self.config.effective_group_size(),
                num_turns=self.config.num_turns,
                do_sample=True,
            )
            samples = build_training_samples(trajectories)
            loss, optimizer_steps = self._train_samples(samples, step)
            metrics = self._summarize(trajectories, len(samples), loss, optimizer_steps, step)
            metrics_history.append(metrics)
            self.metrics_writer.write(metrics.to_dict())
            print(
                f"step {step} loss={metrics.loss:.4f} samples={metrics.num_samples} "
                f"reward={metrics.mean_reward:.3f} speedup={metrics.mean_speedup:.2f} "
                f"best={metrics.best_speedup:.2f}"
            )
            if self.archive is not None:
                self.archive.save()
            self.global_step = step + 1
            if self.config.save_steps > 0 and self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(self.global_step)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.save_checkpoint(self.global_step)
        return metrics_history


class ContrastiveRLTrainer(MultiTurnGRPOTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["enable_contrastive"] = True
        super().__init__(*args, **kwargs)
