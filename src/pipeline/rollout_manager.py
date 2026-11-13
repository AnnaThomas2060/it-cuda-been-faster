import uuid
from typing import Any, Dict, List, Optional, Tuple

from engine.compiler import extract_code_block
from engine.prompt_builder import build_turn_messages, format_chat_prompt
from engine.utils import extract_thoughts
from src.contracts import KernelEvalResponse
from src.pipeline.state import Trajectory, TurnRecord, build_feedback
from src.rewards.reward_engine import RewardEngine, group_normalize


def _empty_response(task_id: str) -> KernelEvalResponse:
    return KernelEvalResponse(
        task_id=task_id,
        compiled=False,
        correct=False,
        status="empty_code",
        stable=False,
        error_message="Model response did not contain compilable code.",
    )


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return model.device


class RolloutManager:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        reward_engine: RewardEngine,
        config: Any,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.reward_engine = reward_engine
        self.config = config
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.device = _model_device(model)

    def _build_prompts(self, trajectories: List[Trajectory], turn_index: int) -> List[str]:
        prompts = []
        for trajectory in trajectories:
            best_code = trajectory.best_code or trajectory.last_code or None
            feedback = trajectory.steps[-1].feedback if trajectory.steps else None
            speedup = trajectory.best_speedup if trajectory.best_code else trajectory.last_speedup
            messages = build_turn_messages(
                trajectory.task_id,
                mode=trajectory.prompt_mode,
                turn_index=turn_index,
                best_code=best_code,
                feedback=feedback,
                speedup=speedup if best_code else None,
                history=trajectory.history(),
                max_teacher_examples=self.config.max_teacher_examples,
            )
            prompts.append(format_chat_prompt(self.tokenizer, messages))
        return prompts

    def _generate(self, prompts: List[str], do_sample: bool) -> Tuple[Any, Any, Any]:
        import torch

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.config.max_completion_length,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "repetition_penalty": self.config.repetition_penalty,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p
            generation_kwargs["top_k"] = self.config.top_k
            if self.config.min_p is not None:
                generation_kwargs["min_p"] = self.config.min_p
        with torch.no_grad():
            output_ids = self.model.generate(**generation_kwargs)
        return input_ids, attention_mask, output_ids

    def _completion_logprobs(
        self,
        input_ids: Any,
        attention_mask: Any,
        output_ids: Any,
        generated_ids: Any,
        completion_lengths: List[int],
    ) -> List[List[float]]:
        import torch

        from src.trainers.losses import gather_completion_logprobs

        full_attention_mask = torch.cat(
            [attention_mask, torch.ones_like(generated_ids)],
            dim=1,
        )
        keep_last = max(1, int(generated_ids.shape[1]) + 1)
        with torch.no_grad():
            try:
                logits = self.model(
                    input_ids=output_ids,
                    attention_mask=full_attention_mask,
                    use_cache=False,
                    num_logits_to_keep=keep_last,
                ).logits
            except TypeError:
                logits = self.model(
                    input_ids=output_ids,
                    attention_mask=full_attention_mask,
                    use_cache=False,
                ).logits
        logit_offset = int(output_ids.shape[1] - logits.shape[1])
        input_length = int(input_ids.shape[1])
        token_logprobs, _ = gather_completion_logprobs(
            logits,
            output_ids,
            [input_length] * int(generated_ids.shape[0]),
            completion_lengths,
            logit_offset=logit_offset,
        )
        # Free the large [batch, keep_last, vocab] logits tensor before copying the
        # per-token scalars back to the CPU. This is the dominant rollout-time
        # allocation for large-vocab models and the main single-GPU OOM risk.
        del logits
        results: List[List[float]] = []
        for row in range(generated_ids.shape[0]):
            length = completion_lengths[row]
            results.append([float(value) for value in token_logprobs[row, :length].tolist()])
        del token_logprobs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    def _prepare_turn_records(
        self,
        trajectories: List[Trajectory],
        prompts: List[str],
        input_ids: Any,
        attention_mask: Any,
        output_ids: Any,
        turn_index: int,
        do_sample: bool,
    ) -> None:

        input_length = int(input_ids.shape[1])
        generated_ids = output_ids[:, input_length:]
        eos_token_id = self.tokenizer.eos_token_id

        completion_texts: List[str] = []
        completion_id_lists: List[List[int]] = []
        prompt_id_lists: List[List[int]] = []
        for row in range(generated_ids.shape[0]):
            row_generated = generated_ids[row]
            if eos_token_id is None:
                stop = int(row_generated.shape[0])
            else:
                eos_positions = (row_generated == eos_token_id).nonzero(as_tuple=False).flatten()
                stop = (
                    int(eos_positions[0].item()) + 1
                    if eos_positions.numel() > 0
                    else int(row_generated.shape[0])
                )
            row_generated = row_generated[:stop]
            completion_id_lists.append([int(token) for token in row_generated.tolist()])
            completion_texts.append(self.tokenizer.decode(row_generated, skip_special_tokens=True))
            prompt_id_lists.append(
                [int(token) for token in input_ids[row][attention_mask[row].bool()].tolist()]
            )

        completion_lengths = [len(ids) for ids in completion_id_lists]
        old_logprobs = self._completion_logprobs(
            input_ids,
            attention_mask,
            output_ids,
            generated_ids,
            completion_lengths,
        )

        for index, trajectory in enumerate(trajectories):
            prompt_text = prompts[index]
            completion_text = completion_texts[index]
            code = extract_code_block(completion_text)
            if code.strip():
                reward_result = self.reward_engine.evaluate_code(
                    code,
                    trajectory.task_id,
                    turn_index=turn_index,
                    prompt_mode=trajectory.prompt_mode,
                    raw_completion=completion_text,
                )
                response = reward_result.response
                reward = reward_result.reward
                status = reward_result.status
                speedup = reward_result.speedup
            else:
                response = _empty_response(trajectory.task_id)
                reward = 0.0
                status = "empty_code"
                speedup = 0.0

            thought = extract_thoughts(completion_text)
            step = TurnRecord(
                turn_index=turn_index,
                prompt=prompt_text,
                prompt_ids=prompt_id_lists[index],
                completion_ids=completion_id_lists[index],
                completion_text=completion_text,
                old_logprobs=old_logprobs[index],
                response=response,
                reward=reward,
                code=code,
                status=status,
                speedup=speedup,
                feedback=build_feedback(response),
                thought=thought,
            )
            trajectory.add_step(step)
            if trajectory.best_speedup >= self.config.early_stop_speedup:
                trajectory.metadata["early_stopped"] = True
                trajectory.metadata["done"] = True

    def collect(
        self,
        task_ids: List[str],
        prompt_mode: str = "reference_only",
        group_size: Optional[int] = None,
        num_turns: Optional[int] = None,
        do_sample: bool = True,
    ) -> List[Trajectory]:
        import torch

        group_size = group_size or self.config.effective_group_size()
        num_turns = num_turns or self.config.num_turns
        trajectories = [
            Trajectory(
                trajectory_id=uuid.uuid4().hex,
                task_id=task_id,
                prompt_mode=prompt_mode,
            )
            for task_id in task_ids
            for _ in range(group_size)
        ]

        self.model.eval()
        for turn_index in range(num_turns):
            active = [
                trajectory for trajectory in trajectories if not trajectory.metadata.get("done")
            ]
            if not active:
                break
            prompts = self._build_prompts(active, turn_index)
            batch_size = max(1, int(self.config.rollout_batch_size))
            for start in range(0, len(active), batch_size):
                batch = active[start : start + batch_size]
                batch_prompts = prompts[start : start + batch_size]
                input_ids, attention_mask, output_ids = self._generate(
                    batch_prompts, do_sample=do_sample
                )
                self._prepare_turn_records(
                    batch,
                    batch_prompts,
                    input_ids,
                    attention_mask,
                    output_ids,
                    turn_index,
                    do_sample,
                )
                # Release this batch's generation tensors before the next batch so
                # peak VRAM stays bounded by one rollout batch rather than the whole
                # active set. Matters most on single-GPU Colab runs.
                del input_ids, attention_mask, output_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for trajectory in trajectories:
            trajectory.compute_returns(self.config.gamma)
        assign_group_advantages(trajectories)
        return trajectories


def assign_group_advantages(trajectories: List[Trajectory]) -> None:
    groups: Dict[Tuple[str, int], List[Any]] = {}
    for trajectory in trajectories:
        for step in trajectory.steps:
            groups.setdefault((trajectory.task_id, step.turn_index), []).append(step)
    for steps in groups.values():
        values = [step.discounted_return for step in steps]
        normalized = group_normalize(values)
        for step, advantage in zip(steps, normalized):
            step.advantage = advantage


def build_training_samples(trajectories: List[Trajectory]) -> List[Dict[str, Any]]:
    samples = []
    for trajectory in trajectories:
        for step in trajectory.steps:
            if not step.completion_ids:
                continue
            if len(step.old_logprobs) != len(step.completion_ids):
                continue
            samples.append(
                {
                    "input_ids": list(step.prompt_ids) + list(step.completion_ids),
                    "prompt_length": len(step.prompt_ids),
                    "completion_length": len(step.completion_ids),
                    "old_logprobs": list(step.old_logprobs),
                    "advantage": float(step.advantage),
                    "discounted_return": float(step.discounted_return),
                    "reward": float(step.reward),
                    "task_id": trajectory.task_id,
                    "trajectory_id": trajectory.trajectory_id,
                    "turn_index": step.turn_index,
                }
            )
    return samples
