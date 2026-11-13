from typing import List, Optional, Tuple

import torch


def gather_completion_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lengths: List[int],
    completion_lengths: List[int],
    logit_offset: int = 0,
    chunk_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, _ = logits.shape
    max_completion = max(completion_lengths) if completion_lengths else 0
    token_logprobs = torch.zeros(
        (batch_size, max_completion),
        dtype=torch.float32,
        device=logits.device,
    )
    mask = torch.zeros(
        (batch_size, max_completion),
        dtype=torch.float32,
        device=logits.device,
    )
    for row in range(batch_size):
        prompt_length = int(prompt_lengths[row])
        completion_length = int(completion_lengths[row])
        if completion_length <= 0:
            continue
        start = prompt_length - 1 - logit_offset
        end = start + completion_length
        if start < 0 or end > sequence_length:
            raise ValueError(
                f"Invalid completion span for row {row}: start={start}, end={end}, "
                f"length={sequence_length}, offset={logit_offset}"
            )
        targets = input_ids[row, prompt_length : prompt_length + completion_length]
        for chunk_start in range(0, completion_length, chunk_size):
            chunk_end = min(chunk_start + chunk_size, completion_length)
            chunk_logits = logits[row, start + chunk_start : start + chunk_end, :].float()
            chunk_targets = targets[chunk_start:chunk_end]
            target_logits = chunk_logits.gather(-1, chunk_targets.unsqueeze(-1)).squeeze(-1)
            logsumexp = torch.logsumexp(chunk_logits, dim=-1)
            token_logprobs[row, chunk_start:chunk_end] = target_logits - logsumexp
            mask[row, chunk_start:chunk_end] = 1.0
    return token_logprobs, mask


def grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.28,
    loss_type: str = "dr_grpo",
    l_norm: Optional[float] = None,
) -> torch.Tensor:
    log_ratio = new_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)
    advantage = advantages.unsqueeze(-1)
    surrogate_1 = ratio * advantage
    clamped = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
    surrogate_2 = clamped * advantage
    per_token_loss = -torch.min(surrogate_1, surrogate_2)
    weighted = per_token_loss * mask
    if loss_type == "dr_grpo":
        if l_norm is None or float(l_norm) <= 0.0:
            raise ValueError(
                "dr_grpo loss requires a positive l_norm; got "
                f"{l_norm!r}. Set config.dr_grpo_l_norm to a positive value "
                "instead of relying on sequence-length normalization."
            )
        denominator = float(l_norm)
        return weighted.sum() / denominator
    active = mask.sum().clamp(min=1.0)
    return weighted.sum() / active


def dpo_loss(
    policy_winner_logp: torch.Tensor,
    policy_loser_logp: torch.Tensor,
    reference_winner_logp: torch.Tensor,
    reference_loser_logp: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    policy_margin = policy_winner_logp - policy_loser_logp
    reference_margin = reference_winner_logp - reference_loser_logp
    logits = beta * (policy_margin - reference_margin)
    return -torch.nn.functional.logsigmoid(logits).mean()


def masked_sequence_logprob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lengths: List[int],
    completion_lengths: List[int],
    logit_offset: int = 0,
) -> torch.Tensor:
    token_logprobs, mask = gather_completion_logprobs(
        logits,
        input_ids,
        prompt_lengths,
        completion_lengths,
        logit_offset=logit_offset,
    )
    return (token_logprobs * mask).sum(dim=-1)
