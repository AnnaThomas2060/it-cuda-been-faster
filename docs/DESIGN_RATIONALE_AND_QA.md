# Design Rationale and Presentation Q&A Reference

This note documents the main methodological decisions in the RL training and evaluation pipeline so they can be cited directly during project presentation Q&A.

## 1. Why Dr. GRPO (`l_norm`) Instead of Token-Mean GRPO
Standard token-level GRPO divides each sequence's loss by its completion length. That creates a systematic length bias: long failed completions have smaller per-token gradients, while short completions dominate updates. Custom CUDA kernels vary from roughly 30 lines (elementwise ops) to over 120 lines (shared-memory reductions). Dividing by a fixed constant `dr_grpo_l_norm` removes sequence-length bias across tasks and turns.

## 2. Why `beta = 0.0` in GRPO
Compiler syntax checks, numerical equivalence across 1,000+ cases, non-default stream checks, and Compute Sanitizer already prevent linguistic or semantic reward hacking. A token-level KL penalty toward the base model penalizes low-level CUDA idioms (warp shuffles, `float4` pointer casts, shared-memory tiling) that have low prior probability under a general PyTorch-heavy coder model. When archive pairwise DPO is active (`pairwise_margin_weight > 0`), `pairwise_beta = 0.10` still regularizes preference margins against the frozen `sft` adapter.

## 3. Why Shaped Reward During Training vs. Raw Speedup at Evaluation
Kernel speedups are heavily right-skewed: a few easy operations can reach high ratios and dominate policy gradients. The training reward uses `0.30 + log1p(max(speedup - 1.0, 0.0)) + 0.10 * bandwidth_delta` (clipped to 2.0) to give immediate signal once a candidate compiles, passes correctness, and times stably, while compressing extreme speedup outliers. All evaluation tables (`evaluate.py`, `compare_arms.py`) report unshaped, toward-1.0 floored speedups, `fast_1.0`, `fast_1.1`, correctness rate, and unresolved timings separately.

## 4. Why 4-bit NF4 LoRA (`r=8`) and AdamW Instead of SinkGD
Full-parameter BF16 training of `Qwen3-Coder-30B-A3B-Instruct` exceeds 2x 24 GiB RTX 3090 VRAM before activations or optimizer states. With 4-bit NF4 weights and rank-8 attention LoRA (~6.7M trainable parameters), AdamW optimizer states occupy roughly 54 MiB (<0.3% of VRAM). The real memory bottleneck is rollout logit materialization (`[batch, seq_len, vocab]`), which is resolved by row-micro-batched logprob evaluation in `RolloutManager._completion_logprobs`.

## 5. Why `reference_only` Evaluation Plus `unscored_context` and `contrastive` Diagnostics
Evaluating trained Arm A and Arm B checkpoints on identical `reference_only` prompts tests whether contrastive conditioning during RL improves general kernel synthesis on unseen holdout operators where no reference CUDA kernel or speedup score is available at inference time. Running frozen-checkpoint diagnostics on training tasks across `reference_only`, `unscored_context` (implementations without scores), and `contrastive` (implementations with scores) separates prompting effects from weight updates and isolates the value of numerical speedup scores.

## 6. Why Equal-Reward Groups Skip Optimizer Steps
When every sample in a group fails or receives the same reward, GRPO advantages are identically zero. Stepping AdamW on zero policy gradients would still apply `weight_decay`, shrinking LoRA weights without reward signal. `MultiTurnGRPOTrainer._train_samples` skips zero-advantage micro-batches when no contrastive archive update is active and logs both `rollout_steps` and actual `optimizer_steps`.
