import argparse
import json
import os
import statistics
import time
from typing import Any, Dict, List

from configs.grpo_config import ExperimentConfig
from engine.benchmark import measure_speedup
from engine.compiler import compile_kernel
from engine.prompt_builder import build_prompt
from engine.verifier import verify_numerical_correctness
from models.loader import load_model_and_tokenizer

# Evaluates trained policy checkpoints across training and holdout evaluation tasks.
# Uses greedy decoding without multi-turn repair to measure zero-shot kernel synthesis capability.
# Aggregates coverage, correctness rates, and speedup percentiles for statistical reporting.

def evaluate_checkpoint(
    model: Any,
    tokenizer: Any,
    task_keys: List[str],
    prompt_mode: str = "reference_only",
) -> Dict[str, Any]:
    import torch

    results_per_task: List[Dict[str, Any]] = []
    correct_count = 0
    fast_1_0_count = 0
    fast_1_1_count = 0
    stable_speedups: List[float] = []

    model.eval()

    # Determine execution device safely
    target_device = getattr(model, "device", None)
    if target_device is None:
        try:
            target_device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for tid in task_keys:
        prompt_text = build_prompt(tid, mode=prompt_mode)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(target_device)

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed_gen_time = time.perf_counter() - start_time

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        num_generated_tokens = len(gen_tokens)
        completion_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # Stage 1: Compilation
        compiled_module, compile_msg = compile_kernel(completion_text)
        is_compiled = compiled_module is not None

        # Stage 2: Numerical Verification
        is_numerically_correct = False
        num_msg = "Not compiled"
        if is_compiled:
            is_numerically_correct, num_msg = verify_numerical_correctness(compiled_module, tid)

        if is_numerically_correct:
            correct_count += 1

        # Stage 3: Latency Benchmarking
        speedup = 0.0
        is_stable = False
        if is_numerically_correct:
            speedup, is_stable = measure_speedup(compiled_module, tid, device="cuda")
            if is_stable:
                stable_speedups.append(speedup)
                if speedup >= 1.0:
                    fast_1_0_count += 1
                if speedup >= 1.1:
                    fast_1_1_count += 1

        task_record = {
            "task_id": tid,
            "compiled": is_compiled,
            "compile_msg": compile_msg,
            "numerically_correct": is_numerically_correct,
            "verification_msg": num_msg,
            "speedup": speedup,
            "is_stable": is_stable,
            "tokens_generated": num_generated_tokens,
            "generation_time_sec": round(elapsed_gen_time, 3),
            "tokens_per_sec": round(num_generated_tokens / max(elapsed_gen_time, 1e-4), 2),
            "completion": completion_text,
        }
        results_per_task.append(task_record)

    total_tasks = len(task_keys)
    correctness_rate = correct_count / total_tasks if total_tasks > 0 else 0.0
    fast_1_0_rate = fast_1_0_count / total_tasks if total_tasks > 0 else 0.0
    fast_1_1_rate = fast_1_1_count / total_tasks if total_tasks > 0 else 0.0
    median_speedup = statistics.median(stable_speedups) if stable_speedups else 0.0

    summary = {
        "total_tasks": total_tasks,
        "correct_count": correct_count,
        "correctness_rate": round(correctness_rate, 4),
        "fast_1_0_rate": round(fast_1_0_rate, 4),
        "fast_1_1_rate": round(fast_1_1_rate, 4),
        "median_speedup": round(median_speedup, 2),
        "tasks": results_per_task,
    }

    return summary

def main():
    parser = argparse.ArgumentParser(description="Greedy evaluation of CUDA generation policy.")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to trained PEFT LoRA adapter checkpoint.")
    parser.add_argument("--prompt_mode", type=str, default="reference_only",
                        choices=["reference_only", "contrastive"],
                        help="Prompt construction mode during evaluation.")
    parser.add_argument("--use_dev_model", action="store_true",
                        help="Use Qwen2.5-Coder-7B-Instruct instead of 30B MoE.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="./outputs/eval_results/eval_summary.json",
        help="File path to save JSON evaluation metrics.",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig()
    cfg.use_dev_model = args.use_dev_model

    model, tokenizer = load_model_and_tokenizer(config=cfg, is_trainable=False)

    if args.adapter_path and os.path.exists(args.adapter_path):
        from peft import PeftModel
        print(f"Loading adapter weights from: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    summary = evaluate_checkpoint(
        model=model,
        tokenizer=tokenizer,
        task_keys=cfg.eval_tasks,
        prompt_mode=args.prompt_mode,
    )

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nEvaluation Summary:")
    print(f"Total tasks evaluated: {summary['total_tasks']}")
    print(f"Correctness rate:      {summary['correctness_rate'] * 100:.2f}%")
    print(f"Fast >= 1.0x rate:     {summary['fast_1_0_rate'] * 100:.2f}%")
    print(f"Fast >= 1.1x rate:     {summary['fast_1_1_rate'] * 100:.2f}%")
    print(f"Median speedup:        {summary['median_speedup']:.2f}x")
    print(f"Results saved to:      {args.output_file}")

if __name__ == "__main__":
    main()
