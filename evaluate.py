import argparse
import json
import os
import statistics
import time
from typing import Any, Dict, List

from configs.grpo_config import ExperimentConfig
from engine.prompt_builder import build_prompt
from models.loader import load_model_and_tokenizer
from src.contracts import KernelEvalRequest
from src.verifier import KernelVerifier, VerifierConfig

# Evaluates trained policy checkpoints across training and holdout evaluation tasks.
# Uses greedy decoding without multi-turn repair to measure zero-shot kernel synthesis capability.
# Aggregates coverage, correctness rates, and speedup percentiles for statistical reporting.

def evaluate_checkpoint(
    model: Any,
    tokenizer: Any,
    task_keys: List[str],
    prompt_mode: str = "reference_only",
    verifier: Any = None,
    device_id: int = 0,
) -> Dict[str, Any]:
    import torch

    verifier = verifier or KernelVerifier(VerifierConfig.evaluation())

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

        # Compile, verify and time in an isolated worker (strict protocol).
        response = verifier.evaluate(
            KernelEvalRequest(task_id=tid, code=completion_text, device_id=device_id)
        )
        is_compiled = response.compiled
        is_numerically_correct = response.correct
        is_stable = response.failure_stage is None and response.speedup is not None
        speedup = response.speedup if is_stable else 0.0
        compile_msg = response.compiler_output
        num_msg = response.error_message or "ok"

        if is_numerically_correct:
            correct_count += 1
        if is_stable:
            stable_speedups.append(speedup)
            fast_1_0_count += speedup >= 1.0
            fast_1_1_count += speedup >= 1.1

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
            "verifier": response.model_dump(exclude={"compiler_output"}),
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
    parser.add_argument("--verifier_device", type=int, default=0,
                        help="GPU ordinal used to compile, validate and time kernels.")
    parser.add_argument("--verification_device", type=int, default=None,
                        help="Second GPU that must reproduce any speedup above 3x.")
    parser.add_argument("--no_memcheck", action="store_true",
                        help="Skip Compute Sanitizer (faster; not for reported results).")
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
        verifier=KernelVerifier(VerifierConfig.evaluation(
            memcheck=not args.no_memcheck,
            verification_device_id=args.verification_device,
        )),
        device_id=args.verifier_device,
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
