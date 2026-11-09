from data.example_store import get_teacher_examples
from data.tasks import get_task

# Generates prompts for reference only and scored context contrastive RL training arms.
# In contrastive mode, previous implementations with measured speedups are included in the context.
# Single teacher tasks are handled gracefully by including only the available reference.

def build_prompt(task_id: str, mode: str = "reference_only") -> str:
    task = get_task(task_id)
    name = task["name"]
    desc = task["description"]
    sig = task["c_signature"]

    base_prompt = (
        "You are an expert CUDA systems engineer. Implement a high performance CUDA kernel "
        f"for the following operation:\n"
        f"Task: {name}\n"
        f"Description: {desc}\n\n"
        f"Interface Contract:\n"
        f"1. Your code must include the required CUDA kernel declared with __global__.\n"
        f"2. Your host wrapper must match the C++ signature: {sig}.\n"
        f"3. Your implementation must allocate its output tensor and launch the CUDA kernel.\n"
        f"4. Do not use PyTorch built-in operator shortcuts such as torch::relu, torch::sigmoid, "
        f"or at::native.\n"
        "5. Wrap your complete, self-contained C++ and CUDA code inside a single ```cpp "
        "code block.\n"
    )

    if mode == "reference_only":
        base_prompt += (
            "\nWrite an optimized, numerically accurate CUDA implementation that "
            "maximizes throughput."
        )
        return base_prompt

    if mode in ("contrastive", "scored_context"):
        teachers = get_teacher_examples(task_id)
        if not teachers:
            base_prompt += (
                "\nWrite an optimized, numerically accurate CUDA implementation that "
                "maximizes throughput."
            )
            return base_prompt

        base_prompt += "\nHistorical Implementations and Measured Speedups:\n"
        for idx, teacher in enumerate(teachers, 1):
            speedup = teacher.get("speedup", 1.0)
            t_desc = teacher.get("description", "")
            t_code = teacher.get("code", "")
            base_prompt += (
                f"\n--- Implementation {idx} (Measured Speedup: {speedup:.2f}x) ---\n"
                f"Notes: {t_desc}\n"
                f"```cpp\n{t_code}\n```\n"
            )

        base_prompt += (
            "\nAnalysis and Synthesis Instructions:\n"
            "Review the historical implementations and their speedups above. Identify memory "
            "bottlenecks, instruction inefficiencies, or latency limits in the earlier versions. "
            "Synthesize an improved implementation that achieves even higher throughput while "
            "preserving strict numerical equivalence."
        )
        return base_prompt

    raise ValueError(f"Unknown prompt mode: {mode}. Expected 'reference_only' or 'contrastive'.")
