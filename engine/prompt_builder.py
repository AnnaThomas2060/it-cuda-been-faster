from typing import Any, Dict, List, Optional

from data.example_store import get_teacher_examples
from data.tasks import get_task
from engine.utils import strip_thoughts

SYSTEM_PROMPT = (
    "You are an expert CUDA systems engineer. You write correct, self-contained CUDA "
    "extensions that match the requested PyTorch reference semantics and optimize for "
    "measured GPU throughput. Return only the requested code block."
)


def _interface_contract(task: Dict[str, Any]) -> str:
    sig = task["c_signature"]
    return (
        "Interface Contract:\n"
        "1. Your code must include the required CUDA kernel declared with __global__.\n"
        f"2. Your host wrapper must match the C++ signature: {sig}.\n"
        "3. Your implementation must allocate its output tensor and launch the CUDA kernel.\n"
        "4. Do not use PyTorch built-in operator shortcuts such as torch::relu, torch::sigmoid, "
        "or at::native.\n"
        "5. Wrap your complete, self-contained C++ and CUDA code inside a single ```cpp "
        "code block.\n"
    )


def _base_prompt(task_id: str) -> str:
    task = get_task(task_id)
    return (
        "You are an expert CUDA systems engineer. Implement a high performance CUDA kernel "
        "for the following operation:\n"
        f"Task: {task['name']}\n"
        f"Description: {task['description']}\n\n"
        f"{_interface_contract(task)}"
    )


def _teacher_context(task_id: str, max_examples: Optional[int] = None) -> str:
    teachers = get_teacher_examples(task_id)
    if not teachers:
        return ""
    if max_examples is not None:
        teachers = teachers[:max_examples]
    context = "\nHistorical Implementations and Measured Speedups:\n"
    for idx, teacher in enumerate(teachers, 1):
        speedup = teacher.get("speedup", 1.0)
        status = "" if teacher.get("verified", False) else " (unverified)"
        context += (
            f"\n--- Implementation {idx} (Measured Speedup: {speedup:.2f}x){status} ---\n"
            f"Notes: {teacher.get('description', '')}\n"
            f"```cpp\n{teacher.get('code', '')}\n```\n"
        )
    context += (
        "\nAnalysis and Synthesis Instructions:\n"
        "Review the historical implementations and their speedups above. Identify memory "
        "bottlenecks, instruction inefficiencies, or latency limits in the earlier versions. "
        "Synthesize an improved implementation that achieves even higher throughput while "
        "preserving strict numerical equivalence."
    )
    return context


def _history_context(history: Optional[List[Dict[str, Any]]]) -> str:
    if not history:
        return ""
    lines = ["Previous turn summaries:"]
    for item in history:
        turn = item.get("turn_index", 0)
        status = item.get("status", "unknown")
        feedback = strip_thoughts(str(item.get("feedback", ""))).strip()
        lines.append(f"- Turn {turn}: {status}. {feedback}".strip())
    return "\n".join(lines)


def build_prompt(task_id: str, mode: str = "reference_only") -> str:
    base_prompt = _base_prompt(task_id)
    if mode == "reference_only":
        return base_prompt + (
            "\nWrite an optimized, numerically accurate CUDA implementation that "
            "maximizes throughput."
        )
    if mode in ("contrastive", "scored_context"):
        teacher_context = _teacher_context(task_id)
        if not teacher_context:
            return base_prompt + (
                "\nWrite an optimized, numerically accurate CUDA implementation that "
                "maximizes throughput."
            )
        return base_prompt + teacher_context
    raise ValueError(f"Unknown prompt mode: {mode}. Expected 'reference_only' or 'contrastive'.")


def build_turn_messages(
    task_id: str,
    mode: str = "reference_only",
    turn_index: int = 0,
    best_code: Optional[str] = None,
    feedback: Optional[str] = None,
    speedup: Optional[float] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    max_teacher_examples: Optional[int] = None,
) -> List[Dict[str, str]]:
    base_prompt = _base_prompt(task_id)
    if mode == "reference_only":
        user_prompt = base_prompt + (
            "\nWrite an optimized, numerically accurate CUDA implementation that "
            "maximizes throughput."
        )
    elif mode in ("contrastive", "scored_context"):
        teacher_context = _teacher_context(task_id, max_examples=max_teacher_examples)
        if teacher_context:
            user_prompt = base_prompt + teacher_context
        else:
            user_prompt = base_prompt + (
                "\nWrite an optimized, numerically accurate CUDA implementation that "
                "maximizes throughput."
            )
    else:
        raise ValueError(
            f"Unknown prompt mode: {mode}. Expected 'reference_only' or 'contrastive'."
        )

    if turn_index > 0:
        state_lines = [f"\nMulti-turn refinement state (turn {turn_index + 1}):"]
        if best_code:
            state_lines.append(f"\nCurrent best implementation:\n```cpp\n{best_code}\n```")
        if feedback:
            state_lines.append(f"\nCompiler and verifier feedback:\n{strip_thoughts(feedback)}")
        if speedup is not None:
            state_lines.append(
                f"\nMeasured speedup of the current best implementation: {speedup:.2f}x"
            )
        history_text = _history_context(history)
        if history_text:
            state_lines.append("\n" + history_text)
        state_lines.append(
            "\nProduce an improved implementation. Fix any correctness problems first, then "
            "reduce memory traffic, launch overhead, and instruction count. Do not repeat the "
            "previous code verbatim."
        )
        user_prompt += "\n".join(state_lines)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def format_chat_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    parts = []
    for message in messages:
        parts.append(f"{message['role'].upper()}: {message['content']}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)
