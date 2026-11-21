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
        "6. Launch kernels on the active PyTorch CUDA stream (include c10/cuda/CUDAStream.h "
        "and pass c10::cuda::getCurrentCUDAStream(x.device().index())), use contiguous inputs "
        "(x.contiguous()), and guard empty tensors (numel == 0).\n"
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


def select_scored_examples(
    task_id: str,
    max_examples: Optional[int] = None,
    archive: Optional[Any] = None,
    allow_holdout_teachers: bool = True,
) -> List[Dict[str, Any]]:
    task = get_task(task_id)
    if not allow_holdout_teachers and not task.get("is_training", True):
        return []

    examples: List[Dict[str, Any]] = []
    seen_codes = set()

    if archive is not None and task_id in getattr(archive, "entries", {}):
        bucket = [e for e in archive.entries[task_id] if e.correct]
        if bucket:
            bucket.sort(key=lambda e: e.speedup)
            # Pick from distinct score buckets (lowest correct and highest correct).
            picks = [bucket[0]] if len(bucket) == 1 else [bucket[0], bucket[-1]]
            for entry in picks:
                code = entry.code.strip()
                if code and code not in seen_codes:
                    seen_codes.add(code)
                    examples.append(
                        {
                            "name": f"archive_turn_{entry.turn_index}",
                            "speedup": entry.speedup,
                            "verified": True,
                            "description": (
                                "Previously verified candidate from distinct score bucket."
                            ),
                            "code": code,
                        }
                    )

    for teacher in get_teacher_examples(task_id):
        code = str(teacher.get("code", "")).strip()
        if code and code not in seen_codes:
            seen_codes.add(code)
            examples.append(teacher)

    if max_examples is not None:
        examples = examples[:max_examples]
    return examples


def _teacher_context(
    task_id: str,
    max_examples: Optional[int] = None,
    archive: Optional[Any] = None,
    allow_holdout_teachers: bool = True,
    include_scores: bool = True,
) -> str:
    teachers = select_scored_examples(
        task_id,
        max_examples=max_examples,
        archive=archive,
        allow_holdout_teachers=allow_holdout_teachers,
    )
    if not teachers:
        return ""
    header = (
        "\nHistorical Implementations and Measured Speedups:\n"
        if include_scores
        else "\nHistorical Implementations:\n"
    )
    context = header
    for idx, teacher in enumerate(teachers, 1):
        if include_scores:
            speedup = teacher.get("speedup", 1.0)
            status = "" if teacher.get("verified", False) else " (unverified)"
            label = f"--- Implementation {idx} (Measured Speedup: {speedup:.2f}x){status} ---"
        else:
            label = f"--- Implementation {idx} ---"
        context += (
            f"\n{label}\n"
            f"Notes: {teacher.get('description', '')}\n"
            f"```cpp\n{teacher.get('code', '')}\n```\n"
        )
    if include_scores:
        context += (
            "\nAnalysis and Synthesis Instructions:\n"
            "Review the historical implementations and their speedups above. Identify memory "
            "bottlenecks, instruction inefficiencies, or latency limits in the earlier versions. "
            "Synthesize an improved implementation that achieves even higher throughput while "
            "preserving strict numerical equivalence."
        )
    else:
        context += (
            "\nAnalysis and Synthesis Instructions:\n"
            "Review the historical implementations above and synthesize an improved "
            "implementation that achieves higher throughput while preserving strict "
            "numerical equivalence."
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
    if mode in ("contrastive", "scored_context", "unscored_context"):
        teacher_context = _teacher_context(
            task_id,
            include_scores=(mode != "unscored_context"),
        )
        if not teacher_context:
            return base_prompt + (
                "\nWrite an optimized, numerically accurate CUDA implementation that "
                "maximizes throughput."
            )
        return base_prompt + teacher_context
    raise ValueError(
        f"Unknown prompt mode: {mode}. "
        "Expected 'reference_only', 'unscored_context', or 'contrastive'."
    )


def build_turn_messages(
    task_id: str,
    mode: str = "reference_only",
    turn_index: int = 0,
    best_code: Optional[str] = None,
    feedback: Optional[str] = None,
    speedup: Optional[float] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    max_teacher_examples: Optional[int] = None,
    archive: Optional[Any] = None,
    allow_holdout_teachers: bool = True,
) -> List[Dict[str, str]]:
    base_prompt = _base_prompt(task_id)
    if mode == "reference_only":
        user_prompt = base_prompt + (
            "\nWrite an optimized, numerically accurate CUDA implementation that "
            "maximizes throughput."
        )
    elif mode in ("contrastive", "scored_context", "unscored_context"):
        teacher_context = _teacher_context(
            task_id,
            max_examples=max_teacher_examples,
            archive=archive,
            allow_holdout_teachers=allow_holdout_teachers,
            include_scores=(mode != "unscored_context"),
        )
        if teacher_context:
            user_prompt = base_prompt + teacher_context
        else:
            user_prompt = base_prompt + (
                "\nWrite an optimized, numerically accurate CUDA implementation that "
                "maximizes throughput."
            )
    else:
        raise ValueError(
            f"Unknown prompt mode: {mode}. "
            "Expected 'reference_only', 'unscored_context', or 'contrastive'."
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
