"""Benchmark PyTorch eager, torch.compile, and verified teacher kernels."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.example_store import get_teacher_examples  # noqa: E402
from data.tasks import get_task, list_task_ids, task_io_bytes  # noqa: E402
from src.verifier.timing import (  # noqa: E402
    collect_timing_pairs,
    denoise_pairs,
    measure_callable,
)


def benchmark_torch_compile(
    task_id: str,
    device: str = "cuda:0",
    warmup_calls: int = 10,
    window_seconds: float = 3.0,
    repeats_per_sample: int = 20,
    seed: int = 635,
) -> Dict[str, Any]:
    import torch

    task = get_task(task_id)
    ref_fn = task["fn"]
    shape = task["shapes"][-1]
    x = task["input_gen"](shape, device=device, seed=seed)
    io_bytes = task_io_bytes(task_id, shape)

    # Always measure eager latency for reference reporting.
    with torch.inference_mode():
        for _ in range(warmup_calls):
            ref_fn(x)
        torch.cuda.synchronize(device)
        eager_sample = measure_callable(torch, ref_fn, (x,), repeats_per_sample)

    eager_ms = eager_sample["event_ms"]
    eager_bw = io_bytes / (eager_ms * 1e6) if eager_ms > 0 else 0.0

    entry: Dict[str, Any] = {
        "task_id": task_id,
        "shape": list(shape),
        "eager_ms": round(eager_ms, 4),
        "eager_bandwidth_gbps": round(eager_bw, 2),
        "torch_compile_status": "unavailable",
        "torch_compile_speedup": None,
        "torch_compile_ms": None,
    }

    try:
        compiled_fn = torch.compile(ref_fn)
        with torch.inference_mode():
            for _ in range(warmup_calls):
                ref_fn(x)
                compiled_fn(x)
            torch.cuda.synchronize(device)
            sampled = collect_timing_pairs(
                lambda name: measure_callable(
                    torch,
                    ref_fn if name == "reference" else compiled_fn,
                    (x,),
                    repeats_per_sample,
                ),
                seed=seed,
                window_seconds=window_seconds,
                min_pairs=7,
                max_pairs=5000,
            )
        decision = denoise_pairs(sampled["pairs"])
        entry["torch_compile_status"] = decision["status"]
        entry["torch_compile_speedup"] = decision["speedup"]
        entry["torch_compile_ms"] = round(float(decision["median_candidate_ms"]), 4)
    except Exception as exc:
        entry["torch_compile_status"] = "unavailable"
        entry["torch_compile_error"] = str(exc)[:300]

    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark eager, torch.compile, and teachers.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--window_seconds", type=float, default=3.0)
    parser.add_argument("--output", default="./outputs/eval_results/hardware_baselines.json")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch is required.")
        return 1
    if not torch.cuda.is_available():
        print("CUDA is required.")
        return 1

    task_ids: List[str] = (
        [t.strip() for t in args.tasks.split(",") if t.strip()]
        if args.tasks
        else list_task_ids(training_only=False)
    )

    torch.cuda.set_device(args.device)
    results: List[Dict[str, Any]] = []
    for task_id in task_ids:
        row = benchmark_torch_compile(
            task_id,
            device=args.device,
            window_seconds=args.window_seconds,
        )
        teachers = get_teacher_examples(task_id)
        row["teachers"] = [
            {
                "name": t["name"],
                "speedup": t.get("speedup"),
                "verified": bool(t.get("verified", False)),
            }
            for t in teachers
        ]
        results.append(row)
        print(
            f"{task_id:<20} eager={row['eager_ms']:.4f}ms "
            f"compile={row['torch_compile_speedup']} ({row['torch_compile_status']})"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "device": torch.cuda.get_device_name(args.device),
                "tasks": results,
            },
            handle,
            indent=2,
        )
    print(f"Wrote baselines to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
