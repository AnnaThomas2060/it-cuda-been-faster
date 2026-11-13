"""Compile, verify, and benchmark stored teacher kernels.

The measured speedups are written to data/teacher_measurements.json, which the
prompt builder uses through data.example_store. Only kernels that compile,
pass numerical verification, and pass the stability gate are recorded.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.example_store import TEACHER_EXAMPLES  # noqa: E402
from engine.benchmark import benchmark_kernel  # noqa: E402
from engine.compiler import compile_kernel_ex  # noqa: E402
from engine.verifier import verify_kernel  # noqa: E402


def evaluate_teacher(
    task_id: str,
    teacher: Dict[str, Any],
    config: Any,
    device: str,
) -> Dict[str, Any]:
    name = teacher["name"]
    compiled = compile_kernel_ex(teacher["code"])
    if compiled.module is None:
        return {
            "name": name,
            "verified": False,
            "status": compiled.status,
            "message": compiled.message,
        }

    verification = verify_kernel(
        compiled.module,
        task_id,
        device=device,
        seeds=list(config.verification_seeds),
        rtol=config.numerical_rtol,
        atol=config.numerical_atol,
    )
    if not verification.correct:
        return {
            "name": name,
            "verified": False,
            "status": verification.status,
            "message": verification.message,
        }

    benchmark = benchmark_kernel(
        compiled.module,
        task_id,
        device=device,
        warmup_iters=config.benchmark_warmup_iters,
        num_blocks=config.benchmark_num_blocks,
        runs_per_block=config.benchmark_runs_per_block,
        variance_threshold=config.variance_threshold,
        min_measurement_ms=config.min_measurement_ms,
        use_bandwidth=config.use_bandwidth,
        fake_gain_ratio=config.fake_gain_ratio,
        max_plausible_speedup=config.max_plausible_speedup,
    )
    return {
        "name": name,
        "verified": bool(benchmark.status == "ok" and benchmark.stable),
        "status": benchmark.status,
        "message": benchmark.message,
        "speedup": benchmark.speedup,
        "variance": benchmark.variance,
        "median_candidate_ms": benchmark.median_cand_ms,
        "median_baseline_ms": benchmark.median_ref_ms,
        "device": benchmark.device_name,
    }


def load_measurements(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Measurement file is not a JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and benchmark teacher kernels.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "multiturn_grpo.yaml"))
    parser.add_argument("--tasks", default=None, help="Comma separated task ids.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "teacher_measurements.json"),
    )
    parser.add_argument("--dry_run", action="store_true", help="Do not write the output file.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any teacher fails verification.",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch is required to verify teachers.")
        return 1
    if not torch.cuda.is_available():
        print("CUDA is required to verify teachers.")
        return 1

    from configs.grpo_config import ExperimentConfig

    config = ExperimentConfig.from_yaml(args.config)
    output_path = Path(args.output)
    measurements = load_measurements(output_path)

    if args.tasks:
        task_ids = [task.strip() for task in args.tasks.split(",") if task.strip()]
    else:
        task_ids = [task_id for task_id, teachers in TEACHER_EXAMPLES.items() if teachers]

    failures = 0
    for task_id in task_ids:
        teachers = TEACHER_EXAMPLES.get(task_id, [])
        if not teachers:
            print(f"[SKIP] {task_id}: no stored teachers")
            continue
        for teacher in teachers:
            result = evaluate_teacher(task_id, teacher, config, args.device)
            if result["verified"]:
                print(
                    f"[PASS] {task_id}/{result['name']}: "
                    f"{result['speedup']:.2f}x variance={result['variance']:.6f}"
                )
                measurements.setdefault(task_id, [])
                measurements[task_id] = [
                    entry for entry in measurements[task_id] if entry["name"] != result["name"]
                ]
                measurements[task_id].append(
                    {
                        "name": result["name"],
                        "speedup": result["speedup"],
                        "variance": result["variance"],
                        "device": result["device"],
                    }
                )
            else:
                failures += 1
                print(
                    f"[FAIL] {task_id}/{result['name']}: "
                    f"{result['status']} {result.get('message', '')}"
                )

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(measurements, handle, indent=2, sort_keys=True)
        print(f"Wrote teacher measurements to {output_path}")

    if args.strict and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
