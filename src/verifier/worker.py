"""Child process that compiles, validates and times one candidate.

Launched only by `src.verifier.harness`; never import this from the
training process. It sees exactly one GPU (`cuda:0`) and writes
`<job>/result.json` after every stage, so if the process segfaults or is
killed, the harness still knows which stage it reached.

Modes:
  full      correctness, then timing (training profile)
  validate  correctness only; run under compute-sanitizer by the harness
  timing    timing only, for an already-validated candidate
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import resource
except ImportError:
    resource = None


def _write(path: Path, result: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False))
    tmp.replace(path)


def _limit_resources(cpu_seconds: int) -> None:
    if resource is None:
        return
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 2**20, 512 * 2**20))


def _time_candidate(torch, task, candidate, config) -> Dict[str, Any]:
    from src.verifier.timing import collect_timing_pairs, denoise_pairs, measure_callable

    reference = task["fn"]
    x = task["input_gen"](task["shapes"][-1], device="cuda:0", seed=config.seed)
    with torch.inference_mode():
        for _ in range(config.warmup_calls):
            reference(x)
            out = candidate(x)
        torch.cuda.synchronize()
        sampled = collect_timing_pairs(
            lambda name: measure_callable(
                torch,
                reference if name == "reference" else candidate,
                (x,),
                config.repeats_per_sample,
            ),
            seed=config.seed,
            window_seconds=config.timing_window_seconds,
            min_pairs=max(config.min_timing_pairs, config.block_count),
            max_pairs=config.max_timing_pairs,
        )
    decision = denoise_pairs(
        sampled["pairs"],
        block_count=config.block_count,
        variance_limit=config.variance_limit,
        suspicious_speedup=config.suspicious_speedup,
    )
    # Minimum DRAM traffic: read the input once, write the output once.
    bytes_moved = (x.numel() + out.numel()) * x.element_size()
    cand_bw = bytes_moved / (decision["median_candidate_ms"] * 1e6)
    ref_bw = bytes_moved / (decision["median_reference_ms"] * 1e6)
    decision["bytes_moved"] = bytes_moved
    decision["candidate_bandwidth_gbps"] = cand_bw
    decision["reference_bandwidth_gbps"] = ref_bw

    from engine.benchmark import peak_memory_bandwidth_gbps

    peak_bw = peak_memory_bandwidth_gbps("cuda:0")
    decision["peak_bandwidth_gbps"] = peak_bw
    if peak_bw and cand_bw > peak_bw * 1.20 and decision["status"] == "accepted":
        decision["status"] = "rejected"
        decision["reason"] = "above_roofline_bandwidth"
        decision["speedup"] = None

    decision["elapsed_seconds"] = sampled["elapsed_seconds"]
    decision["window_complete"] = sampled["window_complete"]
    decision["timing_shape"] = list(x.shape)
    decision["gpu_name"] = torch.cuda.get_device_name(0)
    return decision


def run(job_dir: Path) -> Dict[str, Any]:
    from src.verifier.config import VerifierConfig

    job = json.loads((job_dir / "request.json").read_text())
    config = VerifierConfig(**job["config"])
    mode = job["mode"]
    out_path = job_dir / f"result_{mode}.json"
    result: Dict[str, Any] = {"task_id": job["task_id"], "mode": mode, "stage": "setup"}
    _write(out_path, result)
    _limit_resources(int(config.timeout_seconds))

    from data.tasks import get_task
    from engine.compiler import compile_kernel, extract_code_block, run_static_checks

    task = get_task(job["task_id"])
    code = extract_code_block(job["code"])
    result["source_sha256"] = hashlib.sha256(code.encode()).hexdigest()

    result["stage"] = "static_check"
    ok, message = run_static_checks(code)
    if not ok:
        result.update(status="failed", message=message)
        _write(out_path, result)
        return result

    result["stage"] = "compile"
    _write(out_path, result)
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"worker expects one visible GPU, found {torch.cuda.device_count()}")
    torch.cuda.set_device(0)
    module, compile_log = compile_kernel(
        code,
        module_name=f"cand_{result['source_sha256'][:16]}",
        build_directory=str(job_dir / "build"),
    )
    result["compiler_output"] = compile_log
    if module is None:
        result.update(status="failed", message=compile_log)
        _write(out_path, result)
        return result
    candidate = module.run_cuda

    if mode in ("full", "validate"):
        result["stage"] = "self_check"
        _write(out_path, result)
        from src.verifier.correctness import negative_controls, run_correctness_suite

        controls = negative_controls(torch)
        if not all(controls.values()):
            raise RuntimeError(f"evaluator accepted a known-bad candidate: {controls}")

        result["stage"] = "correctness"
        _write(out_path, result)
        result["correctness"] = run_correctness_suite(torch, task, candidate, config)
        if not result["correctness"]["correct"]:
            result.update(status="failed", message=result["correctness"]["message"])
            _write(out_path, result)
            return result

    if mode in ("full", "timing"):
        result["stage"] = "timing"
        _write(out_path, result)
        result["timing"] = _time_candidate(torch, task, candidate, config)

    result.update(stage="done", status="ok")
    _write(out_path, result)
    return result


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m src.verifier.worker <job_dir>", file=sys.stderr)
        return 2
    job_dir = Path(argv[0])
    try:
        run(job_dir)
    except Exception as exc:  # record, then fail loudly for the harness
        mode = json.loads((job_dir / "request.json").read_text())["mode"]
        path = job_dir / f"result_{mode}.json"
        partial = json.loads(path.read_text()) if path.exists() else {}
        partial.update(status="error", error_type=type(exc).__name__, message=str(exc)[:2000])
        _write(path, partial)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
