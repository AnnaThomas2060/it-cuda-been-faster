import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from engine.benchmark import benchmark_kernel  # noqa: E402
from engine.compiler import compile_kernel_ex  # noqa: E402
from engine.sanitizer import run_compute_sanitizer  # noqa: E402
from engine.verifier import verify_kernel  # noqa: E402


def _set_resource_limits(timeout_sec: float) -> None:
    try:
        import resource

        cpu_limit = max(1, int(timeout_sec * 2))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 5))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024**3, 1024**3))
    except Exception:
        pass


def evaluate_request(payload: Dict[str, Any], device: str, config: Any) -> Dict[str, Any]:
    task_id = payload["task_id"]
    code = payload["code"]
    turn_index = int(payload.get("turn_index", 0))
    run_sanitizer = bool(payload.get("run_sanitizer", False))

    compile_started = time.perf_counter()
    compiled = compile_kernel_ex(code)
    compile_time = time.perf_counter() - compile_started
    compiler_output = compiled.message
    if compiled.log:
        compiler_output += "\n" + compiled.log

    base = {
        "task_id": task_id,
        "compiled": compiled.module is not None,
        "correct": False,
        "speedup": None,
        "execution_time_ms": None,
        "baseline_time_ms": None,
        "memory_bandwidth_gbps": None,
        "baseline_bandwidth_gbps": None,
        "bandwidth_delta": 0.0,
        "compiler_output": compiler_output[:8000],
        "error_message": None,
        "status": compiled.status,
        "stable": False,
        "variance": 0.0,
        "compile_time_sec": compile_time,
        "benchmark_time_sec": 0.0,
        "candidate_code_hash": compiled.code_hash,
        "device_name": "",
        "metadata": {"turn_index": turn_index, "module_name": compiled.module_name},
    }

    if compiled.module is None:
        base["error_message"] = compiled.message
        return base

    verification = verify_kernel(
        compiled.module,
        task_id,
        device=device,
        seeds=getattr(config, "verification_seeds", None),
        rtol=getattr(config, "numerical_rtol", None),
        atol=getattr(config, "numerical_atol", None),
    )
    base["metadata"]["verification"] = {
        "status": verification.status,
        "max_diff": verification.max_diff,
        "input_mutated": verification.input_mutated,
        "output_aliases_input": verification.output_aliases_input,
        "checked_shapes": [list(shape) for shape in verification.checked_shapes],
        "checked_seeds": verification.checked_seeds,
    }
    if not verification.correct:
        base["correct"] = False
        base["status"] = verification.status
        base["error_message"] = verification.message
        return base

    base["correct"] = True

    if run_sanitizer:
        sanitizer_ok, sanitizer_message = run_compute_sanitizer(
            code=code,
            task_id=task_id,
            device=device,
            timeout_sec=float(getattr(config, "sanitizer_timeout_sec", 120)),
        )
        base["metadata"]["sanitizer"] = sanitizer_message[:4000]
        if not sanitizer_ok:
            base["status"] = "sanitizer_error"
            base["error_message"] = sanitizer_message[:2000]
            return base

    benchmark_started = time.perf_counter()
    benchmark = benchmark_kernel(
        compiled.module,
        task_id,
        device=device,
        warmup_iters=int(getattr(config, "benchmark_warmup_iters", 10)),
        num_blocks=int(getattr(config, "benchmark_num_blocks", 7)),
        runs_per_block=int(getattr(config, "benchmark_runs_per_block", 10)),
        variance_threshold=float(getattr(config, "variance_threshold", 0.005)),
        min_measurement_ms=float(getattr(config, "min_measurement_ms", 0.005)),
        use_bandwidth=bool(getattr(config, "use_bandwidth", True)),
        fake_gain_ratio=float(getattr(config, "fake_gain_ratio", 1.20)),
        max_plausible_speedup=float(getattr(config, "max_plausible_speedup", 10.0)),
    )
    base["benchmark_time_sec"] = time.perf_counter() - benchmark_started
    base.update(
        {
            "speedup": benchmark.speedup,
            "execution_time_ms": benchmark.median_cand_ms,
            "baseline_time_ms": benchmark.median_ref_ms,
            "memory_bandwidth_gbps": benchmark.candidate_bandwidth_gbps,
            "baseline_bandwidth_gbps": benchmark.baseline_bandwidth_gbps,
            "bandwidth_delta": benchmark.bandwidth_delta,
            "status": benchmark.status,
            "stable": benchmark.stable,
            "variance": benchmark.variance,
            "device_name": benchmark.device_name,
        }
    )
    base["metadata"]["benchmark"] = benchmark.metadata
    if benchmark.status != "ok":
        base["error_message"] = benchmark.message
    return base


def worker_loop(conn: Any, device: str, config_dict: Dict[str, Any]) -> None:
    sys.path.insert(0, REPO_ROOT)
    config = SimpleNamespace(**config_dict)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if getattr(config, "torch_extensions_dir", None):
        os.environ["TORCH_EXTENSIONS_DIR"] = str(config.torch_extensions_dir)

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            max_memory = getattr(config, "max_worker_memory_gb", None)
            if max_memory:
                total = torch.cuda.get_device_properties(device).total_memory
                fraction = min(1.0, (float(max_memory) * 1024**3) / float(total))
                torch.cuda.set_per_process_memory_fraction(fraction, device)
        torch.set_num_threads(2)
    except Exception:
        pass

    _set_resource_limits(float(getattr(config, "benchmark_timeout_sec", 180)))

    while True:
        try:
            payload = conn.recv()
        except EOFError:
            break
        if payload is None:
            break
        try:
            response = evaluate_request(payload, device, config)
        except Exception:
            response = {
                "task_id": payload.get("task_id", ""),
                "compiled": False,
                "correct": False,
                "status": "worker_error",
                "stable": False,
                "error_message": traceback.format_exc()[-4000:],
                "compiler_output": "",
                "metadata": {},
            }
        try:
            conn.send(response)
        except (BrokenPipeError, EOFError):
            break
    try:
        conn.close()
    except Exception:
        pass
