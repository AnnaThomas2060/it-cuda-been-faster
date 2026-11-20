"""Process-isolated kernel verifier implementing the team contract.

    verifier = KernelVerifier(VerifierConfig.training())
    response = verifier.evaluate(KernelEvalRequest(task_id="swish", code=completion))

Each candidate runs in a fresh child process (`src.verifier.worker`) in its
own process group, with a hard timeout and resource limits. A candidate that
segfaults, hangs, or corrupts the CUDA context ends only that child, never
the GRPO training process.

With `memcheck=True` correctness runs under Compute Sanitizer and timing runs
in a second, uninstrumented process that reuses the same build.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.compiler import extract_code_block, run_static_checks
from src.contracts import KernelEvalRequest, KernelEvalResponse
from src.verifier.config import VerifierConfig
from src.verifier.timing import ACCEPTED, VERIFICATION_REQUIRED, confirm_on_second_gpu

REPO_ROOT = Path(__file__).resolve().parents[2]
SANITIZER_EXIT_CODE = 86


def sanitizer_passed(returncode: int, log: str) -> bool:
    """Memcheck passes only on a clean exit with every ERROR SUMMARY at zero."""
    summaries = re.findall(r"ERROR SUMMARY: (\d+) errors?\b", log)
    return returncode == 0 and bool(summaries) and all(int(n) == 0 for n in summaries)


def physical_device(device_id: int, visible: Optional[str]) -> str:
    """Map a contract device ordinal through the parent's CUDA_VISIBLE_DEVICES."""
    if not visible:
        return str(device_id)
    entries = [d.strip() for d in visible.split(",") if d.strip()]
    if not 0 <= device_id < len(entries):
        raise ValueError(f"device_id {device_id} not in CUDA_VISIBLE_DEVICES={visible}")
    return entries[device_id]


class KernelVerifier:
    def __init__(
        self,
        config: Optional[VerifierConfig] = None,
        work_root: Optional[str] = None,
        keep_jobs: bool = False,
        python: str = sys.executable,
    ) -> None:
        self.config = config or VerifierConfig.training()
        self.work_root = Path(work_root) if work_root else None
        self.keep_jobs = keep_jobs
        self.python = python

    # ------------------------------------------------------------------ public

    def evaluate(self, request: KernelEvalRequest) -> KernelEvalResponse:
        if self.work_root:
            self.work_root.mkdir(parents=True, exist_ok=True)
        job_dir = Path(tempfile.mkdtemp(prefix=f"{request.task_id}_", dir=self.work_root))
        try:
            return self._evaluate(request, job_dir)
        finally:
            if not self.keep_jobs:
                shutil.rmtree(job_dir, ignore_errors=True)

    # ----------------------------------------------------------------- private

    def _evaluate(self, request: KernelEvalRequest, job_dir: Path) -> KernelEvalResponse:
        cfg = self.config
        timeout = max(request.timeout_seconds, cfg.timeout_seconds)
        extra: Dict[str, Any] = {"job_dir": str(job_dir) if self.keep_jobs else None}

        # Cheap text checks in-process: no point starting a GPU worker for them.
        ok, reason = run_static_checks(extract_code_block(request.code))
        if not ok:
            return KernelEvalResponse(
                task_id=request.task_id,
                compiled=False,
                correct=False,
                status="static_failed",
                stable=False,
                error_message=reason,
                failure_stage="static_check",
            )

        if cfg.memcheck:
            result, info = self._launch(request, job_dir, "validate", request.device_id, timeout)
            extra["memcheck_passed"] = info["sanitizer_passed"]
            if result.get("status") == "ok" and not info["sanitizer_passed"]:
                return self._response(request, result, info, extra, failure="memcheck")
            if result.get("status") == "ok":
                timing, info = self._launch(request, job_dir, "timing", request.device_id, timeout)
                result = {**result, **timing, "correctness": result.get("correctness")}
        else:
            result, info = self._launch(request, job_dir, "full", request.device_id, timeout)

        timing = result.get("timing")
        if timing and timing["status"] == VERIFICATION_REQUIRED:
            confirm_device = (
                cfg.verification_device_id
                if (
                    cfg.verification_device_id is not None
                    and cfg.verification_device_id != request.device_id
                )
                else request.device_id
            )
            second, _ = self._launch(request, job_dir, "timing", confirm_device, timeout)
            if second.get("timing"):
                result["timing"] = confirm_on_second_gpu(
                    timing, second["timing"], cfg.cross_gpu_tolerance
                )
        return self._response(request, result, info, extra)

    def _launch(
        self, request: KernelEvalRequest, job_dir: Path, mode: str, device_id: int, timeout: float
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        (job_dir / "request.json").write_text(
            json.dumps(
                {
                    "task_id": request.task_id,
                    "code": request.code,
                    "mode": mode,
                    "config": self.config.to_dict(),
                }
            )
        )
        command: List[str] = [self.python, "-m", "src.verifier.worker", str(job_dir)]
        if mode == "validate":
            command = [
                self.config.compute_sanitizer,
                "--tool", "memcheck",
                "--error-exitcode", str(SANITIZER_EXIT_CODE),
                "--target-processes", "application-only",
            ] + command

        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": physical_device(
                device_id, os.environ.get("CUDA_VISIBLE_DEVICES")
            ),
            "PYTHONPATH": os.pathsep.join(
                filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])
            ),
            "MAX_JOBS": os.environ.get("MAX_JOBS", "4"),
        }
        started = time.monotonic()
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,  # own process group, so nvcc children die too
        )
        timed_out = False
        try:
            log, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            log, _ = proc.communicate()

        result_path = job_dir / f"result_{mode}.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        info = {
            "mode": mode,
            "returncode": proc.returncode,
            "timed_out": timed_out,
            "seconds": time.monotonic() - started,
            "sanitizer_passed": (
                sanitizer_passed(proc.returncode, log) if mode == "validate" else None
            ),
            "log_tail": log[-4000:],
        }
        return result, info

    @staticmethod
    def _response(
        request: KernelEvalRequest,
        result: Dict[str, Any],
        info: Dict[str, Any],
        extra: Dict[str, Any],
        failure: Optional[str] = None,
    ) -> KernelEvalResponse:
        stage = result.get("stage", "setup")
        status = result.get("status")
        compiled = stage not in ("setup", "static_check", "compile")
        timing = result.get("timing") or {}
        correctness = result.get("correctness") or {}
        correct = bool(correctness.get("correct")) and failure is None

        if failure is None:
            if info["timed_out"]:
                failure = "timeout"
            elif status == "failed":
                failure = {"static_check": "static_check", "compile": "compile"}.get(
                    stage, "incorrect"
                )
            elif status == "error" or (status != "ok" and info["returncode"] != 0):
                failure = "runtime_error" if compiled else "worker_crash"
                correct = False
            elif timing and timing["status"] != ACCEPTED:
                failure = timing.get("reason") or "timing_unstable"

        extra.update(
            stage=stage,
            seconds=round(info["seconds"], 3),
            returncode=info["returncode"],
            correctness=correctness or None,
            timing={k: v for k, v in timing.items() if k != "block_means"} or None,
        )
        error = result.get("message")
        if failure in ("timeout", "worker_crash", "runtime_error", "memcheck") and not error:
            error = info["log_tail"][-1000:]

        stable = bool(
            correct
            and failure is None
            and timing.get("status") == ACCEPTED
            and timing.get("speedup") is not None
        )
        failure_status_map = {
            "static_check": "static_failed",
            "compile": "compile_error",
            "incorrect": "incorrect",
            "memcheck": "sanitizer_error",
            "inter_block_variance_exceeded": "unstable",
            "speedup_above_suspicious_threshold": "implausible_speedup",
            "above_roofline_bandwidth": "fake_gain",
            "cross_gpu_mismatch": "cross_gpu_mismatch",
            "timeout": "timeout",
            "runtime_error": "runtime_error",
            "worker_crash": "runtime_error",
        }
        mapped_status = (
            "ok"
            if stable
            else failure_status_map.get(failure or "", failure or "unknown")
        )
        cand_bw = timing.get("candidate_bandwidth_gbps")
        ref_bw = timing.get("reference_bandwidth_gbps")
        if cand_bw is not None and ref_bw:
            bw_delta = (float(cand_bw) - float(ref_bw)) / float(ref_bw)
        elif timing.get("speedup"):
            bw_delta = float(timing["speedup"]) - 1.0
        else:
            bw_delta = 0.0

        return KernelEvalResponse(
            task_id=request.task_id,
            compiled=compiled,
            correct=correct,
            execution_time_ms=timing.get("median_candidate_ms"),
            baseline_time_ms=timing.get("median_reference_ms"),
            speedup=timing.get("speedup"),
            memory_bandwidth_gbps=cand_bw,
            baseline_bandwidth_gbps=ref_bw,
            bandwidth_delta=bw_delta,
            compiler_output=result.get("compiler_output", ""),
            error_message=error,
            status=mapped_status,
            stable=stable,
            variance=float(timing.get("variance") or 0.0),
            candidate_code_hash=str(
                result.get("candidate_code_hash") or result.get("source_sha256") or ""
            ),
            device_name=str(result.get("device_name") or timing.get("gpu_name") or ""),
            failure_stage=failure,
            extra=extra,
        )
