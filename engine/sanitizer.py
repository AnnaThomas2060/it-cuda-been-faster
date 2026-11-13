import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple


def run_compute_sanitizer(
    code: str, task_id: str, device: str, timeout_sec: float = 120.0
) -> Tuple[bool, str]:
    executable = shutil.which("compute-sanitizer")
    if executable is None:
        return False, "compute-sanitizer is not available on PATH."

    repo_root = str(Path(__file__).resolve().parent.parent)
    script = f"""
import sys
sys.path.insert(0, {repo_root!r})
from engine.compiler import compile_kernel_ex
from engine.verifier import verify_kernel

code = {code!r}
task_id = {task_id!r}
device = {device!r}
result = compile_kernel_ex(code)
if result.module is None:
    print(result.message)
    sys.exit(2)
verification = verify_kernel(result.module, task_id, device=device)
if not verification.correct:
    print(verification.message)
    sys.exit(3)
print("sanitizer run completed")
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "sanitizer_runner.py"
        script_path.write_text(script, encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        command = [
            executable,
            "--tool",
            "memcheck",
            "--error-exitcode",
            "1",
            sys.executable,
            str(script_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, f"compute-sanitizer timed out after {timeout_sec:.0f}s."
        except Exception as exc:
            return False, f"compute-sanitizer failed to start: {exc}"

        output = (result.stdout or "") + "\n" + (result.stderr or "")
        if result.returncode != 0:
            return False, output[-4000:]
        return True, output[-2000:]
