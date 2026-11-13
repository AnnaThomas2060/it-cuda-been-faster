"""Preflight checks for local, Colab, and HPC environments.

The script validates the Python stack, CUDA visibility, compiler toolchain,
filesystem space, configuration parsing, and verifier dispatch imports. It
exits with a nonzero status when a required component is missing.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_command(command: List[str], timeout: int = 20) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "output": str(exc)}
    output = (result.stdout or "").strip()
    if result.returncode != 0 and result.stderr:
        output = (output + "\n" + result.stderr.strip()).strip()
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "output": output,
    }


def torch_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {"installed": False, "cuda_available": False}
    try:
        import torch
    except ImportError as exc:
        report["error"] = str(exc)
        return report

    report["installed"] = True
    report["torch_version"] = torch.__version__
    report["cuda_version"] = torch.version.cuda
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["bf16_supported"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_gb": round(properties.total_memory / 1024**3, 2),
                    "capability": f"{properties.major}.{properties.minor}",
                }
            )
        report["device_count"] = torch.cuda.device_count()
        report["devices"] = devices
    return report


def tool_report(names: List[str]) -> Dict[str, Any]:
    report = {}
    for name in names:
        path = shutil.which(name)
        report[name] = {"found": path is not None, "path": path or ""}
    return report


def package_report(names: List[str]) -> Dict[str, Any]:
    from importlib import metadata

    report = {}
    for name in names:
        try:
            report[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            report[name] = None
    return report


def config_report(path: str) -> Dict[str, Any]:
    from configs.grpo_config import ExperimentConfig

    try:
        config = ExperimentConfig.from_yaml(path)
        config.validate()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "mode": config.mode,
        "training_tasks": config.training_tasks,
        "holdout_tasks": config.holdout_tasks,
        "device": config.device,
        "verifier_device": config.verifier_device,
    }


def import_report() -> Dict[str, Any]:
    modules = [
        "train_grpo",
        "evaluate",
        "src.pipeline.env_dispatcher",
        "src.pipeline.rollout_manager",
        "src.rewards.reward_engine",
        "src.trainers.multiturn_grpo",
    ]
    report = {}
    for name in modules:
        try:
            __import__(name)
            report[name] = "ok"
        except Exception as exc:
            report[name] = f"failed: {exc}"
    return report


def build_report(config_path: str) -> Dict[str, Any]:
    usage = shutil.disk_usage(str(REPO_ROOT))
    return {
        "python": sys.version,
        "executable": sys.executable,
        "repo_root": str(REPO_ROOT),
        "disk_free_gb": round(usage.free / 1024**3, 2),
        "torch": torch_report(),
        "packages": package_report(
            ["torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate"]
        ),
        "tools": tool_report(["nvidia-smi", "nvcc", "ninja", "compute-sanitizer", "git"]),
        "config": config_report(config_path),
        "imports": import_report(),
    }


def print_report(report: Dict[str, Any], cuda_required: bool) -> None:
    print("Python:", report["python"].splitlines()[0])
    print("Executable:", report["executable"])
    print("Repository:", report["repo_root"])
    print("Free disk space:", f"{report['disk_free_gb']} GB")
    print()

    torch_info = report["torch"]
    print("Torch:")
    print("  installed:", torch_info["installed"])
    print("  cuda available:", torch_info["cuda_available"])
    if torch_info.get("torch_version"):
        print("  version:", torch_info["torch_version"])
        print("  cuda version:", torch_info["cuda_version"])
        print("  bf16 supported:", torch_info["bf16_supported"])
    if torch_info.get("error"):
        print("  error:", torch_info["error"])
    for device in torch_info.get("devices", []):
        print(
            f"  gpu {device['index']}: {device['name']} "
            f"({device['total_memory_gb']} GB, compute {device['capability']})"
        )
    print()

    print("Packages:")
    for name, version in report["packages"].items():
        print(f"  {name}: {version or 'missing'}")
    print()

    print("Tools:")
    for name, info in report["tools"].items():
        print(f"  {name}: {info['path'] or 'missing'}")
        if name == "nvidia-smi" and info["found"]:
            first_line = run_command(["nvidia-smi"], timeout=15)["output"].splitlines()
            if first_line:
                print("   ", first_line[0])
    print()

    config = report["config"]
    print("Config:")
    if config.get("ok"):
        print("  mode:", config["mode"])
        print("  device:", config["device"])
        print("  verifier device:", config["verifier_device"])
        print("  training tasks:", ", ".join(config["training_tasks"]))
        print("  holdout tasks:", ", ".join(config["holdout_tasks"]))
    else:
        print("  error:", config["error"])
    print()

    print("Imports:")
    for name, status in report["imports"].items():
        print(f"  {name}: {status}")

    problems = []
    if not torch_info["installed"]:
        problems.append("torch is not installed")
    if cuda_required and not torch_info["cuda_available"]:
        problems.append("CUDA is not available")
    if cuda_required and not report["tools"]["nvcc"]["found"]:
        problems.append("nvcc is not on PATH")
    if cuda_required and not report["tools"]["ninja"]["found"]:
        problems.append("ninja is not on PATH")
    if not config.get("ok"):
        problems.append("configuration did not validate")
    for name, status in report["imports"].items():
        if status != "ok":
            problems.append(f"import failed for {name}")

    print()
    if problems:
        print("Preflight problems:")
        for problem in problems:
            print("  -", problem)
    else:
        print("Preflight checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the training environment.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "multiturn_grpo.yaml"))
    parser.add_argument("--allow_cpu", action="store_true", help="Do not fail without CUDA.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    args = parser.parse_args()

    report = build_report(args.config)
    cuda_required = not args.allow_cpu
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report, cuda_required)

    if not report["torch"]["installed"]:
        return 1
    if cuda_required and not report["torch"]["cuda_available"]:
        return 1
    if cuda_required and not report["tools"]["nvcc"]["found"]:
        return 1
    if not report["config"].get("ok"):
        return 1
    if any(status != "ok" for status in report["imports"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
