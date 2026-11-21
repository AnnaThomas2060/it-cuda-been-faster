"""Single-command orchestrator for the full CUDA kernel RL pipeline.

Runs preflight, teacher verification, SFT warm-up, optional ReST self-training,
Arm A (contrastive) and Arm B (reference-only) GRPO training, evaluation sweeps,
paired arm comparison, and Roofline/fake-gain analytics.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parent
STAGES = [
    "preflight",
    "verify_teachers",
    "sft",
    "rest",
    "grpo_arm_a",
    "grpo_arm_b",
    "evaluate",
    "compare_and_audit",
]


def _has_native_toolchain() -> bool:
    if shutil.which("nvcc") is None or shutil.which("ninja") is None:
        return False
    if os.name == "nt":
        return shutil.which("cl") is not None
    return shutil.which("g++") is not None or shutil.which("clang++") is not None


def _run_step(name: str, cmd: List[str], env: Dict[str, str]) -> float:
    print(f"\n=== [{name}] {' '.join(cmd)} ===")
    started = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(f"Stage '{name}' failed with exit code {result.returncode}")
    print(f"=== [{name}] completed in {elapsed:.1f}s ===")
    return elapsed


def _build_local_4050_configs(output_root: Path) -> Dict[str, Path]:
    cfg_dir = output_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = str(output_root / "logs" / "trajectory_telemetry.jsonl")
    archive_path = str(output_root / "contrastive_archive.jsonl")

    sft_cfg = {
        "model_name_or_path": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "use_dev_model": False,
        "use_tiny_model": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "learning_rate": 1.0e-5,
        "num_epochs": 1,
        "max_seq_length": 1024,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "device": "cuda:0",
        "output_dir": str(output_root / "sft_warmup"),
        "training_tasks": ["swish", "column_max", "channel_rmsnorm"],
        "seed": 635,
    }
    rest_cfg = {
        "model_name_or_path": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "use_tiny_model": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "num_iterations": 1,
        "rollouts_per_task": 2,
        "refinement_turns": 1,
        "temperature": 0.8,
        "top_p": 0.95,
        "speedup_thresholds": [1.0],
        "max_kept_per_task": 2,
        "max_seq_length": 1024,
        "max_prompt_length": 768,
        "max_completion_length": 128,
        "rollout_batch_size": 1,
        "learning_rate": 1.0e-5,
        "batch_size": 1,
        "device": "cuda:0",
        "verifier_device": "cuda:0",
        "output_dir": str(output_root / "rest_selftrain"),
        "training_tasks": ["swish", "column_max", "channel_rmsnorm"],
        "seed": 635,
    }
    base_grpo = {
        "base_model_name": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "tiny_model_name": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "use_tiny_model": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "learning_rate": 1.0e-5,
        "weight_decay": 0.01,
        "beta": 0.0,
        "group_size": 2,
        "num_generations": 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "max_steps": 2,
        "training_passes": 1,
        "tasks_per_step": 1,
        "rollout_batch_size": 1,
        "max_prompt_length": 768,
        "max_completion_length": 128,
        "max_teacher_examples": 2,
        "temperature": 0.8,
        "top_p": 0.95,
        "seed": 635,
        "data_seed": 635,
        "save_steps": 1,
        "save_total_limit": 2,
        "logging_steps": 1,
        "num_turns": 2,
        "eval_turns": 2,
        "gamma": 0.4,
        "early_stop_speedup": 2.0,
        "reward_mode": "shaped",
        "correctness_bonus": 0.30,
        "bandwidth_weight": 0.10,
        "dr_grpo_l_norm": 1024,
        "epsilon": 0.20,
        "epsilon_high": 0.28,
        "max_grad_norm": 0.05,
        "use_verifier_worker": False,
        "device": "cuda:0",
        "verifier_device": "cuda:0",
        "archive_path": archive_path,
        "telemetry_path": telemetry_path,
        "output_dir_arm_a": str(output_root / "arm_a_contrastive"),
        "output_dir_arm_b": str(output_root / "arm_b_reference_only"),
        "eval_output_dir": str(output_root / "eval_results"),
    }

    arm_b_cfg = dict(base_grpo, mode="multi_turn")
    arm_a_cfg = dict(
        base_grpo,
        mode="contrastive",
        enable_cross_trajectory_anchors=True,
        archive_capacity_per_task=8,
        min_speedup_margin_delta=0.10,
        pairwise_margin_weight=0.20,
        pairwise_beta=0.10,
        contrastive_pairs_per_step=2,
    )

    paths = {
        "sft": cfg_dir / "sft.yaml",
        "rest": cfg_dir / "rest.yaml",
        "arm_b": cfg_dir / "grpo_arm_b.yaml",
        "arm_a": cfg_dir / "grpo_arm_a.yaml",
    }
    for key, payload in [
        ("sft", sft_cfg),
        ("rest", rest_cfg),
        ("arm_b", arm_b_cfg),
        ("arm_a", arm_a_cfg),
    ]:
        with open(paths[key], "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
    return paths


def _profile_settings(profile: str, output_root: Path) -> Dict[str, Any]:
    if profile == "local_4050":
        paths = _build_local_4050_configs(output_root)
        dispatcher = "in_process" if _has_native_toolchain() else "mock"
        return {
            "sft_cfg": str(paths["sft"]),
            "rest_cfg": str(paths["rest"]),
            "arm_b_cfg": str(paths["arm_b"]),
            "arm_a_cfg": str(paths["arm_a"]),
            "device": "cuda:0",
            "verifier_device": "cuda:0",
            "dispatcher": dispatcher,
            "gpu_name": "RTX 4050",
            "allow_cpu_preflight": True,
            "extra_flags": ["--use_tiny_model"],
            "eval_mode": "single_turn",
        }
    if profile == "rtx3090":
        return {
            "sft_cfg": "configs/sft_lora.yaml",
            "rest_cfg": "configs/rest.yaml",
            "arm_b_cfg": "configs/rtx3090_multiturn_grpo.yaml",
            "arm_a_cfg": "configs/rtx3090_contrastive_rl.yaml",
            "device": "cuda:0",
            "verifier_device": "cuda:1",
            "dispatcher": "auto",
            "gpu_name": "RTX 3090",
            "allow_cpu_preflight": False,
            "extra_flags": [],
            "eval_mode": "multi_turn",
        }
    return {
        "sft_cfg": "configs/sft_lora.yaml",
        "rest_cfg": "configs/rest.yaml",
        "arm_b_cfg": "configs/multiturn_grpo.yaml",
        "arm_a_cfg": "configs/contrastive_rl.yaml",
        "device": "cuda:0",
        "verifier_device": "cuda:1",
        "dispatcher": "auto",
        "gpu_name": "A100-80GB",
        "allow_cpu_preflight": False,
        "extra_flags": [],
        "eval_mode": "multi_turn",
    }


def _write_summary_report(
    output_root: Path,
    eval_dir: Path,
    timings: Dict[str, float],
    profile: str,
) -> Path:
    report_path = eval_dir / "pipeline_report.md"
    lines = [
        "# End-to-End Pipeline Report",
        "",
        f"- Profile: `{profile}`",
        f"- Output root: `{output_root}`",
        "",
        "## Stage Timings",
    ]
    for stage, secs in timings.items():
        lines.append(f"- {stage}: {secs:.1f}s")

    lines.extend(["", "## Checkpoint Evaluation Summary (Single-Turn Greedy)", ""])
    lines.append(
        "| Checkpoint | Correctness | Unresolved Timing | Fast >= 1.0x | Fast >= 1.1x | "
        "Median Speedup | Mean Tokens |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, filename in [
        ("Base Control", "base_control.json"),
        ("SFT Warmup", "sft_warmup.json"),
        ("Arm B (Reference-Only)", "arm_b.json"),
        ("Arm A (Contrastive)", "arm_a.json"),
        ("SFT Unscored Diagnostic (Train)", "sft_unscored_diag.json"),
        ("SFT Scored Diagnostic (Train)", "sft_scored_diag.json"),
        ("Arm B (Multi-Turn Ablation)", "arm_b_multiturn.json"),
        ("Arm A (Multi-Turn Ablation)", "arm_a_multiturn.json"),
    ]:
        path = eval_dir / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"| {label} | {data['correctness_rate']*100:.1f}% | "
            f"{data.get('unresolved_timing_count', 0)} | "
            f"{data['fast_1_0_rate']*100:.1f}% | {data['fast_1_1_rate']*100:.1f}% | "
            f"{data['median_speedup']:.2f}x | {data.get('mean_tokens_generated', 0):.1f} |"
        )

    baselines_path = eval_dir / "hardware_baselines.json"
    if baselines_path.exists():
        bdata = json.loads(baselines_path.read_text(encoding="utf-8"))
        lines.extend(["", f"## Hardware Baselines ({bdata.get('device', 'GPU')})", ""])
        lines.append("| Task | Eager (ms) | Eager BW (GB/s) | torch.compile Speedup | Status |")
        lines.append("| :--- | ---: | ---: | ---: | :--- |")
        for row in bdata.get("tasks", []):
            comp_sp = (
                f"{row['torch_compile_speedup']:.2f}x"
                if row.get("torch_compile_speedup") is not None
                else "-"
            )
            lines.append(
                f"| {row['task_id']} | {row['eager_ms']:.4f} | "
                f"{row['eager_bandwidth_gbps']:.1f} | {comp_sp} | "
                f"{row.get('torch_compile_status', 'unknown')} |"
            )

    for arm_label, arm_dir_name in [
        ("Arm A (Contrastive)", "arm_a_contrastive"),
        ("Arm B (Reference-Only)", "arm_b_reference_only"),
    ]:
        meta_path = output_root / arm_dir_name / "run_metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics = meta.get("metrics", [])
        opt_steps = sum(m.get("optimizer_steps", 0) for m in metrics)
        zero_adv = sum(m.get("zero_advantage_groups", 0) for m in metrics)
        scored_pairs = sum(m.get("scored_pair_groups", 0) for m in metrics)
        fallbacks = sum(m.get("reference_fallback_groups", 0) for m in metrics)
        lines.extend(
            [
                "",
                f"## Training Diagnostics: {arm_label}",
                f"- Rollout steps: {len(metrics)}",
                f"- Actual optimizer steps: {opt_steps}",
                f"- Zero-advantage groups skipped: {zero_adv}",
                f"- Groups with >=2 scored examples: {scored_pairs}",
                f"- Groups falling back to reference-only: {fallbacks}",
            ]
        )

    comp_path = eval_dir / "comparison_all.json"
    if comp_path.exists():
        comp = json.loads(comp_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "## Arm A vs Arm B Paired Comparison (All Tasks)",
                f"- Arm A median: {comp['arm_a_median_speedup']:.4f}x",
                f"- Arm B median: {comp['arm_b_median_speedup']:.4f}x",
                f"- Paired median diff: {comp['median_difference']:+.4f}x "
                f"CI {comp['median_difference_ci']}",
                f"- Wins A={comp['wins_a']}, Ties={comp['ties']}, Wins B={comp['wins_b']}",
            ]
        )

    lines.extend(["", "## Failure Mode Taxonomy Across Checkpoints", ""])
    lines.append(
        "| Checkpoint | Compiled | Numerically Correct | Unresolved Timing | Stable Pass |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for label, filename in [
        ("Base Control", "base_control.json"),
        ("SFT Warmup", "sft_warmup.json"),
        ("Arm B (Single-Turn)", "arm_b.json"),
        ("Arm A (Single-Turn)", "arm_a.json"),
    ]:
        path = eval_dir / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        total = max(1, data.get("total_tasks", 1))
        stable_cnt = sum(
            1
            for r in data.get("task_best", [])
            if r.get("numerically_correct") and r.get("is_stable")
        )
        lines.append(
            f"| {label} | {data.get('compile_rate', 0)*100:.1f}% | "
            f"{data.get('correctness_rate', 0)*100:.1f}% | "
            f"{data.get('unresolved_timing_count', 0)} | "
            f"{stable_cnt / total * 100:.1f}% |"
        )

    for label, filename in [
        ("Arm B Multi-Turn", "arm_b_multiturn.json"),
        ("Arm A Multi-Turn", "arm_a_multiturn.json"),
    ]:
        path = eval_dir / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        by_turn: Dict[int, List[Dict[str, Any]]] = {}
        for task_rec in data.get("tasks", []):
            for turn in task_rec.get("turn_records", []):
                by_turn.setdefault(int(turn.get("turn_index", 0)), []).append(turn)
        if by_turn:
            lines.extend(["", f"## Turn-by-Turn Refinement ({label})", ""])
            lines.append("| Turn | Attempts | OK Rate | Mean Speedup (OK) |")
            lines.append("| ---: | ---: | ---: | ---: |")
            for t_idx in sorted(by_turn):
                turns = by_turn[t_idx]
                ok = [t for t in turns if t.get("status") == "ok"]
                mean_sp = (
                    sum(float(t.get("speedup", 0.0)) for t in ok) / len(ok) if ok else 0.0
                )
                lines.append(
                    f"| {t_idx} | {len(turns)} | {len(ok)/len(turns)*100:.1f}% | {mean_sp:.2f}x |"
                )

    arm_a_path = eval_dir / "arm_a.json"
    if arm_a_path.exists():
        arm_a_data = json.loads(arm_a_path.read_text(encoding="utf-8"))
        passing = [
            r
            for r in arm_a_data.get("task_best", [])
            if r.get("numerically_correct") and r.get("is_stable")
        ]
        if passing:
            from engine.compiler import extract_code_block

            top = max(passing, key=lambda r: float(r.get("speedup", 0.0)))
            snippet = extract_code_block(str(top.get("completion", "")))[:1200]
            lines.extend(
                [
                    "",
                    f"## Top Generated Kernel Showcase (Arm A: {top['task_id']}, "
                    f"{top['speedup']:.2f}x)",
                    "",
                    "```cpp",
                    snippet,
                    "```",
                ]
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full CUDA RL pipeline end-to-end.")
    parser.add_argument(
        "--profile",
        default="rtx3090",
        choices=["local_4050", "rtx3090", "a100"],
        help="Hardware and model preset.",
    )
    parser.add_argument("--start_stage", default="preflight", choices=STAGES)
    parser.add_argument("--stop_stage", default="compare_and_audit", choices=STAGES)
    parser.add_argument("--with_rest", action="store_true", help="Include Stage 2 ReST.")
    parser.add_argument(
        "--pure_grpo",
        action="store_true",
        help="Set pairwise_margin_weight=0.0 on Arm A to isolate prompt conditioning.",
    )
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--seed", type=int, default=635)
    args = parser.parse_args()

    default_root = "./outputs/local_4050" if args.profile == "local_4050" else "./outputs"
    output_root = Path(args.output_root or default_root)
    output_root.mkdir(parents=True, exist_ok=True)
    settings = _profile_settings(args.profile, output_root)

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    py = sys.executable

    start_idx = STAGES.index(args.start_stage)
    stop_idx = STAGES.index(args.stop_stage)
    active = set(STAGES[start_idx : stop_idx + 1])

    sft_dir = output_root / "sft_warmup"
    rest_dir = output_root / "rest_selftrain"
    arm_a_dir = output_root / "arm_a_contrastive"
    arm_b_dir = output_root / "arm_b_reference_only"
    eval_dir = output_root / "eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = (
        output_root / "logs" / "trajectory_telemetry.jsonl"
        if args.profile == "local_4050"
        else Path("logs/trajectory_telemetry.jsonl")
    )

    timings: Dict[str, float] = {}

    if "preflight" in active:
        cmd = [py, "scripts/check_env.py", "--config", settings["arm_b_cfg"]]
        if settings["allow_cpu_preflight"]:
            cmd.append("--allow_cpu")
        timings["preflight"] = _run_step("preflight", cmd, env)

    if "verify_teachers" in active:
        if args.profile == "local_4050" and not _has_native_toolchain():
            print(
                "\n=== [verify_teachers] skipped on local_4050: "
                "cl.exe/ninja not found for native CUDA compilation ==="
            )
        else:
            cmd = [
                py,
                "scripts/verify_teachers.py",
                "--config",
                settings["arm_b_cfg"],
                "--device",
                settings["verifier_device"],
                "--strict",
            ]
            timings["verify_teachers"] = _run_step("verify_teachers", cmd, env)

        baseline_cmd = [
            py,
            "scripts/benchmark_baselines.py",
            "--device",
            settings["verifier_device"],
            "--window_seconds",
            "1.0" if args.profile == "local_4050" else "3.0",
            "--output",
            str(eval_dir / "hardware_baselines.json"),
        ]
        timings["hardware_baselines"] = _run_step("hardware_baselines", baseline_cmd, env)

    sft_adapter = sft_dir / "final_adapter"
    if "sft" in active:
        if args.skip_completed and (sft_adapter / "adapter_config.json").exists():
            print(f"\n=== [sft] skipping existing {sft_adapter} ===")
        else:
            cmd = [
                py,
                "train_sft.py",
                "--config",
                settings["sft_cfg"],
                "--output_dir",
                str(sft_dir),
                "--device",
                settings["device"],
                "--seed",
                str(args.seed),
            ] + settings["extra_flags"]
            timings["sft"] = _run_step("sft", cmd, env)

    init_adapter = sft_adapter
    rest_adapter = rest_dir / "final_adapter"
    if "rest" in active and args.with_rest:
        if args.skip_completed and (rest_adapter / "adapter_config.json").exists():
            print(f"\n=== [rest] skipping existing {rest_adapter} ===")
        else:
            cmd = [
                py,
                "train_rest.py",
                "--config",
                settings["rest_cfg"],
                "--sft_adapter_path",
                str(sft_adapter),
                "--output_dir",
                str(rest_dir),
                "--device",
                settings["device"],
                "--verifier_device",
                settings["verifier_device"],
                "--dispatcher",
                settings["dispatcher"],
            ] + settings["extra_flags"]
            timings["rest"] = _run_step("rest", cmd, env)
        init_adapter = rest_adapter

    arm_a_adapter = arm_a_dir / "final_adapter"
    if "grpo_arm_a" in active:
        if args.skip_completed and (arm_a_adapter / "adapter_config.json").exists():
            print(f"\n=== [grpo_arm_a] skipping existing {arm_a_adapter} ===")
        else:
            cmd = [
                py,
                "train_grpo.py",
                "--arm",
                "arm_a",
                "--config",
                settings["arm_a_cfg"],
                "--sft_adapter_path",
                str(init_adapter),
                "--output_dir",
                str(arm_a_dir),
                "--device",
                settings["device"],
                "--verifier_device",
                settings["verifier_device"],
                "--dispatcher",
                settings["dispatcher"],
                "--seed",
                str(args.seed),
            ]
            if args.pure_grpo:
                cmd.extend(["--pairwise_margin_weight", "0.0"])
            cmd.extend(settings["extra_flags"])
            timings["grpo_arm_a"] = _run_step("grpo_arm_a", cmd, env)

    arm_b_adapter = arm_b_dir / "final_adapter"
    if "grpo_arm_b" in active:
        if args.skip_completed and (arm_b_adapter / "adapter_config.json").exists():
            print(f"\n=== [grpo_arm_b] skipping existing {arm_b_adapter} ===")
        else:
            cmd = [
                py,
                "train_grpo.py",
                "--arm",
                "arm_b",
                "--config",
                settings["arm_b_cfg"],
                "--sft_adapter_path",
                str(init_adapter),
                "--output_dir",
                str(arm_b_dir),
                "--device",
                settings["device"],
                "--verifier_device",
                settings["verifier_device"],
                "--dispatcher",
                settings["dispatcher"],
                "--seed",
                str(args.seed),
            ] + settings["extra_flags"]
            timings["grpo_arm_b"] = _run_step("grpo_arm_b", cmd, env)

    if "evaluate" in active:
        eval_jobs = [
            (
                "eval_base",
                "base_control.json",
                "single_turn",
                ["--base_model_control", "--prompt_mode", "reference_only"],
            ),
            (
                "eval_sft",
                "sft_warmup.json",
                "single_turn",
                ["--sft_adapter_path", str(sft_adapter), "--prompt_mode", "reference_only"],
            ),
            (
                "eval_arm_b",
                "arm_b.json",
                "single_turn",
                [
                    "--sft_adapter_path",
                    str(init_adapter),
                    "--adapter_path",
                    str(arm_b_adapter),
                    "--prompt_mode",
                    "reference_only",
                ],
            ),
            (
                "eval_arm_a",
                "arm_a.json",
                "single_turn",
                [
                    "--sft_adapter_path",
                    str(init_adapter),
                    "--adapter_path",
                    str(arm_a_adapter),
                    "--prompt_mode",
                    "reference_only",
                ],
            ),
            (
                "eval_sft_unscored_diag",
                "sft_unscored_diag.json",
                "single_turn",
                [
                    "--sft_adapter_path",
                    str(sft_adapter),
                    "--prompt_mode",
                    "unscored_context",
                    "--tasks",
                    "swish,column_max,channel_rmsnorm",
                ],
            ),
            (
                "eval_sft_diag",
                "sft_scored_diag.json",
                "single_turn",
                [
                    "--sft_adapter_path",
                    str(sft_adapter),
                    "--prompt_mode",
                    "contrastive",
                    "--tasks",
                    "swish,column_max,channel_rmsnorm",
                ],
            ),
            (
                "eval_arm_b_mt",
                "arm_b_multiturn.json",
                "multi_turn",
                [
                    "--sft_adapter_path",
                    str(init_adapter),
                    "--adapter_path",
                    str(arm_b_adapter),
                    "--prompt_mode",
                    "reference_only",
                ],
            ),
            (
                "eval_arm_a_mt",
                "arm_a_multiturn.json",
                "multi_turn",
                [
                    "--sft_adapter_path",
                    str(init_adapter),
                    "--adapter_path",
                    str(arm_a_adapter),
                    "--prompt_mode",
                    "reference_only",
                ],
            ),
        ]
        for label, out_name, eval_mode, extra in eval_jobs:
            out_file = eval_dir / out_name
            if args.skip_completed and out_file.exists():
                print(f"\n=== [{label}] skipping existing {out_file} ===")
                continue
            cmd = [
                py,
                "evaluate.py",
                "--config",
                settings["arm_b_cfg"],
                "--mode",
                eval_mode,
                "--device",
                settings["device"],
                "--verifier_device",
                settings["verifier_device"],
                "--dispatcher",
                settings["dispatcher"],
                "--output_file",
                str(out_file),
            ] + extra + settings["extra_flags"]
            timings[label] = _run_step(label, cmd, env)

    if "compare_and_audit" in active:
        for split in ["all", "training", "holdout"]:
            cmd = [
                py,
                "scripts/compare_arms.py",
                "--arm_a",
                str(eval_dir / "arm_a.json"),
                "--arm_b",
                str(eval_dir / "arm_b.json"),
                "--split",
                split,
                "--output",
                str(eval_dir / f"comparison_{split}.json"),
            ]
            timings[f"compare_{split}"] = _run_step(f"compare_{split}", cmd, env)

        if telemetry_path.exists():
            cmd = [
                py,
                "-m",
                "src.analytics",
                str(telemetry_path),
                "--gpu",
                settings["gpu_name"],
            ]
            timings["analytics"] = _run_step("analytics", cmd, env)

        report_path = _write_summary_report(output_root, eval_dir, timings, args.profile)
        print(f"\nPipeline report written to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
