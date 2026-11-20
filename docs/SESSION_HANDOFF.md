# Session Handoff

## Current State
- Branch: `implement/full-multiturn-rl-pipeline`
- Single-command orchestrator: `run_pipeline.py`
- Local GPU verification: passed end-to-end on NVIDIA GeForce RTX 4050 Laptop GPU (6 GB, CUDA 12.4)

## Bugs Found and Fixed During Local RTX 4050 Testing
1. `engine/compiler.py`: Added `module_name` and `build_directory` parameters to `compile_kernel_ex` and `compile_kernel` so `src/verifier/worker.py` can pass cached build paths.
2. `src/verifier/worker.py` and `src/verifier/harness.py`: Guarded Unix-only `resource` and `os.killpg` for cross-platform compatibility, and mapped `source_sha256` -> `candidate_code_hash` and `gpu_name` -> `device_name`.
3. `train_grpo.py`, `train_sft.py`, `train_rest.py`, `src/trainers/multiturn_grpo.py`: Fixed multi-adapter saving (`_save_adapter` / `save_checkpoint`) to use `peft.get_peft_model_state_dict(model, adapter_name="rl")` and save `adapter_config.json` + `adapter_model.safetensors` at the checkpoint root instead of an empty dict or nested `rl/` folder.

## Local RTX 4050 GPU Run Verified
- Command: `python run_pipeline.py --profile local_4050 --with_rest` (with `Inference-Guard-main\.venv`, `Qwen/Qwen2.5-Coder-0.5B-Instruct`, `cuda:0`)
- Verified stages: `preflight`, `sft`, `rest`, `grpo_arm_a`, `grpo_arm_b`, `eval_base`, `eval_sft`, `eval_arm_b`, `eval_arm_a`, `eval_sft_diag`, `compare_all`, `compare_training`, `compare_holdout`, `analytics`.
- Verified GPU checkpoint/resume replay: `train_grpo.py --resume_from_checkpoint outputs/local_4050/arm_a_contrastive/checkpoint-1` resumed at step 1 and completed step 2 cleanly.
- Note: `verify_teachers` native CUDA compilation was skipped on Windows because `cl.exe` (MSVC C++ workload) and `ninja` are not installed locally.
- Static + unit suite: `ruff check .` (0 errors) and `pytest tests/ -q` (78 passed, 4 GPU opt-in skipped).

## Running the Full Pipeline on the Real GPU Cluster (2x RTX 3090 or A100)
Single command for 2x RTX 3090 (24 GiB):
```bash
python run_pipeline.py --profile rtx3090
```
Optional flags:
- `--with_rest` to include Stage 2 ReST self-training before GRPO.
- `--start_stage <stage>` / `--stop_stage <stage>` to resume or run a subset (`preflight`, `verify_teachers`, `sft`, `rest`, `grpo_arm_a`, `grpo_arm_b`, `evaluate`, `compare_and_audit`).
- `--skip_completed` to skip stages whose output adapter/JSON already exists.
- `--profile a100` for A100 80GB (`lora_r=64`, `group_size=16`, `num_turns=4`).
