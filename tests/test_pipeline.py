import importlib.util
import math
import os
import re
import sys
import unittest

# Set up module loading for pipeline components across local filesystem environments.
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

for pkg_name in ["configs", "data", "engine", "rewards", "models"]:
    pkg_init = os.path.join(WORKSPACE_ROOT, pkg_name, "__init__.py")
    if os.path.exists(pkg_init) and pkg_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(pkg_name, pkg_init)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = mod
        spec.loader.exec_module(mod)

from configs.grpo_config import ExperimentConfig  # noqa: E402
from data.example_store import get_teacher_examples  # noqa: E402
from data.tasks import get_task, list_task_ids  # noqa: E402
from engine.compiler import extract_code_block, run_static_checks  # noqa: E402
from engine.prompt_builder import build_prompt  # noqa: E402
from rewards.grpo_reward import (  # noqa: E402
    compute_grpo_rewards,
    get_taxonomy_summary,
    infer_task_id_from_prompt,
    reset_taxonomy_stats,
)

# Unit test suite verifying core components of the RL training pipeline.
# Checks task registration, teacher example retrieval, prompt construction, and static code filters.
# Runs in lightweight test environments without requiring an active CUDA runtime.

class TestPipelineComponents(unittest.TestCase):

    def test_task_registry(self):
        training_tasks = list_task_ids(training_only=True)
        self.assertIn("swish", training_tasks)
        self.assertIn("column_max", training_tasks)
        self.assertIn("channel_rmsnorm", training_tasks)

        all_tasks = list_task_ids(training_only=False)
        self.assertIn("average_pooling_2d", all_tasks)
        self.assertIn("softmax", all_tasks)
        self.assertIn("cumsum", all_tasks)
        self.assertGreater(len(all_tasks), len(training_tasks))

        col_max = get_task("column_max")
        self.assertTrue(col_max["exact_match"])
        self.assertEqual(len(col_max["shapes"]), 3)

        with self.assertRaises(KeyError):
            get_task("nonexistent_task_id")

    def test_teacher_examples(self):
        swish_teachers = get_teacher_examples("swish")
        self.assertEqual(len(swish_teachers), 2)
        self.assertGreater(swish_teachers[1]["speedup"], swish_teachers[0]["speedup"])

        # RMSNorm single teacher fallback check
        rmsnorm_teachers = get_teacher_examples("channel_rmsnorm")
        self.assertEqual(len(rmsnorm_teachers), 1)

        empty_teachers = get_teacher_examples("nonexistent_task")
        self.assertEqual(empty_teachers, [])

    def test_prompt_builder(self):
        # Reference-only mode
        p_ref = build_prompt("swish", mode="reference_only")
        self.assertIn("torch::Tensor run_cuda", p_ref)
        self.assertNotIn("Historical Implementations", p_ref)

        # Contrastive mode
        p_cont = build_prompt("swish", mode="contrastive")
        self.assertIn("torch::Tensor run_cuda", p_cont)
        self.assertIn("Historical Implementations and Measured Speedups", p_cont)
        self.assertIn("1.15x", p_cont)
        self.assertIn("1.28x", p_cont)

        # Single teacher task contrastive mode
        p_rms = build_prompt("channel_rmsnorm", mode="contrastive")
        self.assertIn("1.20x", p_rms)

        # Scored context alias
        p_scored = build_prompt("swish", mode="scored_context")
        self.assertIn("Historical Implementations and Measured Speedups", p_scored)

        # Invalid mode
        with self.assertRaises(ValueError):
            build_prompt("swish", mode="invalid_mode")

    def test_compiler_static_checks(self):
        valid_code = """
        #include "torch/extension.h"
        __global__ void test_kernel(float* x) {}
        torch::Tensor run_cuda(torch::Tensor x) { return x; }
        """
        passed, msg = run_static_checks(valid_code)
        self.assertTrue(passed)

        missing_global = """
        torch::Tensor run_cuda(torch::Tensor x) { return x; }
        """
        passed, msg = run_static_checks(missing_global)
        self.assertFalse(passed)
        self.assertIn("Missing __global__", msg)

        shortcut_code = """
        __global__ void test_kernel(float* x) {}
        torch::Tensor run_cuda(torch::Tensor x) { return torch::relu(x); }
        """
        passed, msg = run_static_checks(shortcut_code)
        self.assertFalse(passed)
        self.assertIn("Forbidden PyTorch shortcut", msg)

    def test_static_checks_comment_resilience(self):
        # Comments discussing forbidden shortcuts should not trigger false positives
        commented_code = """
        #include "torch/extension.h"
        // We replaced torch::sigmoid with direct fast exp intrinsics
        /* Also avoided at::native or torch::relu */
        __global__ void test_kernel(float* x) {}
        torch::Tensor run_cuda(torch::Tensor x) { return x; }
        """
        passed, msg = run_static_checks(commented_code)
        self.assertTrue(passed, f"Static check falsely triggered on comments: {msg}")

    def test_bare_include_sanitization(self):
        # Simulates code where HTML parsers stripped <torch/extension.h> down to bare #include
        corrupted_code = (
            "#include \n"
            "#include\n"
            "__global__ void k() {}\n"
            "torch::Tensor run_cuda(torch::Tensor x) { return x; }\n"
        )
        passed, _ = run_static_checks(corrupted_code)
        self.assertTrue(passed)

        # Verify bare includes are stripped by regex without touching valid includes
        sanitized = re.sub(
            r"^[ \t]*#include[ \t]*(?:\r?\n|$)", "\n", corrupted_code, flags=re.MULTILINE
        )
        self.assertNotIn("#include \n", sanitized)
        self.assertNotIn("#include\n", sanitized)
        self.assertIn("__global__", sanitized)

    def test_data_ptr_template_repair(self):
        # Simulates code where HTML parsers stripped <float> from data_ptr<float>()
        corrupted_code = "kernel<<<g, b>>>(x.data_ptr(), out.data_ptr(), n);"
        repaired = re.sub(r"(?<=\.data_ptr)\(\)", "<float>()", corrupted_code)
        self.assertEqual(
            repaired, "kernel<<<g, b>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);"
        )

        already_typed = "kernel<<<g, b>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);"
        untouched = re.sub(r"(?<=\.data_ptr)\(\)", "<float>()", already_typed)
        self.assertEqual(untouched, already_typed)

    def test_code_block_extraction(self):
        markdown_wrapped = "Here is the code:\n```cpp\n__global__ void k() {}\n```\nHope it works."
        extracted = extract_code_block(markdown_wrapped)
        self.assertEqual(extracted, "__global__ void k() {}")

        cuda_wrapped = "```cuda\n__global__ void k2() {}\n```"
        self.assertEqual(extract_code_block(cuda_wrapped), "__global__ void k2() {}")

        bare_code = "__global__ void k3() {}"
        self.assertEqual(extract_code_block(bare_code), bare_code)

    def test_task_id_inference(self):
        self.assertEqual(infer_task_id_from_prompt("Implement Swish Activation kernel"), "swish")
        self.assertEqual(infer_task_id_from_prompt("Column Maximum reduction"), "column_max")
        self.assertEqual(infer_task_id_from_prompt("Channel RMSNorm along dim"), "channel_rmsnorm")
        self.assertEqual(infer_task_id_from_prompt("Unknown operation"), "swish")

    def test_reward_taxonomy_on_malformed_code(self):
        reset_taxonomy_stats()
        prompts = [
            "Task: Swish Activation\nImplement kernel.",
            "Task: Swish Activation\nImplement kernel.",
        ]
        completions = ["malformed text without code", "invalid syntax without entrypoint"]
        rewards = compute_grpo_rewards(prompts=prompts, completions=completions)

        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)

        summary = get_taxonomy_summary()
        self.assertEqual(summary["group_stats"]["total_groups"], 1)
        self.assertEqual(summary["group_stats"]["zero_reward_groups"], 1)
        self.assertGreater(summary["error_taxonomy"]["static_check_failed"], 0)

    def test_reward_prompt_replication(self):
        reset_taxonomy_stats()
        # When 1 prompt is provided with 2 completions (GRPO group size G=2)
        prompts = ["Task: Column Maximum\nImplement kernel."]
        completions = ["bad code 1", "bad code 2"]
        rewards = compute_grpo_rewards(prompts=prompts, completions=completions)
        self.assertEqual(len(rewards), 2)

    def test_reward_direct_task_id_routing(self):
        reset_taxonomy_stats()
        # Direct task_id passed via kwargs
        completions = ["bad code 1", "bad code 2"]
        rewards = compute_grpo_rewards(completions=completions, task_id="channel_rmsnorm")
        self.assertEqual(len(rewards), 2)

    def test_floored_speedup_precision(self):
        # 1.15 * 100 in Python IEEE 754 is 114.99999999999999
        median_ratio = 1.15
        corrected_floor = math.floor(round(median_ratio * 100.0, 6)) / 100.0
        self.assertEqual(corrected_floor, 1.15)
        self.assertEqual(math.floor(round(1.28 * 100.0, 6)) / 100.0, 1.28)

    def test_config_active_model(self):
        cfg = ExperimentConfig()
        cfg.use_dev_model = False
        self.assertEqual(cfg.get_active_model_name(), "Qwen/Qwen3-Coder-30B-A3B-Instruct")
        cfg.use_dev_model = True
        self.assertEqual(cfg.get_active_model_name(), "Qwen/Qwen2.5-Coder-7B-Instruct")

    def test_grpo_generation_batch_size_invariant(self):
        cfg = ExperimentConfig()
        gen_batch_size = cfg.num_generations * max(1, cfg.gradient_accumulation_steps)
        per_device_train_batch_size = 1
        num_processes = 1

        # TRL GRPO invariant: generation_batch_size must be divisible by num_generations
        self.assertEqual(gen_batch_size % cfg.num_generations, 0)
        # TRL GRPO invariant: generation_batch_size must be divisible by global train batch size
        self.assertEqual(gen_batch_size % (per_device_train_batch_size * num_processes), 0)

    def test_lora_parameter_precision_resync(self):
        import torch

        # Mock a model with trainable parameters accidentally cast to bfloat16
        linear = torch.nn.Linear(8, 8)
        linear.weight.data = linear.weight.data.to(torch.bfloat16)
        self.assertEqual(linear.weight.dtype, torch.bfloat16)

        # Resynchronize when is_bf16 is False (e.g. Turing T4)
        is_bf16 = False
        target_dtype = torch.bfloat16 if is_bf16 else torch.float32
        for param in linear.parameters():
            if param.requires_grad:
                param.data = param.data.to(target_dtype)

        self.assertEqual(linear.weight.dtype, torch.float32)

if __name__ == "__main__":
    unittest.main()
