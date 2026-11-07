from dataclasses import dataclass, field
from typing import List

# Central configuration module for training and evaluation.
# Defines model architectures, LoRA rank parameters, and GRPO training arguments.
# All paths and hyperparameters adhere to the MSAI635 experimental protocol.

@dataclass
class ExperimentConfig:
    # Model configuration
    base_model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    dev_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    use_dev_model: bool = False

    # Quantization and LoRA
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # GRPO training parameters
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    beta: float = 0.0
    num_generations: int = 2
    max_prompt_length: int = 1024
    max_completion_length: int = 1024
    temperature: float = 0.8
    seed: int = 635
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    bf16: bool = True
    logging_steps: int = 1

    # Task scheduling
    training_tasks: List[str] = field(
        default_factory=lambda: ["swish", "column_max", "channel_rmsnorm"]
    )
    eval_tasks: List[str] = field(
        default_factory=lambda: [
            "swish",
            "column_max",
            "channel_rmsnorm",
            "average_pooling_2d",
        ]
    )
    training_passes: int = 2

    # Verification and benchmarking gates
    numerical_rtol: float = 1e-4
    numerical_atol: float = 1e-4
    benchmark_warmup_iters: int = 10
    benchmark_num_blocks: int = 7
    benchmark_runs_per_block: int = 10
    variance_threshold: float = 0.005
    cross_gpu_ratio_threshold: float = 3.0
    cross_gpu_tolerance: float = 0.10

    # Output directories
    output_dir_arm_a: str = "./outputs/arm_a_contrastive"
    output_dir_arm_b: str = "./outputs/arm_b_reference_only"
    eval_output_dir: str = "./outputs/eval_results"

    def get_active_model_name(self) -> str:
        return self.dev_model_name if self.use_dev_model else self.base_model_name

default_config = ExperimentConfig()
