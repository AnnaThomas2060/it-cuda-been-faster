from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentConfig:
    mode: str = "multi_turn"

    base_model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    dev_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    tiny_model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    use_dev_model: bool = False
    use_tiny_model: bool = False
    model_revision: Optional[str] = None
    sft_adapter_path: Optional[str] = None
    device: str = "cuda:0"
    verifier_device: str = "cuda:1"
    device_map: Optional[str] = None
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = True
    local_files_only: bool = False
    cache_dir: Optional[str] = None

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    beta: float = 0.0
    num_generations: int = 4
    group_size: int = 4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_steps: int = 100
    num_train_epochs: int = 1
    max_prompt_length: int = 4096
    max_completion_length: int = 1536
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    seed: int = 635
    data_seed: int = 635
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 1
    save_steps: int = 25
    save_total_limit: int = 2
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    loss_type: str = "dr_grpo"
    dr_grpo_l_norm: int = 16384
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    scale_rewards: bool = False
    mask_truncated_completions: bool = True
    report_to: str = "none"
    run_name: str = "msai635_multiturn"

    num_turns: int = 4
    eval_turns: int = 8
    gamma: float = 0.4
    early_stop_speedup: float = 2.0
    training_tasks: List[str] = field(
        default_factory=lambda: ["swish", "column_max", "channel_rmsnorm"]
    )
    holdout_tasks: List[str] = field(
        default_factory=lambda: ["average_pooling_2d", "softmax", "cumsum"]
    )
    eval_tasks: List[str] = field(
        default_factory=lambda: [
            "swish",
            "column_max",
            "channel_rmsnorm",
            "average_pooling_2d",
            "softmax",
            "cumsum",
        ]
    )
    training_passes: int = 2
    tasks_per_step: int = 1
    rollout_batch_size: int = 2
    max_teacher_examples: int = 4

    reward_mode: str = "shaped"
    correctness_bonus: float = 0.30
    bandwidth_weight: float = 0.10
    reward_clip: float = 2.0
    use_bandwidth: bool = True
    fake_gain_ratio: float = 1.20
    max_plausible_speedup: float = 10.0

    numerical_rtol: float = 1e-4
    numerical_atol: float = 1e-4
    verification_seeds: List[int] = field(default_factory=lambda: [101, 202, 303, 404])
    benchmark_warmup_iters: int = 10
    benchmark_num_blocks: int = 7
    benchmark_runs_per_block: int = 10
    variance_threshold: float = 0.005
    min_measurement_ms: float = 0.005
    enable_sanitizer: bool = False
    sanitizer_timeout_sec: int = 120

    use_verifier_worker: bool = True
    verifier_workers: int = 1
    verifier_startup_timeout_sec: int = 90
    compile_timeout_sec: int = 120
    verify_timeout_sec: int = 120
    benchmark_timeout_sec: int = 180
    torch_extensions_dir: Optional[str] = None
    max_worker_memory_gb: Optional[float] = None
    # Optional per-GPU memory cap for the base model (e.g. "36GiB"). Leaves headroom
    # for activations and a co-resident verifier. None lets the backend decide.
    max_model_memory_per_gpu: Optional[str] = None
    max_model_memory_cpu: Optional[str] = None

    archive_path: str = "./outputs/contrastive_archive.jsonl"
    archive_capacity_per_task: int = 16
    enable_cross_trajectory_anchors: bool = True
    min_speedup_margin_delta: float = 0.25
    pairwise_margin_weight: float = 0.20
    pairwise_beta: float = 0.10
    contrastive_pairs_per_step: int = 4

    telemetry_path: str = "./logs/trajectory_telemetry.jsonl"
    output_dir_arm_a: str = "./outputs/arm_a_contrastive"
    output_dir_arm_b: str = "./outputs/arm_b_reference_only"
    eval_output_dir: str = "./outputs/eval_results"

    def get_active_model_name(self) -> str:
        if self.use_tiny_model:
            return self.tiny_model_name
        return self.dev_model_name if self.use_dev_model else self.base_model_name

    def prompt_mode_for_arm(self, arm: str) -> str:
        if arm == "arm_a":
            return "contrastive"
        if arm == "arm_b":
            return "reference_only"
        raise ValueError(f"Unknown arm: {arm}")

    def output_dir_for_arm(self, arm: str) -> str:
        if arm == "arm_a":
            return self.output_dir_arm_a
        if arm == "arm_b":
            return self.output_dir_arm_b
        raise ValueError(f"Unknown arm: {arm}")

    def archive_path_for_arm(self, arm: str) -> str:
        if arm not in {"arm_a", "arm_b"}:
            raise ValueError(f"Unknown arm: {arm}")
        path = Path(self.archive_path)
        return str(path.with_name(f"{path.stem}_{arm}{path.suffix}"))

    def effective_group_size(self) -> int:
        return max(2, int(self.group_size or self.num_generations))

    def validate(self) -> None:
        if self.mode not in {"single_turn", "multi_turn", "contrastive"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.reward_mode not in {"speedup", "shaped"}:
            raise ValueError(f"Unsupported reward_mode: {self.reward_mode}")
        if self.num_turns < 1:
            raise ValueError("num_turns must be at least 1")
        if self.eval_turns < 1:
            raise ValueError("eval_turns must be at least 1")
        if self.effective_group_size() < 2:
            raise ValueError("group_size must be at least 2 for GRPO")
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if self.max_completion_length < 16:
            raise ValueError("max_completion_length is too small")
        if self.benchmark_num_blocks < 1 or self.benchmark_runs_per_block < 1:
            raise ValueError("Benchmark block counts must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 are mutually exclusive")
        if self.loss_type == "dr_grpo" and self.dr_grpo_l_norm <= 0:
            raise ValueError("dr_grpo_l_norm must be > 0 for the dr_grpo loss")
        if self.reward_clip <= 0:
            raise ValueError("reward_clip must be positive")
        if self.rollout_batch_size < 1:
            raise ValueError("rollout_batch_size must be at least 1")
        if self.verifier_workers < 1:
            raise ValueError("verifier_workers must be at least 1")
        if self.max_prompt_length < 1:
            raise ValueError("max_prompt_length must be at least 1")
        if self.max_teacher_examples < 0:
            raise ValueError("max_teacher_examples must be non-negative")
        if self.pairwise_margin_weight < 0 or self.pairwise_beta < 0:
            raise ValueError("pairwise_margin_weight and pairwise_beta must be non-negative")
        missing = [
            task for task in self.eval_tasks if task not in self.training_tasks + self.holdout_tasks
        ]
        if missing:
            raise ValueError(f"eval_tasks contains unregistered tasks: {missing}")
        try:
            from data.tasks import list_task_ids

            registered = set(list_task_ids(training_only=False))
            unregistered = [
                task
                for task in self.training_tasks + self.holdout_tasks
                if task not in registered
            ]
            if unregistered:
                raise ValueError(f"Config references unregistered tasks: {unregistered}")
        except ImportError:
            pass

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ExperimentConfig":
        aliases = {
            "gamma_discount": "gamma",
            "bw_bonus_weight": "bandwidth_weight",
            "clip_eps_low": "epsilon",
            "clip_eps_high": "epsilon_high",
            "kl_coeff": "beta",
            "batch_tasks": "tasks_per_step",
            "train_turns": "num_turns",
        }
        normalized: Dict[str, Any] = {}
        for key, value in values.items():
            canonical = aliases.get(key, key)
            if key != canonical and canonical in values:
                continue
            normalized[canonical] = value
        known = {field_name for field_name in cls.__dataclass_fields__}
        unknown = sorted(set(normalized) - known)
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        return cls(**normalized)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        return cls.from_dict(values)

    def to_yaml(self, path: str) -> None:
        import yaml

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    def resolve_paths(self) -> None:
        if self.torch_extensions_dir is None:
            self.torch_extensions_dir = str(
                Path(self.output_dir_arm_a).resolve().parent / "torch_extensions"
            )
        Path(self.torch_extensions_dir).mkdir(parents=True, exist_ok=True)
        Path(self.archive_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.telemetry_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.eval_output_dir).mkdir(parents=True, exist_ok=True)


default_config = ExperimentConfig()
