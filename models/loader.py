import logging
from pathlib import Path
from typing import Any, Optional, Tuple

from configs.grpo_config import ExperimentConfig, default_config

logger = logging.getLogger(__name__)


def _is_adapter_dir(path: Optional[str]) -> bool:
    return bool(path) and (Path(path) / "adapter_config.json").exists()


def _is_full_model_dir(path: Optional[str]) -> bool:
    return bool(path) and (Path(path) / "config.json").exists()


def _lora_config(config: ExperimentConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        target_modules=config.target_modules,
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(
    config: Optional[ExperimentConfig] = None,
    model_name_or_path: Optional[str] = None,
    is_trainable: bool = True,
    adapter_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    cfg = config or default_config
    target_model_name = model_name_or_path or cfg.get_active_model_name()
    sft_adapter_path = cfg.sft_adapter_path

    if _is_full_model_dir(sft_adapter_path) and not _is_adapter_dir(sft_adapter_path):
        target_model_name = sft_adapter_path
        sft_adapter_path = None

    try:
        import torch
        from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(f"Required model dependencies are missing: {exc}")

    tokenizer_kwargs = {
        "trust_remote_code": cfg.trust_remote_code,
        "padding_side": "left",
        "cache_dir": cfg.cache_dir,
        "local_files_only": cfg.local_files_only,
    }
    if cfg.model_revision:
        tokenizer_kwargs["revision"] = cfg.model_revision
    tokenizer = AutoTokenizer.from_pretrained(target_model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_cuda = torch.cuda.is_available()
    is_bf16_supported = has_cuda and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if is_bf16_supported else torch.float16
    device_map = cfg.device_map if cfg.device_map else (cfg.device if has_cuda else "cpu")

    model_kwargs = {
        "device_map": device_map,
        "torch_dtype": compute_dtype if has_cuda else torch.float32,
        "trust_remote_code": cfg.trust_remote_code,
        "cache_dir": cfg.cache_dir,
        "local_files_only": cfg.local_files_only,
    }
    if cfg.model_revision:
        model_kwargs["revision"] = cfg.model_revision
    if cfg.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.attn_implementation

    if has_cuda and cfg.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(target_model_name, **model_kwargs)

    if is_trainable and has_cuda and cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.gradient_checkpointing,
        )

    if sft_adapter_path:
        model = PeftModel.from_pretrained(
            model,
            sft_adapter_path,
            is_trainable=False,
            adapter_name="sft",
        )
        if is_trainable:
            model.add_adapter("rl", _lora_config(cfg))
            if adapter_path:
                model.load_adapter(adapter_path, adapter_name="rl", is_trainable=True)
            model.set_adapter("rl")
        elif adapter_path:
            model.load_adapter(adapter_path, adapter_name="rl", is_trainable=False)
            model.set_adapter("rl")
        else:
            model.set_adapter("sft")
    elif adapter_path:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=is_trainable,
            adapter_name="rl",
        )
        model.set_adapter("rl")
    elif is_trainable:
        model = get_peft_model(model, _lora_config(cfg))

    if is_trainable:
        _resync_trainable_dtype(model, dtype=torch.float32 if not is_bf16_supported else None)
        model.config.use_cache = False
        if cfg.gradient_checkpointing and not (has_cuda and cfg.load_in_4bit):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()
        verify_trainable_parameters(model, target_model_name)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = True

    return model, tokenizer


def _resync_trainable_dtype(model: Any, dtype: Any) -> None:
    if dtype is None:
        return
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(dtype)


def verify_trainable_parameters(model: Any, model_name: str) -> Tuple[int, int, float]:
    trainable_params = 0
    all_param = 0
    for _, parameter in model.named_parameters():
        all_param += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()

    percentage = 100 * trainable_params / all_param if all_param > 0 else 0.0
    print(
        f"Model: {model_name} | Trainable params: {trainable_params:,} | "
        f"All params: {all_param:,} | Trainable: {percentage:.4f}%"
    )

    if trainable_params == 0:
        logger.warning("No trainable parameters detected after adapter setup.")
    return trainable_params, all_param, percentage
