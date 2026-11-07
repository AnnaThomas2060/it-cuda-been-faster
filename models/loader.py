import logging
from typing import Any, Optional, Tuple

from configs.grpo_config import ExperimentConfig, default_config

# Model and tokenizer loading engine configuring 4-bit NF4 quantization and LoRA adapters.
# Targets attention projections (q_proj, k_proj, v_proj, o_proj) with rank 8 and alpha 16.
# Verifies exact trainable parameter counts against the MSAI635 architectural baseline.

logger = logging.getLogger(__name__)

EXPECTED_TRAINABLE_PARAMS = 6_684_672

def load_model_and_tokenizer(
    config: Optional[ExperimentConfig] = None,
    model_name_or_path: Optional[str] = None,
    is_trainable: bool = True,
) -> Tuple[Any, Any]:
    cfg = config or default_config
    target_model_name = model_name_or_path or cfg.get_active_model_name()

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        raise ImportError(f"Required model dependencies are missing: {e}")

    # Set up left padding for autoregressive generation
    tokenizer = AutoTokenizer.from_pretrained(
        target_model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_cuda = torch.cuda.is_available()

    if has_cuda and cfg.load_in_4bit:
        # 4-bit NF4 quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        if not has_cuda:
            logger.warning(
                "CUDA is not available. Loading model in full/bfloat16 precision on CPU."
            )
        model = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            device_map="auto" if has_cuda else "cpu",
            torch_dtype=torch.bfloat16 if has_cuda else torch.float32,
            trust_remote_code=True,
        )

    if is_trainable:
        if has_cuda and cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=cfg.gradient_checkpointing,
            )

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.lora_bias,
            target_modules=cfg.target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        verify_trainable_parameters(model, target_model_name)

    return model, tokenizer

def verify_trainable_parameters(model: Any, model_name: str) -> Tuple[int, int, float]:
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    percentage = 100 * trainable_params / all_param if all_param > 0 else 0.0
    print(
        f"Model: {model_name} | Trainable params: {trainable_params:,} | "
        f"All params: {all_param:,} | Trainable: {percentage:.4f}%"
    )

    if "30B" in model_name and trainable_params != EXPECTED_TRAINABLE_PARAMS:
        logger.warning(
            f"Trainable param count {trainable_params} deviates from "
            f"expected {EXPECTED_TRAINABLE_PARAMS}."
        )

    return trainable_params, all_param, percentage
