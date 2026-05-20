"""Опциональная обёртка над PEFT/LoRA.

LoRA даёт двух- и трёхкратное ускорение обучения и снижает требования к памяти —
особенно важно для MPS/CPU. Конфиг живёт в :class:`LoraConfig`, target-модули
автодетектятся по архитектуре через карту из peft.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from aiResponder.config.settings import LoraConfig

if TYPE_CHECKING:
    from transformers import PreTrainedModel


_DEFAULT_TARGETS = {
    "gpt2": ["c_attn"],
    "gpt_neox": ["query_key_value"],
    "llama": ["q_proj", "v_proj"],
    "mistral": ["q_proj", "v_proj"],
    "qwen2": ["q_proj", "v_proj"],
    "falcon": ["query_key_value"],
}


def maybe_apply_lora(model: "PreTrainedModel", cfg: LoraConfig) -> "PreTrainedModel":
    """Применить LoRA к модели, если включено в конфиге, иначе вернуть модель как есть."""
    if not cfg.enabled:
        return model

    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType, get_peft_model

    targets = cfg.target_modules
    if not targets:
        model_type = getattr(model.config, "model_type", "") or ""
        targets = _DEFAULT_TARGETS.get(model_type)
    if not targets:
        targets = ["c_attn", "q_proj", "v_proj", "query_key_value"]
        logger.warning(
            "Не найден маппинг LoRA-target_modules для {}; беру универсальный набор {}",
            getattr(model.config, "model_type", "?"),
            targets,
        )

    peft_cfg = PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        bias=cfg.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
    )
    model = get_peft_model(model, peft_cfg)
    logger.info("LoRA включена: r={}, alpha={}, targets={}", cfg.r, cfg.alpha, targets)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model
