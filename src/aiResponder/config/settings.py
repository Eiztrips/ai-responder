"""РўРёРїРёР·РёСЂРѕРІР°РЅРЅС‹Рµ РЅР°СЃС‚СЂРѕР№РєРё РїСЂРёР»РѕР¶РµРЅРёСЏ РЅР° pydantic-settings.

РСЃС‚РѕС‡РЅРёРє РґР°РЅРЅС‹С…:
    1. ``config/config.yaml`` вЂ” РІСЃРµ РѕСЃРЅРѕРІРЅС‹Рµ РїР°СЂР°РјРµС‚СЂС‹.
    2. ``.env`` / РїРµСЂРµРјРµРЅРЅС‹Рµ РѕРєСЂСѓР¶РµРЅРёСЏ вЂ” С‡СѓРІСЃС‚РІРёС‚РµР»СЊРЅС‹Рµ Р·РЅР°С‡РµРЅРёСЏ
       (``API_ID``, ``API_HASH``, ``PHONE``, ``LOGIN``, Рё С‚.Рґ.).

YAML СЏРІР»СЏРµС‚СЃСЏ РѕСЃРЅРѕРІРѕР№, РїРµСЂРµРјРµРЅРЅС‹Рµ РѕРєСЂСѓР¶РµРЅРёСЏ РїРµСЂРµРѕРїСЂРµРґРµР»СЏСЋС‚ С‚РѕР»СЊРєРѕ СЃРµРєС†РёСЋ
``telegram`` (api_id/api_hash/phone/login Рё СЃРїРёСЃРєРё С†РµР»РµР№). РўР°Рє СѓС…РѕРґРёС‚ РІРµСЃСЊ
``yaml.safe_load`` Рё С…СЂСѓРїРєРёРµ ``eval()`` РёР· СЃС‚Р°СЂРѕРіРѕ РєРѕРґР°.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aiResponder.utils.paths import (
    config_path,
    datasets_dir,
    env_path,
    models_dir,
    rag_index_dir,
)


class AppConfig(BaseModel):
    """РњРµС‚Р°РґР°РЅРЅС‹Рµ РїСЂРёР»РѕР¶РµРЅРёСЏ."""

    model_config = ConfigDict(extra="allow")

    name: str = "AI-Responder"
    version: str = "3.0.0"


class LoggingConfig(BaseModel):
    """РџР°СЂР°РјРµС‚СЂС‹ Р»РѕРіРёСЂРѕРІР°РЅРёСЏ."""

    model_config = ConfigDict(extra="allow")

    level: str = "INFO"
    file: str | None = None


class TelegramConfig(BaseModel):
    """РџР°СЂР°РјРµС‚СЂС‹ Telegram-РєР»РёРµРЅС‚Р° (Pyrogram)."""

    model_config = ConfigDict(extra="allow")

    api_id: int | None = None
    api_hash: str | None = None
    phone: str | None = None
    login: str | None = None
    target_user_ids: list[int] = Field(default_factory=lambda: [-1])
    target_channel_ids: list[int] = Field(default_factory=lambda: [-1])
    mode: str = "only_private_chats"
    mode_descriptions: dict[str, str] = Field(default_factory=dict)


class GenerationProfile(BaseModel):
    """РџСЂРѕС„РёР»СЊ СЃРµРјРїР»РёРЅРіР° РґР»СЏ ``model.generate``."""

    model_config = ConfigDict(extra="allow")

    max_length: int = 60
    num_return_sequences: int = 1
    do_sample: bool = True
    temperature: float = 0.9
    top_p: float = 0.95
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    early_stopping: bool = True


class InferenceCacheConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    max_size: int = 256


class InferenceModelConfig(BaseModel):
    """Р“РґРµ РёСЃРєР°С‚СЊ РѕР±СѓС‡РµРЅРЅС‹Рµ РјРѕРґРµР»Рё Рё РєР°РєРёРµ РїСЂРѕС„РёР»Рё РіРµРЅРµСЂР°С†РёРё РґРѕСЃС‚СѓРїРЅС‹."""

    model_config = ConfigDict(extra="allow")

    models_dir: str = "models"
    generation_profiles: dict[str, GenerationProfile] = Field(default_factory=dict)
    generation: GenerationProfile = Field(default_factory=GenerationProfile)


class InferenceConfig(BaseModel):
    """РќР°СЃС‚СЂРѕР№РєРё РёРЅС„РµСЂРµРЅСЃР°."""

    model_config = ConfigDict(extra="allow")

    active_profile: str = "balanced"
    model: InferenceModelConfig = Field(default_factory=InferenceModelConfig)
    cache: InferenceCacheConfig = Field(default_factory=InferenceCacheConfig)


class FiltersConfig(BaseModel):
    """Р­РІСЂРёСЃС‚РёРєРё С„РёР»СЊС‚СЂР°С†РёРё РјСѓСЃРѕСЂР° РІ РёСЃС…РѕРґРЅРѕР№ РїРµСЂРµРїРёСЃРєРµ.

    Р”РµС„РѕР»С‚С‹ РїРѕРґРѕР±СЂР°РЅС‹ РјСЏРіРєРѕ: РєРѕСЂРѕС‚РєРёРµ С‡РµР»РѕРІРµС‡РµСЃРєРёРµ СЂРµРїР»РёРєРё (В«РѕРєВ», В«+В», В«РґР°В»)
    СЃРѕС…СЂР°РЅСЏСЋС‚СЃСЏ, Р° СѓРґР°Р»СЏРµС‚СЃСЏ С‚РѕР»СЊРєРѕ РѕС‡РµРІРёРґРЅС‹Р№ С€СѓРј.
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    min_words: int = 1
    drop_emoji_only: bool = True
    drop_url_only: bool = True
    drop_pure_punct_or_digits: bool = True
    drop_repeat_spam: bool = True
    repeat_threshold: int = 6
    drop_duplicate_window: int = 5
    drop_system_messages: bool = True
    require_letters: bool = True
    allow_short_replies: bool = True
    short_reply_whitelist: list[str] = Field(
        default_factory=lambda: [
            "РѕРє",
            "+",
            "+1",
            "РґР°",
            "РЅРµС‚",
            "РЅСѓ",
            "С…Рј",
            "СЃРѕРіР»",
            "Р°РіР°",
            "РѕРє.",
            "spas",
            "СЃРїСЃ",
            "Р»РѕР»",
            "РѕРє!",
        ]
    )


class NicknameConfig(BaseModel):
    """РќРѕСЂРјР°Р»РёР·Р°С†РёСЏ Рё СЃРєР»РµР№РєР° Р°Р»РёР°СЃРѕРІ."""

    model_config = ConfigDict(extra="allow")

    fuzzy_merge: bool = True
    similarity_threshold: float = 0.92
    strip_emoji: bool = True


class DataProcessorConfig(BaseModel):
    """РџР°СЂР°РјРµС‚СЂС‹ РєРѕРЅРІРµСЂС‚Р°С†РёРё/С‡РёСЃС‚РєРё РґР°РЅРЅС‹С…."""

    model_config = ConfigDict(extra="allow")

    emoji_pattern: str = (
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\u200d\u2640-\u2642\u2600-\u2B55\u23cf\u23e9\u231a\u3030\ufe0f"
        "]+"
    )
    url_pattern: str = r"https?://\S+|www\.\S+"
    mention_hashtag_pattern: str = r"@\S+|#\S+"
    whitespace_pattern: str = r"\s+"
    control_chars_pattern: str = r"[\x00-\x1F\x7F-\x9F]"
    html_tags_pattern: str = r"<[^>]+>"
    system_message_pattern: str = r"^(joined|left|pinned|changed|removed|added|created)"
    datasets_dir: str = "data/datasets"


class LoraConfig(BaseModel):
    """LoRA-РїР°СЂР°РјРµС‚СЂС‹ РґР»СЏ PEFT."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] | None = None


class TrainingDeviceArgs(BaseModel):
    """``TrainingArguments`` РґР»СЏ РєРѕРЅРєСЂРµС‚РЅРѕРіРѕ СѓСЃС‚СЂРѕР№СЃС‚РІР°.

    РџРѕР»СЏ СЃРѕРІРїР°РґР°СЋС‚ СЃ РєР»СЋС‡Р°РјРё ``transformers.TrainingArguments``. Р—РЅР°С‡РµРЅРёСЏ С…СЂР°РЅСЏС‚СЃСЏ
    РєР°Рє РѕР±С‹С‡РЅС‹Рµ РїРёС‚РѕРЅРѕРІС‹Рµ С‚РёРїС‹ вЂ” Р±РµР· ``eval``-СЃС‚СЂРѕРє, РєР°Рє Р±С‹Р»Рѕ РІ legacy-РєРѕРЅС„РёРіРµ.
    """

    model_config = ConfigDict(extra="allow")

    num_train_epochs: float = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    save_steps: int = 500
    save_total_limit: int = 1
    logging_steps: int = 50
    fp16: bool = False
    bf16: bool = False
    optim: str = "adamw_torch"
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    report_to: str = "none"
    dataloader_num_workers: int | str = 0  # РґРѕРїСѓСЃРєР°РµС‚СЃСЏ "auto"
    gradient_checkpointing: bool = True
    torch_compile: bool = False


class TrainingConfig(BaseModel):
    """РљРѕСЂРЅРµРІР°СЏ СЃРµРєС†РёСЏ РѕР±СѓС‡РµРЅРёСЏ."""

    model_config = ConfigDict(extra="allow")

    model: str = "ai-forever/rugpt3medium_based_on_gpt2"
    min_training_pairs: int = 5
    model_name_format: str = "{user}_{model}"
    dataset_max_length: int = 512
    dataset_prompt_format: str = "Q: {question}\nA: {answer}"
    eval_split: float = 0.1
    early_stopping_patience: int = 2
    resume_from_checkpoint: bool = False
    device_priority: list[str] = Field(default_factory=lambda: ["cuda", "mps", "cpu"])
    device_names: dict[str, str] = Field(
        default_factory=lambda: {
            "cpu": "РџСЂРѕС†РµСЃСЃРѕСЂ (CPU)",
            "cuda": "NVIDIA GPU (CUDA)",
            "mps": "Apple Silicon GPU (MPS)",
        }
    )
    args: dict[str, TrainingDeviceArgs] = Field(default_factory=dict)
    lora: LoraConfig = Field(default_factory=LoraConfig)


class RagConfig(BaseModel):
    """РџР°СЂР°РјРµС‚СЂС‹ RAG."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    embeddings_model: str = "intfloat/multilingual-e5-small"
    top_k: int = 4
    similarity_threshold: float = 0.55
    index_dir: str = "rag_index"
    rebuild_on_train: bool = True
    max_pairs_in_prompt: int = 4


class MainSettings(BaseModel):
    """Runtime-РЅР°СЃС‚СЂРѕР№РєРё, РѕР±С‹С‡РЅРѕ РјРµРЅСЏРµРјС‹Рµ С‡РµСЂРµР· РјРµРЅСЋ."""

    model_config = ConfigDict(extra="allow")

    active_generation_profile: str = "balanced"
    model: str = "ai-forever/rugpt3medium_based_on_gpt2"
    telegram_mode: str = "only_private_chats"
    training_device: str = "auto"
    selected_model: str | None = None


class Settings(BaseSettings):
    """РџРѕР»РЅР°СЏ РєРѕРЅС„РёРіСѓСЂР°С†РёСЏ РїСЂРёР»РѕР¶РµРЅРёСЏ.

    РЎРѕР·РґР°С‘С‚СЃСЏ С‡РµСЂРµР· :func:`load_settings`, РєРѕС‚РѕСЂС‹Р№ РѕР±СЉРµРґРёРЅСЏРµС‚ РґР°РЅРЅС‹Рµ РёР·
    ``config/config.yaml`` Рё РїРµСЂРµРјРµРЅРЅС‹С… РѕРєСЂСѓР¶РµРЅРёСЏ.
    """

    model_config = SettingsConfigDict(
        env_file=str(env_path()) if env_path().exists() else None,
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="allow",
    )

    app: AppConfig = Field(default_factory=AppConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    data_processor: DataProcessorConfig = Field(default_factory=DataProcessorConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    nicknames: NicknameConfig = Field(default_factory=NicknameConfig)
    main_settings: MainSettings = Field(default_factory=MainSettings)

    @field_validator("training", mode="before")
    @classmethod
    def _ensure_training_args(cls, value: Any) -> Any:  # noqa: D401
        """РџСЂРёРЅСЏС‚СЊ ``args`` РєР°Рє dict СѓСЃС‚СЂРѕР№СЃС‚РІРѕв†’dict, Р±РµР· eval-СЃС‚СЂРѕРє."""
        if isinstance(value, dict) and "args" not in value:
            value["args"] = {}
        return value

    def write_yaml(self, path: Path | None = None) -> Path:
        """РЎРµСЂРёР°Р»РёР·РѕРІР°С‚СЊ РЅР°СЃС‚СЂРѕР№РєРё РІ YAML.

        Р—Р°РїРёСЃСЊ РёРґС‘С‚ РІ СѓРєР°Р·Р°РЅРЅС‹Р№ РїСѓС‚СЊ Р»РёР±Рѕ РІ :func:`config_path`. Р’РѕР·РІСЂР°С‰Р°РµС‚ РїСѓС‚СЊ.
        """
        target = path or config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(
                self.model_dump(mode="json"),
                fh,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
        return target

    def datasets_path(self) -> Path:
        """РљР°С‚Р°Р»РѕРі РґР°С‚Р°СЃРµС‚РѕРІ (СѓС‡РёС‚С‹РІР°СЏ РІРѕР·РјРѕР¶РЅС‹Р№ override РІ YAML)."""
        configured = self.data_processor.datasets_dir
        path = Path(configured)
        return path if path.is_absolute() else datasets_dir().parent / Path(configured).name

    def models_path(self) -> Path:
        """РљР°С‚Р°Р»РѕРі РѕР±СѓС‡РµРЅРЅС‹С… РјРѕРґРµР»РµР№."""
        configured = self.inference.model.models_dir
        path = Path(configured)
        return path if path.is_absolute() else models_dir()

    def rag_index_path(self) -> Path:
        """РљР°С‚Р°Р»РѕРі RAG-РёРЅРґРµРєСЃРѕРІ."""
        configured = self.rag.index_dir
        path = Path(configured)
        return path if path.is_absolute() else rag_index_dir()


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """РќР°РєР°С‚РёС‚СЊ С‡СѓРІСЃС‚РІРёС‚РµР»СЊРЅС‹Рµ env-РїРµСЂРµРјРµРЅРЅС‹Рµ РїРѕРІРµСЂС… YAML."""
    telegram = data.setdefault("telegram", {})
    env_api_id = os.getenv("API_ID")
    env_api_hash = os.getenv("API_HASH")
    env_phone = os.getenv("PHONE")
    env_login = os.getenv("LOGIN")
    env_users = os.getenv("TARGET_USER_IDS")
    env_channels = os.getenv("TARGET_CHANNEL_IDS")

    if env_api_id:
        try:
            telegram["api_id"] = int(env_api_id)
        except ValueError:
            pass
    if env_api_hash:
        telegram["api_hash"] = env_api_hash
    if env_phone:
        telegram["phone"] = env_phone
    if env_login:
        telegram["login"] = env_login
    if env_users:
        telegram["target_user_ids"] = [
            int(x.strip()) for x in env_users.split(",") if x.strip().lstrip("-").isdigit()
        ]
    if env_channels:
        telegram["target_channel_ids"] = [
            int(x.strip()) for x in env_channels.split(",") if x.strip().lstrip("-").isdigit()
        ]
    return data


def load_settings(path: Path | None = None) -> Settings:
    """РџСЂРѕС‡РёС‚Р°С‚СЊ РЅР°СЃС‚СЂРѕР№РєРё РёР· YAML + .env Рё РІРµСЂРЅСѓС‚СЊ :class:`Settings`.

    Args:
        path: РђР»СЊС‚РµСЂРЅР°С‚РёРІРЅС‹Р№ РїСѓС‚СЊ Рє YAML-С„Р°Р№Р»Сѓ. РџРѕ СѓРјРѕР»С‡Р°РЅРёСЋ вЂ” ``config/config.yaml``.
    """
    yaml_path = path or config_path()
    raw: dict[str, Any] = {}
    if yaml_path.exists():
        with yaml_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    raw = _apply_env_overrides(raw)
    return Settings.model_validate(raw)
