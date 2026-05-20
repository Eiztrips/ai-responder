"""Тренер языковой модели с автодетектом устройства, LoRA и RAG-индексом.

Архитектура:

* :class:`ModelTrainer` инкапсулирует выбор устройства, загрузку модели,
  построение ``TrainingArguments`` (без ``eval()``-строк), сохранение модели,
  метаданных и опционального RAG-индекса.
* Тренировочные пары собираются ``DataProcessor.prepare_training_data``.
* eval-сплит 10% по умолчанию + ``EarlyStoppingCallback``.
* После обучения, если ``rag.enabled``, строится FAISS-индекс по парам.
"""

from __future__ import annotations

import inspect
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import transformers
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from aiResponder.config.settings import Settings, TrainingDeviceArgs
from aiResponder.data.processor import (
    DataProcessor,
    DatasetInfo,
    TrainingPair,
)
from aiResponder.ml.dataset import ConversationDataset
from aiResponder.ml.device import DeviceProfile, detect_device, prepare_runtime
from aiResponder.ml.lora import maybe_apply_lora
from aiResponder.rag.embeddings import EmbeddingModel
from aiResponder.rag.index import RagIndexBuilder


@dataclass(slots=True)
class TrainedModelInfo:
    """Описание модели, найденной на диске."""

    name: str
    path: Path
    metadata: dict[str, Any]


class ModelTrainer:
    """Высокоуровневый тренер языковой модели."""

    def __init__(
        self,
        settings: Settings,
        data_processor: DataProcessor,
        device: str | None = None,
        model: str | None = None,
    ) -> None:
        self.settings = settings
        self.data_processor = data_processor
        self.model_name = model or settings.training.model
        self.profile = detect_device(device or settings.main_settings.training_device,
                                     priority=settings.training.device_priority)
        prepare_runtime(self.profile)
        self.models_dir = settings.models_path()
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Управление устройством
    # ------------------------------------------------------------------ #
    def available_devices(self) -> dict[str, str]:
        from aiResponder.ml.device import available_devices

        names = self.settings.training.device_names
        return {name: names.get(name, name) for name, ok in available_devices().items() if ok}

    def set_device(self, device: str) -> bool:
        new_profile = detect_device(device, priority=self.settings.training.device_priority)
        self.profile = new_profile
        prepare_runtime(self.profile)
        return True

    def current_device(self) -> str:
        return self.profile.device

    # ------------------------------------------------------------------ #
    # Список обученных моделей
    # ------------------------------------------------------------------ #
    def list_trained_models(self) -> list[TrainedModelInfo]:
        if not self.models_dir.exists():
            return []
        out: list[TrainedModelInfo] = []
        for path in sorted(self.models_dir.iterdir()):
            if not path.is_dir():
                continue
            meta_path = path / "metadata.json"
            metadata: dict[str, Any] = {}
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as fh:
                    metadata = json.load(fh)
            out.append(TrainedModelInfo(name=path.name, path=path, metadata=metadata))
        return out

    # ------------------------------------------------------------------ #
    # Подготовка TrainingArguments
    # ------------------------------------------------------------------ #
    def _resolve_args(self, output_dir: Path) -> TrainingArguments:
        device_args = self.settings.training.args.get(self.profile.device)
        if device_args is None:
            logger.warning(
                "Нет training-args для устройства {} в config.yaml; беру дефолт",
                self.profile.device,
            )
            device_args = TrainingDeviceArgs()
        params = device_args.model_dump()
        # допускаем строковый плейсхолдер "auto" для воркеров
        workers = params.get("dataloader_num_workers", 0)
        if isinstance(workers, str):
            params["dataloader_num_workers"] = (
                self.profile.recommended_workers if workers.lower() == "auto" else 0
            )
        # bf16 на устройствах, где это безопасно
        if self.profile.device == "cuda":
            params["bf16"] = bool(params.get("bf16")) and self.profile.supports_bf16
            params["fp16"] = bool(params.get("fp16")) and not params["bf16"] and self.profile.supports_fp16
        else:
            params["bf16"] = False
            params["fp16"] = False
        # torch.compile только если устройство реально поддерживает
        params["torch_compile"] = bool(params.get("torch_compile")) and self.profile.supports_compile
        params.setdefault("gradient_checkpointing", self.profile.gradient_checkpointing)

        params["output_dir"] = str(output_dir)
        params.setdefault("logging_dir", str(output_dir / "logs"))
        # eval включается ниже в train_model (eval_strategy/eval_steps)
        params.setdefault("eval_strategy", "no")

        # Защищаемся от drift'а API transformers: в 4.46+ удалены некоторые
        # «исторические» kwargs (например `overwrite_output_dir`). Фильтруем
        # `params` по реальной сигнатуре `TrainingArguments.__init__` и
        # логируем то, что пришлось выбросить — чтобы это сразу было видно
        # в diagnostics, а не падало в TypeError.
        supported = set(inspect.signature(TrainingArguments.__init__).parameters)
        dropped = [k for k in list(params) if k not in supported]
        for k in dropped:
            params.pop(k, None)

        # #region agent log
        try:
            _payload = {
                "sessionId": "9da7f4",
                "runId": "training-args",
                "hypothesisId": "A+B+C",
                "location": "src/aiResponder/ml/trainer.py:_resolve_args",
                "message": "TrainingArguments resolved",
                "data": {
                    "transformers_version": transformers.__version__,
                    "device": self.profile.device,
                    "params_keys": sorted(params.keys()),
                    "dropped_unsupported": sorted(dropped),
                },
                "timestamp": int(time.time() * 1000),
            }
            with open("/app/logs/debug-9da7f4.log", "a", encoding="utf-8") as _fh:
                _fh.write(json.dumps(_payload) + "\n")
        except Exception:
            pass
        # #endregion

        if dropped:
            logger.warning(
                "TrainingArguments: отброшены неподдерживаемые ключи {} (transformers={})",
                dropped,
                transformers.__version__,
            )
        return TrainingArguments(**params)

    # ------------------------------------------------------------------ #
    # Основной метод обучения
    # ------------------------------------------------------------------ #
    def train_model(self, dataset: DatasetInfo, target_user_id: str) -> str:
        logger.info(
            "Старт обучения: датасет='{}', target_user_id='{}', device={}",
            dataset.name,
            target_user_id,
            self.profile.device,
        )
        logger.info("Доступная память: {}", self.profile.memory_summary())

        messages = self.data_processor.load_dataset(dataset)
        conversation, _ = self.data_processor.extract_conversation(messages)
        pairs = self.data_processor.prepare_training_data(conversation, target_user_id)
        participants = self.data_processor.participants(messages)
        target_participant = next(
            (p for p in participants if p.canonical_id == target_user_id),
            None,
        )
        target_user_name = target_participant.canonical_name if target_participant else "Неизвестный"

        min_pairs = self.settings.training.min_training_pairs
        if len(pairs) < min_pairs:
            msg = (
                f"Недостаточно данных для обучения: {len(pairs)} пар, нужно минимум {min_pairs}. "
                "Попробуйте смягчить фильтры или взять больший датасет."
            )
            logger.error(msg)
            return msg

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.profile.dtype,
            attn_implementation=self.profile.attn_implementation,
            low_cpu_mem_usage=True,
        )
        if hasattr(model, "config"):
            model.config.use_cache = False
        model.to(self.profile.device)
        model = maybe_apply_lora(model, self.settings.training.lora)
        if self.profile.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        random.seed(42)
        random.shuffle(pairs)
        split = max(1, int(len(pairs) * self.settings.training.eval_split))
        eval_pairs = pairs[:split] if len(pairs) > 20 else []
        train_pairs = pairs[split:] if eval_pairs else pairs

        train_ds = ConversationDataset(
            train_pairs,
            tokenizer,
            max_length=self.settings.training.dataset_max_length,
            prompt_format=self.settings.training.dataset_prompt_format,
        )
        eval_ds = (
            ConversationDataset(
                eval_pairs,
                tokenizer,
                max_length=self.settings.training.dataset_max_length,
                prompt_format=self.settings.training.dataset_prompt_format,
            )
            if eval_pairs
            else None
        )

        safe_name = (target_user_name or "user").replace(" ", "_").replace("/", "_")
        model_basename = self.model_name.split("/")[-1]
        save_name = self.settings.training.model_name_format.format(user=safe_name, model=model_basename)
        save_path = self.models_dir / save_name
        save_path.mkdir(parents=True, exist_ok=True)

        training_args = self._resolve_args(save_path)
        if eval_ds is not None:
            # `eval_strategy` — каноничное имя в transformers >= 4.41 (старое
            # `evaluation_strategy` остаётся как deprecated-алиас).
            training_args.eval_strategy = "steps"
            training_args.eval_steps = max(training_args.logging_steps, 50)
            training_args.load_best_model_at_end = True
            training_args.metric_for_best_model = "eval_loss"
            training_args.greater_is_better = False

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        callbacks = []
        if eval_ds is not None and self.settings.training.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.settings.training.early_stopping_patience
                )
            )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collator,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )

        resume = self.settings.training.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=resume if resume else None)
        trainer.save_model()
        tokenizer.save_pretrained(save_path)

        rag_meta: dict[str, Any] | None = None
        if self.settings.rag.enabled and self.settings.rag.rebuild_on_train:
            try:
                emb_device = "cuda" if self.profile.device == "cuda" else "cpu"
                emb = EmbeddingModel(self.settings.rag.embeddings_model, device=emb_device)
                builder = RagIndexBuilder(emb, self.settings.rag_index_path())
                rag_artifacts = builder.build(pairs, save_name)
                rag_meta = {
                    "enabled": True,
                    "embeddings_model": rag_artifacts.embeddings_model,
                    "index_path": str(rag_artifacts.index_path),
                    "meta_path": str(rag_artifacts.meta_path),
                    "pair_count": rag_artifacts.pair_count,
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("Не удалось построить RAG-индекс: {}", exc)
                rag_meta = {"enabled": False, "error": str(exc)}

        metadata: dict[str, Any] = {
            "target_user": target_user_name,
            "target_user_id": target_user_id,
            "username": getattr(target_participant, "username", None),
            "aliases": getattr(target_participant, "aliases", []),
            "source_file": dataset.name,
            "source_file_type": dataset.type,
            "training_pairs_count": len(pairs),
            "model_base": self.model_name,
            "training_device": self.profile.device,
            "lora_enabled": bool(self.settings.training.lora.enabled),
            "rag": rag_meta,
            "training_timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with (save_path / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)

        del trainer, model, train_ds, eval_ds
        if self.profile.device == "cuda":
            torch.cuda.empty_cache()
        elif self.profile.device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        logger.info("Обучение завершено, модель сохранена в {}", save_path)
        return str(save_path)

    @property
    def model_dir(self) -> str:
        """Совместимость с CLI — путь к каталогу обученных моделей."""
        return str(self.models_dir)


def free_disk_for_models(models_dir: Path) -> str:
    """Хелпер: ёмкость каталога моделей (читаемая строка)."""
    try:
        usage = os.statvfs(models_dir)  # type: ignore[attr-defined]
        free = usage.f_bavail * usage.f_frsize
        return f"{free / 1024**3:.1f} GiB"
    except (AttributeError, OSError):
        return "n/a"
