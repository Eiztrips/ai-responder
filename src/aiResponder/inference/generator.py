"""RAG-aware генератор ответов.

Главные отличия от legacy-версии:

* Никаких ``functools.lru_cache`` на методах экземпляра (это держало ``self`` в
  глобальном кэше и текло). Кэш — поле экземпляра :class:`cachetools.LRUCache`.
* Авто-загрузка LoRA-адаптера, если он лежит рядом с моделью.
* Если для модели существует RAG-индекс — он подмешивается в промпт.
* Безопасный выбор устройства через :func:`detect_device`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from cachetools import LRUCache
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from aiResponder.config.settings import GenerationProfile, Settings
from aiResponder.ml.device import detect_device, prepare_runtime
from aiResponder.rag.embeddings import EmbeddingModel
from aiResponder.rag.retriever import RagHit, RagRetriever


@dataclass(slots=True)
class GenerationResult:
    """Результат одной генерации."""

    text: str
    used_rag: bool
    hits: list[RagHit]


class ResponseGenerator:
    """LLM-инференс с поддержкой RAG и кэширования последних ответов."""

    def __init__(self, settings: Settings, model_path: str | Path | None = None) -> None:
        self.settings = settings
        self.profile = detect_device(settings.main_settings.training_device,
                                     priority=settings.training.device_priority)
        prepare_runtime(self.profile)

        self.model: Any = None
        self.tokenizer: Any = None
        self.metadata: dict[str, Any] = {}
        self.profile_name = settings.inference.active_profile

        self._cache: LRUCache[tuple[str, str, int], str] = LRUCache(
            maxsize=max(8, settings.inference.cache.max_size)
        )
        self._retriever: RagRetriever | None = None
        self._embedding_model: EmbeddingModel | None = None
        self._models_dir: Path = settings.models_path()

        target = Path(model_path) if model_path else self._latest_model_path()
        if target is not None:
            self.load_model(target)
        else:
            logger.warning("Обученных моделей не найдено — load_model нужно вызвать вручную")

    # ------------------------------------------------------------------ #
    # Загрузка модели + RAG
    # ------------------------------------------------------------------ #
    def _latest_model_path(self) -> Path | None:
        if not self._models_dir.exists():
            return None
        dirs = [p for p in self._models_dir.iterdir() if p.is_dir()]
        if not dirs:
            return None
        return max(dirs, key=lambda p: p.stat().st_mtime)

    def load_model(self, model_path: str | Path) -> bool:
        try:
            path = Path(model_path)
            if self.model is not None:
                del self.model, self.tokenizer
                self.model = self.tokenizer = None
                if self.profile.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.profile.device == "mps" and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

            logger.info("Загружаю модель из {}", path)
            self.tokenizer = AutoTokenizer.from_pretrained(str(path))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            dtype = torch.float16 if self.profile.device == "cuda" else self.profile.dtype
            adapter_config = path / "adapter_config.json"
            if adapter_config.exists():
                from peft import AutoPeftModelForCausalLM

                self.model = AutoPeftModelForCausalLM.from_pretrained(
                    str(path),
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(path),
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
            self.model.to(self.profile.device)
            self.model.eval()

            self._cache.clear()
            meta_path = path / "metadata.json"
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as fh:
                    self.metadata = json.load(fh)
                logger.info("Модель для пользователя: {}", self.metadata.get("target_user", "?"))

            self._load_rag(path)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка загрузки модели: {}", exc)
            return False

    def _load_rag(self, model_path: Path) -> None:
        self._retriever = None
        rag_cfg = self.settings.rag
        if not rag_cfg.enabled:
            return
        rag_dir = self.settings.rag_index_path()
        index_path, meta_path = RagRetriever.index_paths_for_model(rag_dir, model_path.name)
        if not index_path.exists() or not meta_path.exists():
            logger.info("RAG-индекс для '{}' не найден — работаю без RAG", model_path.name)
            return
        try:
            emb_device = "cuda" if self.profile.device == "cuda" else "cpu"
            if self._embedding_model is None or self._embedding_model.model_name != rag_cfg.embeddings_model:
                self._embedding_model = EmbeddingModel(rag_cfg.embeddings_model, device=emb_device)
            self._retriever = RagRetriever(
                index_path=index_path,
                meta_path=meta_path,
                embeddings=self._embedding_model,
                top_k=rag_cfg.top_k,
                similarity_threshold=rag_cfg.similarity_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Не удалось поднять RAG-ретривер: {}", exc)
            self._retriever = None

    # ------------------------------------------------------------------ #
    # Профили генерации
    # ------------------------------------------------------------------ #
    def _get_profile(self) -> GenerationProfile:
        profiles = self.settings.inference.model.generation_profiles
        if self.profile_name in profiles:
            return profiles[self.profile_name]
        if profiles:
            first = next(iter(profiles))
            logger.warning("Профиль '{}' не найден — беру '{}'", self.profile_name, first)
            return profiles[first]
        return self.settings.inference.model.generation

    def set_profile(self, name: str) -> bool:
        if name in self.settings.inference.model.generation_profiles:
            self.profile_name = name
            self._cache.clear()
            return True
        return False

    def available_profiles(self) -> list[str]:
        return list(self.settings.inference.model.generation_profiles.keys())

    # ------------------------------------------------------------------ #
    # Промпт и пост-обработка
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_response(text: str) -> str:
        import re

        text = re.sub(r"@@[^@]*@@", "", text)
        text = re.sub(r"(?m)^[A-Z]:\s*", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _build_prompt(self, message: str) -> tuple[str, list[RagHit]]:
        hits: list[RagHit] = []
        if self._retriever is not None:
            hits = self._retriever.search(message)[: self.settings.rag.max_pairs_in_prompt]
        if not hits:
            return f"Q: {message}\nA:", hits
        examples = "\n".join(f"Q: {h.prompt}\nA: {h.response}" for h in hits)
        return f"{examples}\nQ: {message}\nA:", hits

    # ------------------------------------------------------------------ #
    # Публичный API
    # ------------------------------------------------------------------ #
    def generate(self, message: str, profile: str | None = None) -> GenerationResult:
        if not self.model or not self.tokenizer:
            return GenerationResult(
                text="Модель не загружена. Сначала обучите или выберите модель.",
                used_rag=False,
                hits=[],
            )
        active_profile = profile or self.profile_name
        cache_key = (message, active_profile, int(self._retriever is not None))
        if cache_key in self._cache:
            return GenerationResult(text=self._cache[cache_key], used_rag=bool(self._retriever), hits=[])

        gen = self.settings.inference.model.generation_profiles.get(active_profile) or self._get_profile()
        prompt, hits = self._build_prompt(message)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.profile.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=gen.max_length,
                num_return_sequences=gen.num_return_sequences,
                do_sample=gen.do_sample,
                temperature=gen.temperature,
                top_p=gen.top_p,
                no_repeat_ngram_size=gen.no_repeat_ngram_size,
                repetition_penalty=gen.repetition_penalty,
                length_penalty=gen.length_penalty,
                early_stopping=gen.early_stopping,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        answer = decoded.split("A:")[-1] if "A:" in decoded else decoded
        answer = self._clean_response(answer)
        self._cache[cache_key] = answer
        return GenerationResult(text=answer, used_rag=bool(hits), hits=hits)

    def generate_response(self, message: str, profile: str | None = None) -> str:
        """Совместимый со старым API метод — возвращает только текст."""
        return self.generate(message, profile=profile).text
