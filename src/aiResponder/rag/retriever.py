"""Поиск похожих пар (prompt → response) для подмешивания в промпт.

Используется во время инференса: получив сообщение собеседника, ищем
наиболее похожие реплики, на которые целевой пользователь когда-то отвечал,
и складываем их как few-shot контекст.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from aiResponder.rag.embeddings import EmbeddingModel


@dataclass(slots=True)
class RagHit:
    """Одна найденная пара из индекса."""

    prompt: str
    response: str
    score: float


class RagRetriever:
    """Тонкая обёртка над FAISS-индексом с порогом сходства.

    Args:
        index_path: Путь к ``*.faiss``.
        meta_path: Путь к ``*.meta.jsonl`` (по строке на каждый prompt-response).
        embeddings: Эмбеддинг-модель, совместимая с тем, чем строили индекс.
        top_k: Сколько кандидатов возвращать.
        similarity_threshold: Минимальный cosine-сходства.
    """

    def __init__(
        self,
        index_path: Path,
        meta_path: Path,
        embeddings: EmbeddingModel,
        top_k: int = 4,
        similarity_threshold: float = 0.55,
    ) -> None:
        import faiss

        if not index_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"Индекс не найден: {index_path}")

        self.index = faiss.read_index(str(index_path))
        self.embeddings = embeddings
        self.top_k = max(1, int(top_k))
        self.threshold = float(similarity_threshold)

        self._meta: list[dict[str, str]] = []
        with meta_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                self._meta.append(json.loads(line))

        logger.info(
            "RAG-индекс готов: {} записей, top_k={}, threshold={:.2f}",
            len(self._meta),
            self.top_k,
            self.threshold,
        )

    def search(self, query: str) -> list[RagHit]:
        if not query or not self._meta:
            return []
        vec = self.embeddings.encode([query], kind="query")
        if vec.size == 0:
            return []
        scores, indices = self.index.search(vec.astype(np.float32, copy=False), self.top_k)
        hits: list[RagHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._meta):
                continue
            if float(score) < self.threshold:
                continue
            entry = self._meta[int(idx)]
            hits.append(
                RagHit(prompt=entry.get("prompt", ""), response=entry.get("response", ""), score=float(score))
            )
        return hits

    @staticmethod
    def index_paths_for_model(rag_dir: Path, model_name: str) -> tuple[Path, Path]:
        return rag_dir / f"{model_name}.faiss", rag_dir / f"{model_name}.meta.jsonl"
