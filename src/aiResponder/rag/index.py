"""Сборка и сохранение FAISS-индекса по обучающим парам пользователя.

Каждой обученной модели соответствует один индекс:
``{rag_index_dir}/{model_name}.faiss`` + ``{model_name}.meta.jsonl``.
Индекс — плоский inner-product (после нормализации = cosine), достаточно
быстрый для десятков и сотен тысяч пар на CPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from loguru import logger

from aiResponder.data.processor import TrainingPair
from aiResponder.rag.embeddings import EmbeddingModel


@dataclass(slots=True)
class IndexArtifacts:
    """Артефакты сохранённого индекса."""

    index_path: Path
    meta_path: Path
    embeddings_model: str
    pair_count: int


class RagIndexBuilder:
    """Строит и сохраняет FAISS-индекс по парам (prompt → response)."""

    def __init__(self, embeddings: EmbeddingModel, output_dir: Path) -> None:
        self.embeddings = embeddings
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(self, pairs: Iterable[TrainingPair], model_name: str) -> IndexArtifacts:
        import faiss

        pair_list = list(pairs)
        if not pair_list:
            raise ValueError("Нельзя строить RAG-индекс без обучающих пар")

        logger.info("Кодирую {} пар для RAG-индекса '{}'", len(pair_list), model_name)
        vectors = self.embeddings.encode([p.prompt for p in pair_list], kind="passage")

        index = faiss.IndexFlatIP(self.embeddings.dim)
        index.add(vectors.astype(np.float32, copy=False))

        index_path = self.output_dir / f"{model_name}.faiss"
        meta_path = self.output_dir / f"{model_name}.meta.jsonl"

        faiss.write_index(index, str(index_path))
        with meta_path.open("w", encoding="utf-8") as fh:
            for pair in pair_list:
                fh.write(
                    json.dumps(
                        {"prompt": pair.prompt, "response": pair.response},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        logger.info("RAG-индекс сохранён в {}", index_path)
        return IndexArtifacts(
            index_path=index_path,
            meta_path=meta_path,
            embeddings_model=self.embeddings.model_name,
            pair_count=len(pair_list),
        )
