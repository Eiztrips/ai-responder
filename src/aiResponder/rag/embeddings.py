"""Кэшированная обёртка над sentence-transformers.

Для интерфейса с FAISS нам нужна L2-нормализованная матрица эмбеддингов
``float32``. Загрузка модели в текущем процессе кэшируется по ``model_name``,
чтобы между обучением и инференсом не дёргать диск повторно.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Iterable

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


_E5_PREFIX_PASSAGE = "passage: "
_E5_PREFIX_QUERY = "query: "


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, device: str) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    logger.info("Загружаю эмбеддинг-модель '{}' на {}", model_name, device)
    return SentenceTransformer(model_name, device=device)


class EmbeddingModel:
    """Тонкая обёртка над sentence-transformers с e5-prefix-логикой."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = _load_sentence_transformer(model_name, device)
        self.dim: int = int(self._model.get_sentence_embedding_dimension())
        self._is_e5 = "e5" in model_name.lower()

    def _decorate(self, texts: Iterable[str], *, kind: str) -> list[str]:
        if not self._is_e5:
            return list(texts)
        prefix = _E5_PREFIX_PASSAGE if kind == "passage" else _E5_PREFIX_QUERY
        return [prefix + (t or "") for t in texts]

    def encode(
        self,
        texts: Iterable[str],
        *,
        kind: str = "passage",
        batch_size: int = 32,
    ) -> np.ndarray:
        """Закодировать тексты в нормализованную матрицу ``float32``.

        Args:
            texts: Итерируемое строк.
            kind: ``passage`` для документов индекса, ``query`` для запросов.
            batch_size: Размер батча кодирования.
        """
        decorated = self._decorate(texts, kind=kind)
        if not decorated:
            return np.zeros((0, self.dim), dtype="float32")
        vectors = self._model.encode(
            decorated,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32", copy=False)
