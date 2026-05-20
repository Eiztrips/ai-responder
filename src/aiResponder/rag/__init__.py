"""Retrieval-Augmented Generation поверх FAISS."""

from __future__ import annotations

from aiResponder.rag.embeddings import EmbeddingModel
from aiResponder.rag.index import RagIndexBuilder
from aiResponder.rag.retriever import RagHit, RagRetriever

__all__ = ["EmbeddingModel", "RagHit", "RagIndexBuilder", "RagRetriever"]
