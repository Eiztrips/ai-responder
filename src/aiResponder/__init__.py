"""AI Responder — Telegram-бот, имитирующий стиль общения пользователя.

Подсистемы:
    * ``aiResponder.config``    — типизированные настройки на pydantic-settings.
    * ``aiResponder.data``      — конвертация и фильтрация переписки, разрешение никнеймов.
    * ``aiResponder.ml``        — обучение модели (опц. LoRA) и автодетект устройства.
    * ``aiResponder.rag``       — Retrieval-Augmented Generation поверх FAISS.
    * ``aiResponder.inference`` — генерация ответов с RAG-контекстом.
    * ``aiResponder.bot``       — Pyrogram-клиент для Telegram.
    * ``aiResponder.cli``       — интерактивное меню запуска.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "3.0.0"
