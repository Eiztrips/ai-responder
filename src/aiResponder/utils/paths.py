"""Стандартизованные пути проекта.

Корень определяется относительно местоположения пакета и резервно — через
переменную окружения ``AI_RESPONDER_HOME`` (актуально для запуска в контейнере).
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Возвращает корневой каталог проекта.

    Учитывает запуск как из исходников, так и из установленного пакета. Если
    задано ``AI_RESPONDER_HOME``, оно имеет наивысший приоритет.
    """
    env = os.getenv("AI_RESPONDER_HOME")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    return here.parents[3]


def config_path() -> Path:
    """Путь к ``config/config.yaml``."""
    return project_root() / "config" / "config.yaml"


def env_path() -> Path:
    """Путь к опциональному ``.env`` (если присутствует)."""
    return project_root() / ".env"


def data_dir() -> Path:
    """Каталог для исходных данных пользователя (``data/raw``)."""
    return project_root() / "data" / "raw"


def datasets_dir() -> Path:
    """Каталог для сгенерированных датасетов (``data/datasets``)."""
    return project_root() / "data" / "datasets"


def models_dir() -> Path:
    """Каталог для обученных моделей (``models``)."""
    return project_root() / "models"


def rag_index_dir() -> Path:
    """Каталог для RAG-индексов (``rag_index``)."""
    return project_root() / "rag_index"


def logs_dir() -> Path:
    """Каталог для лог-файлов (``logs``)."""
    return project_root() / "logs"
