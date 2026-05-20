"""Единая инициализация логирования через loguru.

Логи дублируются в stderr и (опционально) в файл. Стандартный ``logging``
перехватывается и пересылается в loguru, чтобы сообщения transformers, peft и
pyrogram писались единообразно.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from loguru import logger

_INITIALIZED = False


class _InterceptHandler(logging.Handler):
    """Перенаправляет записи stdlib-logging в loguru с сохранением уровня."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Сконфигурировать loguru.

    Args:
        level: Минимальный уровень (``DEBUG``/``INFO``/``WARNING``/``ERROR``).
        log_file: Если задан, логи дополнительно пишутся в файл с ротацией 10 МБ.
    """
    global _INITIALIZED
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        backtrace=False,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            "<level>{level: <8}</level> "
            "<cyan>{name}</cyan> - <level>{message}</level>"
        ),
    )
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level.upper(), rotation="10 MB", retention=5, enqueue=True)

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for noisy in ("transformers", "datasets", "peft", "accelerate", "pyrogram"):
        logging.getLogger(noisy).handlers = [_InterceptHandler()]
        logging.getLogger(noisy).propagate = False

    _INITIALIZED = True


def get_logger(name: str | None = None):
    """Возвращает loguru-logger c контекстом ``name``."""
    return logger.bind(name=name or "aiResponder")
