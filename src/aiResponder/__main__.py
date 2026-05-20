"""Точка входа для ``python -m aiResponder``.

Делегирует управление функции :func:`aiResponder.cli.run`, которая поднимает
интерактивное меню и асинхронный event-loop. Эта обёртка вынесена в отдельный
модуль, чтобы запуск пакета был идиоматичным и не требовал shell-скриптов.
"""

from __future__ import annotations

from aiResponder.cli import run

if __name__ == "__main__":
    run()
