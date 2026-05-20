"""Интерактивное CLI-меню AI-Responder.

Полностью заменяет старую папку ``start/`` — это единственная точка входа в
приложение. Без ``tkinter``: путь к JSON либо вводится вручную, либо берётся
из подкаталога ``data/raw/`` (автосканирование).

Запуск:

* ``python -m aiResponder``
* ``ai-responder`` (после ``pip install -e .``)
* через docker-compose: ``docker compose --profile cpu run --rm app``
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Sequence

from loguru import logger

from aiResponder.config.settings import Settings, load_settings
from aiResponder.data.processor import DataProcessor, DatasetInfo
from aiResponder.utils.logging import configure_logging
from aiResponder.utils.paths import data_dir, env_path, logs_dir


def _prompt(text: str) -> str:
    try:
        return input(text)
    except (EOFError, KeyboardInterrupt):
        return ""


def _print_header(title: str) -> None:
    print("\n" + "=" * 40)
    print(title.center(40))
    print("=" * 40)


def _print_main_menu() -> str:
    _print_header("AI-RESPONDER")
    print("1. Обучить модель")
    print("2. Конвертировать JSON в датасет")
    print("3. Список моделей")
    print("4. Выбрать модель")
    print("5. Запустить Telegram-бота")
    print("6. Настройки")
    print("7. Выход")
    print("=" * 40)
    return _prompt("Выберите опцию (1-7): ").strip()


def _select_json_file() -> Path | None:
    """Найти JSON-файл для конвертации.

    Алгоритм:
        1. Если в ``data/raw/`` лежит хотя бы один ``*.json``, показываем выбор.
        2. Иначе просим ввести абсолютный путь.
    """
    raw_dir = data_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(raw_dir.glob("*.json"))
    if candidates:
        print(f"\nНайдено JSON-файлов в {raw_dir}:")
        for i, p in enumerate(candidates, 1):
            print(f"  {i}. {p.name}")
        print("  0. Ввести путь вручную")
        choice = _prompt("Выберите файл: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1]
            if idx == 0:
                pass
    manual = _prompt("Введите путь к JSON-файлу: ").strip().strip('"')
    return Path(manual) if manual else None


# ---------------------------------------------------------------------- #
# Меню «Настройки»
# ---------------------------------------------------------------------- #
def _settings_menu(settings: Settings, trainer) -> None:
    while True:
        _print_header("НАСТРОЙКИ")
        print("1. Профиль генерации")
        print("2. Базовая модель")
        print("3. Режим Telegram-бота")
        print("4. Устройство для обучения")
        print("5. Сохранить и выйти")
        print("6. Назад без сохранения")
        choice = _prompt("Выбор: ").strip()
        if choice == "1":
            _change_generation_profile(settings)
        elif choice == "2":
            _change_base_model(settings)
        elif choice == "3":
            _change_telegram_mode(settings)
        elif choice == "4":
            _change_training_device(settings, trainer)
        elif choice == "5":
            settings.write_yaml()
            print("Сохранено.")
            return
        elif choice == "6":
            return


def _change_generation_profile(settings: Settings) -> None:
    profiles = list(settings.inference.model.generation_profiles.keys())
    if not profiles:
        print("В config.yaml нет ни одного профиля.")
        return
    active = settings.inference.active_profile
    print("\nПрофили:")
    for i, name in enumerate(profiles, 1):
        mark = "✓" if name == active else " "
        cfg = settings.inference.model.generation_profiles[name]
        print(f"  {i}. [{mark}] {name} (T={cfg.temperature}, max_len={cfg.max_length})")
    raw = _prompt("Номер профиля (0 — отмена): ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(profiles):
        settings.inference.active_profile = profiles[int(raw) - 1]
        settings.main_settings.active_generation_profile = settings.inference.active_profile
        print(f"Активный профиль: {settings.inference.active_profile}")


def _change_base_model(settings: Settings) -> None:
    print(f"Текущая модель: {settings.training.model}")
    value = _prompt("Новая модель (Enter — отмена): ").strip()
    if value:
        settings.training.model = value
        settings.main_settings.model = value


def _change_telegram_mode(settings: Settings) -> None:
    descriptions = settings.telegram.mode_descriptions or {
        "only_private_chats": "только личные",
        "only_channel_messages": "только группы/каналы",
        "stalker": "ответы конкретным пользователям",
    }
    modes = list(descriptions.items())
    active = settings.telegram.mode
    print("\nРежимы:")
    for i, (mode, desc) in enumerate(modes, 1):
        mark = "✓" if mode == active else " "
        print(f"  {i}. [{mark}] {mode} — {desc}")
    raw = _prompt("Номер режима (0 — отмена): ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(modes):
        settings.telegram.mode = modes[int(raw) - 1][0]
        settings.main_settings.telegram_mode = settings.telegram.mode


def _change_training_device(settings: Settings, trainer) -> None:
    devices = trainer.available_devices()
    if not devices:
        print("Нет доступных устройств.")
        return
    items = list(devices.items())
    current = trainer.current_device()
    print(f"\nТекущее устройство: {current}")
    for i, (dev_id, label) in enumerate(items, 1):
        mark = "✓" if dev_id == current else " "
        print(f"  {i}. [{mark}] {label} ({dev_id})")
    raw = _prompt("Номер устройства (0 — отмена): ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(items):
        dev_id = items[int(raw) - 1][0]
        if trainer.set_device(dev_id):
            settings.main_settings.training_device = dev_id
            print(f"Устройство переключено на {dev_id}")


# ---------------------------------------------------------------------- #
# Действия меню
# ---------------------------------------------------------------------- #
def _select_dataset(datasets: Sequence[DatasetInfo]) -> DatasetInfo | None:
    if not datasets:
        print("Датасеты не найдены. Сначала конвертируйте JSON (опция 2).")
        return None
    print("\nДоступные датасеты:")
    for i, ds in enumerate(datasets, 1):
        print(f"  {i}. {ds.name} ({ds.type})")
    raw = _prompt("Номер: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(datasets):
        return datasets[int(raw) - 1]
    return None


def _action_train(settings: Settings, processor: DataProcessor, trainer) -> None:
    datasets = processor.available_datasets()
    selected = _select_dataset(datasets)
    if selected is None:
        return
    messages = processor.load_dataset(selected)
    participants = processor.participants(messages)
    print("\nУчастники чата:")
    for i, p in enumerate(participants, 1):
        print(f"  {i}. {p.display}")
    raw = _prompt("Номер пользователя для имитации: ").strip()
    if not (raw.isdigit() and 1 <= int(raw) <= len(participants)):
        return
    target = participants[int(raw) - 1]
    print(f"Выбран: {target.display}")
    print(f"Устройство: {trainer.current_device()}")
    if _prompt("Начать обучение? (д/н): ").lower() not in ("д", "y", "yes", "да"):
        return
    result = trainer.train_model(selected, target.canonical_id)
    print(f"Готово: {result}")


def _action_convert(processor: DataProcessor) -> None:
    json_path = _select_json_file()
    if json_path is None or not json_path.exists():
        print("Файл не выбран.")
        return
    name = _prompt("Имя выходного датасета (Enter — авто): ").strip() or None
    artifacts = processor.parse_json_to_dataset(json_path, name)
    if artifacts is None:
        print("Конвертация не удалась.")
        return
    print(f"CSV : {artifacts.csv_path}")
    print(f"JSONL: {artifacts.jsonl_path}")
    print(f"Сообщений: {artifacts.message_count}")
    print(f"Фильтр  : {artifacts.filter_summary}")


def _action_list(settings: Settings, trainer) -> None:
    selected = settings.main_settings.selected_model
    models = trainer.list_trained_models()
    if not models:
        print("Обученных моделей нет.")
        return
    for i, m in enumerate(models, 1):
        meta = m.metadata
        active = " ✓" if selected and Path(selected).name == m.name else ""
        print(f"  {i}. {m.name}{active}")
        print(f"     пользователь: {meta.get('target_user', '—')}")
        print(f"     пар        : {meta.get('training_pairs_count', '—')}")
        print(f"     устройство : {meta.get('training_device', '—')}")
        rag = meta.get("rag") or {}
        print(f"     RAG         : {'on' if rag.get('enabled') else 'off'}")


def _action_select(settings: Settings, trainer) -> None:
    models = trainer.list_trained_models()
    if not models:
        print("Нет доступных моделей.")
        return
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m.name} (пользователь: {m.metadata.get('target_user', '—')})")
    raw = _prompt("Номер модели (0 — отмена): ").strip()
    if not (raw.isdigit() and 1 <= int(raw) <= len(models)):
        return
    chosen = models[int(raw) - 1]
    settings.main_settings.selected_model = str(chosen.path)
    settings.write_yaml()
    print(f"Выбрана модель: {chosen.name}")


async def _action_run_bot(settings: Settings) -> None:
    from aiResponder.bot.telegram import TelegramResponder

    model_path = settings.main_settings.selected_model
    if not model_path:
        print("Модель не выбрана. Будет использована последняя обученная.")
    try:
        responder = TelegramResponder(settings, model_path=model_path)
    except ValueError as exc:
        print(f"Ошибка конфигурации: {exc}")
        return
    try:
        await responder.start()
    except KeyboardInterrupt:
        await responder.stop()


# ---------------------------------------------------------------------- #
# Точка входа
# ---------------------------------------------------------------------- #
async def _main_async() -> None:
    settings = load_settings()
    configure_logging(settings.logging.level, log_file=logs_dir() / "ai-responder.log")
    if env_path().exists():
        logger.debug(".env найден: {}", env_path())

    processor = DataProcessor(settings)
    from aiResponder.ml.trainer import ModelTrainer

    trainer = ModelTrainer(settings, processor, device=settings.main_settings.training_device)

    while True:
        choice = _print_main_menu()
        if choice == "1":
            _action_train(settings, processor, trainer)
        elif choice == "2":
            _action_convert(processor)
        elif choice == "3":
            _action_list(settings, trainer)
        elif choice == "4":
            _action_select(settings, trainer)
        elif choice == "5":
            await _action_run_bot(settings)
        elif choice == "6":
            _settings_menu(settings, trainer)
        elif choice == "7":
            print("Выход.")
            return
        elif choice == "":
            return
        else:
            print("Введите число 1–7.")


def run() -> None:
    """Синхронная обёртка для console-script (см. ``pyproject.toml``)."""
    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
