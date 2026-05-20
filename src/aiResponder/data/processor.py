"""Обработка переписки: парсинг Telegram-JSON, чистка текста, генерация датасетов.

В отличие от legacy-версии:

* всё работает через :class:`pathlib.Path`;
* возвращаются dataclass-структуры, а не «голые» словари;
* участники чата проходят через :class:`AliasResolver` (см. ``data.nicknames``);
* фильтрация мусора делегируется :class:`GarbageFilter` (см. ``data.filters``);
* из Telegram-JSON вытаскиваются ``from_id`` и поле ``via_bot``/``forwarded_from``,
  чтобы корректно разделять алиасы тёзок.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from loguru import logger

from aiResponder.config.settings import Settings
from aiResponder.data.filters import GarbageFilter
from aiResponder.data.nicknames import AliasResolver, NicknameNormalizer, Participant


@dataclass(slots=True)
class RawMessage:
    """Сообщение чата после первичного парсинга, до фильтрации."""

    author: str
    author_id: str
    username: str | None
    text: str
    date: str | None = None


@dataclass(slots=True)
class CleanMessage:
    """Сообщение после нормализации/фильтрации."""

    from_name: str
    from_id: str
    text: str


@dataclass(slots=True)
class DatasetArtifacts:
    """Артефакты, полученные после ``parse_json_to_dataset``."""

    csv_path: Path
    jsonl_path: Path
    name: str
    message_count: int
    filter_summary: str


@dataclass(slots=True)
class TrainingPair:
    """Пара (prompt, response) для языкового моделирования."""

    prompt: str
    response: str


@dataclass(slots=True)
class DatasetInfo:
    """Описание найденного датасета на диске."""

    name: str
    path: Path
    type: str  # "csv" | "jsonl"


class DataProcessor:
    """Высокоуровневая «фабрика» датасетов и обучающих пар."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._normalizer = NicknameNormalizer(
            settings.data_processor, strip_emoji=settings.nicknames.strip_emoji
        )
        self._alias_resolver = AliasResolver(settings.nicknames, self._normalizer)

        regex = settings.data_processor
        self._url_re = re.compile(regex.url_pattern, flags=re.IGNORECASE)
        self._mention_re = re.compile(regex.mention_hashtag_pattern)
        self._whitespace_re = re.compile(regex.whitespace_pattern)
        self._control_re = re.compile(regex.control_chars_pattern)
        self._html_re = re.compile(regex.html_tags_pattern)
        self._emoji_re = re.compile(regex.emoji_pattern, flags=re.UNICODE)

        self.datasets_dir = settings.datasets_path()
        self.csv_dir = self.datasets_dir / "csv"
        self.jsonl_dir = self.datasets_dir / "jsonl"
        for d in (self.csv_dir, self.jsonl_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Поиск и загрузка готовых датасетов
    # ------------------------------------------------------------------ #
    def available_datasets(self) -> list[DatasetInfo]:
        items: list[DatasetInfo] = []
        for path in self.jsonl_dir.glob("*.jsonl"):
            items.append(DatasetInfo(name=path.name, path=path, type="jsonl"))
        for path in self.csv_dir.glob("*.csv"):
            items.append(DatasetInfo(name=path.name, path=path, type="csv"))
        return sorted(items, key=lambda x: x.name)

    def load_dataset(self, info: DatasetInfo) -> list[RawMessage]:
        if info.type == "jsonl":
            return list(self._read_jsonl(info.path))
        if info.type == "csv":
            return list(self._read_csv(info.path))
        raise ValueError(f"Неизвестный тип датасета: {info.type}")

    def _read_jsonl(self, path: Path) -> Iterable[RawMessage]:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                author = str(obj.get("author") or "").strip()
                text = str(obj.get("text") or "").strip()
                if not author or not text:
                    continue
                yield RawMessage(
                    author=author,
                    author_id=str(obj.get("author_id") or author),
                    username=obj.get("username") or None,
                    text=text,
                    date=obj.get("date"),
                )

    def _read_csv(self, path: Path) -> Iterable[RawMessage]:
        with path.open("r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                author = (row.get("author") or "").strip()
                text = (row.get("text") or "").strip()
                if not author or not text:
                    continue
                yield RawMessage(
                    author=author,
                    author_id=row.get("author_id") or author,
                    username=row.get("username") or None,
                    text=text,
                    date=row.get("date"),
                )

    # ------------------------------------------------------------------ #
    # Чистка и нормализация
    # ------------------------------------------------------------------ #
    def clean_text(self, text: str) -> str:
        """Применить регулярные чистки текста (URL, упоминания, html и т.п.)."""
        text = self._emoji_re.sub("", text)
        text = self._url_re.sub("", text)
        text = self._mention_re.sub("", text)
        text = self._html_re.sub("", text)
        text = self._control_re.sub("", text)
        text = self._whitespace_re.sub(" ", text)
        return text.strip()

    def participants(self, messages: Iterable[RawMessage]) -> list[Participant]:
        """Группировка/нормализация авторов через :class:`AliasResolver`."""
        as_dicts = (
            {
                "author": m.author,
                "author_id": m.author_id,
                "username": m.username,
            }
            for m in messages
        )
        return self._alias_resolver.resolve(as_dicts)

    def extract_conversation(self, messages: Iterable[RawMessage]) -> tuple[list[CleanMessage], str]:
        """Применить фильтр мусора и почистить тексты.

        Returns:
            Кортеж ``(сообщения, summary)`` — список валидных + строка-сводка.
        """
        flt = GarbageFilter(self.settings.filters, self.settings.data_processor)
        cleaned: list[CleanMessage] = []
        for msg in messages:
            text = self.clean_text(msg.text)
            if not text:
                continue
            reason = flt.classify(msg.author_id or msg.author, text)
            if reason is not None:
                continue
            cleaned.append(
                CleanMessage(
                    from_name=msg.author,
                    from_id=str(msg.author_id or msg.author),
                    text=text,
                )
            )
        summary = flt.stats.as_summary()
        logger.info("Фильтрация датасета — {}", summary)
        return cleaned, summary

    def prepare_training_data(
        self,
        conversation: Iterable[CleanMessage],
        target_user_id: str,
    ) -> list[TrainingPair]:
        """Собрать пары «сообщение собеседника → ответ целевого пользователя»."""
        convo = list(conversation)
        pairs: list[TrainingPair] = []
        for prev, current in zip(convo, convo[1:]):
            if current.from_id == target_user_id and prev.from_id != target_user_id:
                pairs.append(TrainingPair(prompt=prev.text, response=current.text))
        logger.info("Сформировано {} обучающих пар", len(pairs))
        return pairs

    # ------------------------------------------------------------------ #
    # Конвертация TG JSON → CSV/JSONL
    # ------------------------------------------------------------------ #
    def parse_json_to_dataset(
        self,
        input_path: Path,
        output_name: str | None = None,
    ) -> DatasetArtifacts | None:
        """Сконвертировать выгрузку Telegram в CSV + JSONL."""
        if not input_path.is_file():
            logger.error("Файл не найден: {}", input_path)
            return None
        try:
            with input_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            logger.error("Ошибка JSON: {}", exc)
            return None

        raw_messages = _extract_raw_messages(data)
        flt = GarbageFilter(self.settings.filters, self.settings.data_processor)
        out_rows: list[dict[str, str]] = []
        for msg in raw_messages:
            text = self.clean_text(_coerce_text(msg.get("text", "")))
            if not text:
                continue
            author = (msg.get("from") or msg.get("author") or "Unknown").strip()
            author_id = str(msg.get("from_id") or msg.get("author_id") or author)
            username = (
                msg.get("username")
                or (msg.get("from_user") or {}).get("username")
                if isinstance(msg.get("from_user"), dict)
                else msg.get("username")
            )
            reason = flt.classify(author_id, text)
            if reason is not None:
                continue
            out_rows.append(
                {
                    "author": author,
                    "author_id": author_id,
                    "username": str(username) if username else "",
                    "text": text,
                    "date": str(msg.get("date") or ""),
                }
            )

        if not out_rows:
            logger.error("После фильтрации не осталось сообщений — нечего сохранять")
            return None

        name = output_name or input_path.stem
        csv_path = self.csv_dir / f"{name}.csv"
        jsonl_path = self.jsonl_dir / f"{name}.jsonl"

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["author", "author_id", "username", "text", "date"])
            writer.writeheader()
            writer.writerows(out_rows)

        with jsonl_path.open("w", encoding="utf-8") as fh:
            for row in out_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = flt.stats.as_summary()
        logger.info("Датасет '{}' создан: {} сообщений ({})", name, len(out_rows), summary)
        return DatasetArtifacts(
            csv_path=csv_path,
            jsonl_path=jsonl_path,
            name=name,
            message_count=len(out_rows),
            filter_summary=summary,
        )


def _coerce_text(value: object) -> str:
    """Telegram-экспорт хранит ``text`` как строку или как список фрагментов."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(str(item.get("text", "")))
        return " ".join(out).strip()
    return ""


def _extract_raw_messages(data: object) -> list[dict]:
    """Найти список сообщений в произвольной структуре TG-экспорта."""
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)]
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("messages"), list):
        return [m for m in data["messages"] if isinstance(m, dict) and m.get("type") == "message"]
    if isinstance(data.get("chats"), list):
        out: list[dict] = []
        for chat in data["chats"]:
            if isinstance(chat, dict) and isinstance(chat.get("messages"), list):
                out.extend(
                    m for m in chat["messages"] if isinstance(m, dict) and m.get("type") == "message"
                )
        return out
    if isinstance(data.get("chats"), dict):
        chats = data["chats"]
        if isinstance(chats.get("list"), list):
            out = []
            for chat in chats["list"]:
                if isinstance(chat, dict) and isinstance(chat.get("messages"), list):
                    out.extend(
                        m
                        for m in chat["messages"]
                        if isinstance(m, dict) and m.get("type") == "message"
                    )
            return out
    return []
