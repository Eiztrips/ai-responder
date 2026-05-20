"""Нормализация никнеймов и склейка алиасов одного и того же пользователя.

Старый код полагался только на ``author`` (отображаемое имя), из-за чего
сообщения «Иван 🚀», «Иван!», «Иван' ' с zero-width» считались тремя разными
людьми. Здесь мы:

* Нормализуем имя (NFKC, удаление эмодзи, zero-width, control chars);
* Группируем по ``author_id`` (надёжный telegram id) — это первичный ключ;
* Если ``author_id`` отсутствует — fuzzy-merge нормализованных имён через
  rapidfuzz (Jaro-Winkler ≥ порога).

В итоге у каждого участника есть каноничное имя, telegram username (если был),
список всех известных алиасов и счётчик сообщений.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from rapidfuzz import fuzz

from aiResponder.config.settings import DataProcessorConfig, NicknameConfig
from aiResponder.data.filters import ZERO_WIDTH

_CONTROL_RE = re.compile(r"[\x00-\x1F\x7F-\x9F]")
_MULTISPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class Participant:
    """Каноничное представление участника чата."""

    canonical_id: str
    canonical_name: str
    username: str | None
    aliases: list[str] = field(default_factory=list)
    message_count: int = 0

    @property
    def display(self) -> str:
        parts = [self.canonical_name]
        if self.username:
            parts.append(f"@{self.username}")
        parts.append(f"id={self.canonical_id}")
        parts.append(f"{self.message_count} сообщ.")
        if self.aliases and self.aliases != [self.canonical_name]:
            preview = ", ".join(self.aliases[:3])
            if len(self.aliases) > 3:
                preview += "…"
            parts.append(f"алиасы: {preview}")
        return " · ".join(parts)


class NicknameNormalizer:
    """Чистит ник от эмодзи, zero-width и неотображаемых символов."""

    def __init__(self, data_cfg: DataProcessorConfig, strip_emoji: bool = True) -> None:
        self._emoji_re = re.compile(data_cfg.emoji_pattern, flags=re.UNICODE) if strip_emoji else None

    def normalize(self, name: str | None) -> str:
        if not name:
            return ""
        text = unicodedata.normalize("NFKC", name)
        text = "".join(ch for ch in text if ch not in ZERO_WIDTH)
        text = _CONTROL_RE.sub("", text)
        if self._emoji_re is not None:
            text = self._emoji_re.sub("", text)
        return _MULTISPACE_RE.sub(" ", text).strip()

    def display(self, name: str | None) -> str:
        normalized = self.normalize(name)
        return normalized or "Без имени"


@dataclass(slots=True)
class _Bucket:
    canonical_id: str
    canonical_name: str
    username: str | None
    aliases: set[str] = field(default_factory=set)
    counter: int = 0

    def merge(self, name: str, username: str | None, count: int) -> None:
        self.counter += count
        if name:
            self.aliases.add(name)
        if not self.username and username:
            self.username = username


class AliasResolver:
    """Сворачивает «грязные» имена с одинаковым ``author_id`` (или похожие) в одного человека.

    Args:
        nicknames_cfg: Параметры fuzzy-склейки.
        normalizer: Нормализатор имён.
    """

    def __init__(self, nicknames_cfg: NicknameConfig, normalizer: NicknameNormalizer) -> None:
        self.cfg = nicknames_cfg
        self.normalizer = normalizer

    def resolve(self, messages: Iterable[dict]) -> list[Participant]:
        """Построить список участников по списку сообщений.

        Каждое сообщение должно быть словарём с ключами ``author``, ``author_id``
        (опц.), ``username`` (опц.).
        """
        by_id: dict[str, _Bucket] = {}
        unknown_id_buckets: list[_Bucket] = []  # для случаев, когда id отсутствует

        for msg in messages:
            raw_name = msg.get("author") or ""
            name = self.normalizer.normalize(raw_name)
            username = (msg.get("username") or "").lstrip("@") or None
            raw_id = msg.get("author_id")
            author_id = str(raw_id) if raw_id not in (None, "") else None

            if author_id:
                bucket = by_id.get(author_id)
                if bucket is None:
                    bucket = _Bucket(
                        canonical_id=author_id,
                        canonical_name=name or raw_name or author_id,
                        username=username,
                    )
                    by_id[author_id] = bucket
                bucket.merge(name, username, 1)
            else:
                merged = False
                if self.cfg.fuzzy_merge and name:
                    threshold = int(self.cfg.similarity_threshold * 100)
                    for b in unknown_id_buckets:
                        score = fuzz.token_set_ratio(name, b.canonical_name)
                        if score >= threshold:
                            b.merge(name, username, 1)
                            merged = True
                            break
                if not merged:
                    fake_id = f"name::{name or raw_name}"
                    unknown_id_buckets.append(
                        _Bucket(
                            canonical_id=fake_id,
                            canonical_name=name or raw_name or "Без имени",
                            username=username,
                        )
                    )
                    unknown_id_buckets[-1].merge(name, username, 1)

        participants: list[Participant] = []
        for bucket in (*by_id.values(), *unknown_id_buckets):
            most_common = Counter(filter(None, bucket.aliases)).most_common(1)
            canonical_name = most_common[0][0] if most_common else bucket.canonical_name
            participants.append(
                Participant(
                    canonical_id=bucket.canonical_id,
                    canonical_name=canonical_name,
                    username=bucket.username,
                    aliases=sorted(bucket.aliases),
                    message_count=bucket.counter,
                )
            )

        participants.sort(key=lambda p: p.message_count, reverse=True)
        return participants

    @staticmethod
    def by_id(participants: Iterable[Participant]) -> dict[str, Participant]:
        return {p.canonical_id: p for p in participants}

    @staticmethod
    def author_to_participant(participants: Iterable[Participant]) -> dict[str, Participant]:
        """Карта «любой известный alias → участник» для пост-нормализации датасета."""
        mapping: dict[str, Participant] = defaultdict(lambda: None)  # type: ignore[arg-type]
        for p in participants:
            for alias in {*p.aliases, p.canonical_name, p.canonical_id}:
                if alias:
                    mapping[alias] = p
        return mapping
