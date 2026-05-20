"""Конфигурируемые эвристики фильтрации мусора в исходной переписке.

Цель — убрать очевидный шум (стикеры, URL-only, повторы, спам символов), но
сохранить короткие «человеческие» реплики («ок», «+», «да»), потому что бот
должен подражать живому общению, а не выбрасывать каждый односложный ответ.

Каждая проверка возвращает причину отбраковки или ``None`` (значит «пропустить
сообщение дальше»). Это даёт прозрачную статистику в :class:`FilterStats`.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable

from aiResponder.config.settings import DataProcessorConfig, FiltersConfig

ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\ufeff"}
_PURE_NON_LETTER_RE = re.compile(r"^[\W\d_]+$", flags=re.UNICODE)
_HAS_LETTER_RE = re.compile(r"[^\W\d_]", flags=re.UNICODE)


@dataclass(slots=True)
class FilterStats:
    """Сводная статистика фильтрации."""

    total: int = 0
    kept: int = 0
    dropped: int = 0
    reasons: Counter[str] = field(default_factory=Counter)

    def record(self, reason: str | None) -> None:
        self.total += 1
        if reason is None:
            self.kept += 1
        else:
            self.dropped += 1
            self.reasons[reason] += 1

    def as_summary(self) -> str:
        if self.total == 0:
            return "Сообщений: 0"
        reasons = ", ".join(f"{r}={n}" for r, n in self.reasons.most_common())
        return (
            f"Всего {self.total}, оставлено {self.kept}, отброшено {self.dropped}"
            + (f" ({reasons})" if reasons else "")
        )


class GarbageFilter:
    """Объединённый фильтр мусора, настраиваемый через :class:`FiltersConfig`.

    Использование::

        flt = GarbageFilter(cfg.filters, cfg.data_processor)
        for msg in messages:
            reason = flt.classify(msg.author_id, msg.text)
            if reason is None:
                yield msg
        print(flt.stats.as_summary())
    """

    def __init__(
        self,
        filters: FiltersConfig,
        data_cfg: DataProcessorConfig,
        whitelist: Iterable[str] | None = None,
    ) -> None:
        self.cfg = filters
        self.stats = FilterStats()

        self._emoji_re = re.compile(data_cfg.emoji_pattern, flags=re.UNICODE)
        self._url_re = re.compile(data_cfg.url_pattern, flags=re.IGNORECASE)
        self._system_re = re.compile(data_cfg.system_message_pattern, flags=re.IGNORECASE)

        items = whitelist if whitelist is not None else filters.short_reply_whitelist
        self._whitelist = {self._normalize_for_match(w) for w in items}

        # окно последних сообщений на каждого автора для проверки дубликатов
        self._dup_windows: dict[str, deque[str]] = {}

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        return unicodedata.normalize("NFKC", text).strip().casefold()

    def _strip_zero_width(self, text: str) -> str:
        return "".join(ch for ch in text if ch not in ZERO_WIDTH)

    def _is_emoji_only(self, text: str) -> bool:
        stripped = self._emoji_re.sub("", text).strip()
        return stripped == ""

    def _is_url_only(self, text: str) -> bool:
        stripped = self._url_re.sub("", text).strip()
        return stripped == ""

    def _is_system(self, text: str) -> bool:
        return bool(self._system_re.match(text.strip()))

    def _is_repeat_spam(self, text: str) -> bool:
        threshold = self.cfg.repeat_threshold
        if threshold <= 0:
            return False
        run = 1
        prev = ""
        for ch in text:
            if ch == prev and ch.isalnum():
                run += 1
                if run >= threshold:
                    return True
            else:
                run = 1
                prev = ch
        return False

    def _is_pure_punct_or_digits(self, text: str) -> bool:
        return bool(_PURE_NON_LETTER_RE.match(text.strip()))

    def _is_duplicate(self, author_id: str, text: str) -> bool:
        window_size = self.cfg.drop_duplicate_window
        if window_size <= 0:
            return False
        window = self._dup_windows.setdefault(author_id, deque(maxlen=window_size))
        normalized = self._normalize_for_match(text)
        is_dup = normalized in window
        window.append(normalized)
        return is_dup

    def _under_min_words(self, text: str) -> bool:
        return len(text.split()) < max(self.cfg.min_words, 1)

    def classify(self, author_id: str, text: str) -> str | None:
        """Вернуть код причины удаления или ``None`` если сообщение валидно."""
        if not self.cfg.enabled:
            self.stats.record(None)
            return None

        cleaned = self._strip_zero_width(text or "").strip()
        if not cleaned:
            self.stats.record("empty")
            return "empty"

        normalized = self._normalize_for_match(cleaned)
        in_whitelist = normalized in self._whitelist

        if self.cfg.drop_system_messages and self._is_system(cleaned):
            self.stats.record("system")
            return "system"

        if self.cfg.drop_emoji_only and self._is_emoji_only(cleaned):
            self.stats.record("emoji_only")
            return "emoji_only"

        if self.cfg.drop_url_only and self._is_url_only(cleaned):
            self.stats.record("url_only")
            return "url_only"

        if self.cfg.drop_repeat_spam and self._is_repeat_spam(cleaned):
            self.stats.record("repeat_spam")
            return "repeat_spam"

        if self.cfg.drop_duplicate_window and self._is_duplicate(author_id, cleaned):
            self.stats.record("duplicate")
            return "duplicate"

        # «человеческие» короткие реплики имеют право жить, если разрешены.
        if not (self.cfg.allow_short_replies and in_whitelist):
            if self.cfg.require_letters and not _HAS_LETTER_RE.search(cleaned):
                self.stats.record("no_letters")
                return "no_letters"
            if self.cfg.drop_pure_punct_or_digits and self._is_pure_punct_or_digits(cleaned):
                self.stats.record("pure_punct")
                return "pure_punct"
            if self._under_min_words(cleaned):
                self.stats.record("too_short")
                return "too_short"

        self.stats.record(None)
        return None
