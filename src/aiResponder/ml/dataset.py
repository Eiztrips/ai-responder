"""Torch-датасет для языкового моделирования пары «реплика → ответ».

Формат строки идентичен legacy-варианту (``Q: ...\\nA: ...``), что даёт прямую
совместимость со старыми обученными моделями. Маскируем токены до ``A:`` в
``labels`` — модель учится только генерации ответа, а не повторению вопроса.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from aiResponder.data.processor import TrainingPair

IGNORE_INDEX = -100


class ConversationDataset(Dataset):
    """Преобразует ``list[TrainingPair]`` в torch-тензоры для HF Trainer."""

    def __init__(
        self,
        pairs: Sequence[TrainingPair],
        tokenizer,
        max_length: int = 512,
        prompt_format: str = "Q: {question}\nA: {answer}",
    ) -> None:
        self._pairs = list(pairs)
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._prompt_format = prompt_format

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self._pairs[idx]
        prompt_part = f"Q: {pair.prompt}\nA:"
        full = self._prompt_format.format(question=pair.prompt, answer=pair.response)

        prompt_ids = self._tokenizer(
            prompt_part, truncation=True, max_length=self._max_length, add_special_tokens=False
        )["input_ids"]
        full_enc = self._tokenizer(
            full,
            truncation=True,
            max_length=self._max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # маскируем prompt-часть, чтобы learning rate шёл только на ответ
        prompt_len = min(len(prompt_ids), labels.shape[0])
        labels[:prompt_len] = IGNORE_INDEX
        labels[attention_mask == 0] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
