"""Автодетект устройства и оптимальных параметров вычислений.

Возвращаемый :class:`DeviceProfile` затем используется как тренером, так и
инференс-генератором — никаких ``torch.cuda.is_available`` в трёх местах.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Literal

import torch
from loguru import logger

DeviceName = Literal["cuda", "mps", "cpu"]


@dataclass(slots=True, frozen=True)
class DeviceProfile:
    """Снимок выбранного устройства и связанных оптимизаций.

    Attributes:
        device: Имя устройства (``cuda`` / ``mps`` / ``cpu``).
        dtype: Рекомендованный ``torch.dtype`` для весов модели.
        supports_bf16: Поддерживает ли устройство bfloat16.
        supports_fp16: Поддерживает ли устройство float16 без NaN.
        supports_compile: Стоит ли применять ``torch.compile``.
        attn_implementation: Хинт для ``AutoModel.from_pretrained(attn_implementation=...)``.
        recommended_workers: Кол-во воркеров для DataLoader.
        gradient_checkpointing: По умолчанию включать ли gradient checkpointing.
    """

    device: DeviceName
    dtype: torch.dtype
    supports_bf16: bool
    supports_fp16: bool
    supports_compile: bool
    attn_implementation: str
    recommended_workers: int
    gradient_checkpointing: bool

    @property
    def is_gpu(self) -> bool:
        return self.device in ("cuda", "mps")

    def memory_summary(self) -> str:
        """Человекочитаемая строка о доступной памяти."""
        if self.device == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return f"{torch.cuda.get_device_name(0)} ({props.total_memory / 1024**3:.1f} ГБ)"
        if self.device == "mps":
            allocated = torch.mps.current_allocated_memory() / 1024**2
            return f"Apple Silicon MPS (выделено {allocated:.0f} МБ)"
        return f"CPU x{os.cpu_count() or 1}"


def available_devices() -> dict[DeviceName, bool]:
    """Карта доступности устройств."""
    return {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available() and torch.backends.mps.is_built(),
    }


def _cuda_compute_capability() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability(0)


def _detect_cuda() -> DeviceProfile:
    cc_major, _ = _cuda_compute_capability()
    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    fp16_ok = True
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    return DeviceProfile(
        device="cuda",
        dtype=dtype,
        supports_bf16=bf16_ok,
        supports_fp16=fp16_ok,
        supports_compile=cc_major >= 7,
        attn_implementation="sdpa",
        recommended_workers=min(os.cpu_count() or 1, 8),
        gradient_checkpointing=False,
    )


def _detect_mps() -> DeviceProfile:
    return DeviceProfile(
        device="mps",
        dtype=torch.float32,  # fp16/bf16 на MPS до сих пор нестабильны
        supports_bf16=False,
        supports_fp16=False,
        supports_compile=False,  # torch.compile + MPS = боль
        attn_implementation="eager",
        recommended_workers=0,  # обязателен 0 на MPS
        gradient_checkpointing=True,
    )


def _detect_cpu() -> DeviceProfile:
    bf16_ok = bool(getattr(torch.cpu, "is_bf16_supported", lambda: False)())
    return DeviceProfile(
        device="cpu",
        dtype=torch.bfloat16 if bf16_ok else torch.float32,
        supports_bf16=bf16_ok,
        supports_fp16=False,
        supports_compile=False,
        attn_implementation="sdpa",
        recommended_workers=min(os.cpu_count() or 1, 8),
        gradient_checkpointing=True,
    )


def detect_device(
    requested: str | None = None,
    priority: list[str] | None = None,
) -> DeviceProfile:
    """Подобрать оптимальное устройство и его профиль.

    Args:
        requested: Конкретное устройство (``cuda``/``mps``/``cpu``/``auto``).
            При ``None`` или ``auto`` срабатывает приоритетный поиск.
        priority: Кастомный приоритет (по умолчанию ``[cuda, mps, cpu]``).
    """
    priority = priority or ["cuda", "mps", "cpu"]
    devices = available_devices()

    candidate: DeviceName | None
    if requested and requested.lower() not in ("auto", ""):
        candidate = requested.lower()  # type: ignore[assignment]
        if candidate not in devices or not devices[candidate]:
            logger.warning(
                "Устройство {} запрошено, но недоступно — переключаюсь на авто",
                candidate,
            )
            candidate = None
    else:
        candidate = None

    if candidate is None:
        for name in priority:
            if devices.get(name, False):
                candidate = name  # type: ignore[assignment]
                break
    if candidate is None:
        candidate = "cpu"

    if candidate == "cuda":
        return _detect_cuda()
    if candidate == "mps":
        return _detect_mps()
    return _detect_cpu()


def prepare_runtime(profile: DeviceProfile) -> None:
    """Установить переменные окружения и очистить кеши под выбранное устройство."""
    gc.collect()
    if profile.device == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    elif profile.device == "mps":
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    os.environ.setdefault("OMP_NUM_THREADS", str(profile.recommended_workers or 1))
