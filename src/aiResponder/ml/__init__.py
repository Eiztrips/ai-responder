"""Обучение языковой модели и автодетект устройства."""

from __future__ import annotations

from aiResponder.ml.device import DeviceProfile, detect_device
from aiResponder.ml.trainer import ModelTrainer

__all__ = ["DeviceProfile", "ModelTrainer", "detect_device"]
