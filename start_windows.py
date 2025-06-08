import os
import sys
import logging
from main import main
import asyncio
import gc
import torch

# Добавляем корневую директорию проекта в путь поиска модулей
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

os.environ["OMP_NUM_THREADS"] = "4"

# Установить CUDA для использования на Windows, если доступна
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Использовать первую видеокарту
    torch.backends.cudnn.benchmark = True     # Оптимизация скорости для CUDA

if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"NVIDIA GPU обнаружена: {device_name} ({device_memory:.1f} ГБ)")
    else:
        print("NVIDIA GPU не обнаружена, будет использоваться CPU")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training_log.txt')
        ]
    )

    # Проверка наличия модуля model_trainer и его версии
    try:
        from src.ml.model_trainer import ModelTrainer
        test_instance = ModelTrainer()
        print("Модуль ModelTrainer успешно загружен")
    except Exception as e:
        print(f"Ошибка при инициализации ModelTrainer: {e}")
    
    logging.info("Запуск на Windows с оптимизацией для NVIDIA GPU")
    asyncio.run(main())
