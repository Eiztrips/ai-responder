import os
import sys
import logging
from start.main import main
import asyncio
import gc
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

os.environ["OMP_NUM_THREADS"] = "4"

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"NVIDIA GPU обнаружена: {device_name} ({device_memory:.1f} ГБ)")
    else:
        logging.info("NVIDIA GPU не обнаружена, будет использоваться CPU")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('../../training_log.txt')
        ]
    )

    try:
        from src.ml.model_trainer import ModelTrainer
        test_instance = ModelTrainer()
        logging.info("Модуль ModelTrainer успешно загружен")
    except Exception as e:
        logging.error(f"Ошибка при инициализации ModelTrainer: {e}")
    
    logging.info("Запуск на Windows с оптимизацией для NVIDIA GPU")
    asyncio.run(main())
