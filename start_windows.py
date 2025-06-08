import os
import logging
from main import main
import asyncio
import gc
import torch

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

    logging.info("Запуск на Windows с оптимизацией для NVIDIA GPU")
    asyncio.run(main())
