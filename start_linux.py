import os
import logging
from main import main
import asyncio
import gc
import torch

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training_log.txt')
        ]
    )

    logging.info("Запуск на Linux с оптимизацией для NVIDIA GPU")
    asyncio.run(main())