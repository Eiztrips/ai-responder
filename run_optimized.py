#!/usr/bin/env python3

import os
import logging
from main import main
import asyncio
import gc
import torch

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["OMP_NUM_THREADS"] = "4"

if __name__ == "__main__":
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training_log.txt')
        ]
    )
    
    logging.info("Запуск с оптимизированными параметрами памяти")
    asyncio.run(main())
