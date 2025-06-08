import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
from src.utils.data_processor import DataProcessor
from typing import List, Tuple, Dict
import json
import logging
import platform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

class ConversationDataset(Dataset):
    def __init__(self, conversations: List[Tuple[str, str]], tokenizer, max_length=512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        question, answer = self.conversations[idx]
        
        text = f"Q: {question}\nA: {answer}"
        
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encodings.input_ids.squeeze()
        attention_mask = encodings.attention_mask.squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }

class ModelTrainer:
    def __init__(self, device=None):
        self.model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
        self.data_processor = DataProcessor()
        
        # Определяем доступные устройства
        self.available_devices = self._get_available_devices()
        
        # Используем устройство из параметра или автоматический выбор
        if device and device in self.available_devices:
            self.device = device
        else:
            self.device = self._get_default_device()
        
        self._setup_device()
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def _get_available_devices(self) -> Dict[str, bool]:
        """Определяет доступные устройства для обучения"""
        devices = {"cpu": True}  # CPU всегда доступен
        
        # Проверяем доступность CUDA (NVIDIA GPU)
        devices["cuda"] = torch.cuda.is_available()
        
        # Проверяем доступность MPS (Apple Silicon)
        devices["mps"] = torch.backends.mps.is_available()
        
        return devices
    
    def _get_default_device(self) -> str:
        """Выбирает оптимальное устройство по умолчанию"""
        if self.available_devices.get("cuda", False):
            return "cuda"
        elif self.available_devices.get("mps", False):
            return "mps"
        else:
            return "cpu"
    
    def _setup_device(self):
        """Настраивает выбранное устройство"""
        if self.device == "mps":
            torch.mps.set_per_process_memory_fraction(0.7)
            logger.info(f"Используется Apple MPS ускоритель на {torch.mps.current_allocated_memory()/1024/1024:.2f} МБ памяти GPU")
        elif self.device == "cuda":
            torch.cuda.empty_cache()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Используется NVIDIA GPU: {gpu_name} с {gpu_memory:.1f} ГБ памяти")
        else:
            logger.info("Используется CPU для обучения")
        
        import gc
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
    
    def get_available_devices(self) -> Dict[str, str]:
        """Возвращает словарь доступных устройств с их описанием"""
        device_names = {
            "cpu": "Процессор (CPU)",
            "cuda": "NVIDIA GPU (CUDA)",
            "mps": "Apple Silicon GPU (MPS)"
        }
        
        available = {}
        for device, is_available in self.available_devices.items():
            if is_available:
                available[device] = device_names.get(device, device)
        
        return available
    
    def set_device(self, device: str) -> bool:
        """Устанавливает устройство для обучения"""
        if device in self.available_devices and self.available_devices[device]:
            self.device = device
            self._setup_device()
            return True
        return False
    
    def get_current_device(self) -> str:
        """Возвращает текущее устройство обучения"""
        return self.device
    
    def train_model(self, dataset_info: Dict, target_user_id: str) -> str:
        logger.info(f"Начинаем обучение на файле: {dataset_info['name']} для пользователя с ID {target_user_id}")
        logger.info(f"Используется устройство: {self.device}")
        
        data = self.data_processor.load_dataset(dataset_info)
        conversation = self.data_processor.extract_conversation(data)
        
        participants = self.data_processor.get_chat_participants(data)
        target_user_name = next((user['name'] for user in participants if user['id'] == target_user_id), "Неизвестный пользователь")
        
        training_data = self.data_processor.prepare_training_data(conversation, target_user_id)
        
        logger.info(f"Подготовлено {len(training_data)} пар для обучения пользователя {target_user_name}")
        
        if len(training_data) < 5:
            logger.error("Недостаточно данных для обучения.")
            return "Недостаточно данных для обучения. Нужно минимум 5 пар диалогов."
        
        # Настраиваем размер батча в зависимости от устройства
        if self.device == "cuda":
            batch_size = 2  # Для NVIDIA GPU
        elif self.device == "mps":
            batch_size = 1  # Для Apple Silicon
        else:
            batch_size = 1  # Для CPU
        
        try:
            model_name = "tinkoff-ai/ruDialoGPT-small"
            logger.info(f"Загрузка модели {model_name}...")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Установка dtype в зависимости от устройства
            if self.device == "cuda":
                dtype = torch.float16  # half precision для NVIDIA GPU
            else:
                dtype = torch.float32  # full precision для CPU и MPS
            
            model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                        torch_dtype=dtype)
            
            if hasattr(model, "config"):
                model.config.use_cache = False
            
            model.to(self.device)
            
            dataset = ConversationDataset(training_data, tokenizer, max_length=256)
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            model_save_path = os.path.join(self.model_dir, f"{target_user_name.replace(' ', '_')}_model")

            training_args = TrainingArguments(
                output_dir=model_save_path,
                overwrite_output_dir=True,
                num_train_epochs=3,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=8,
                save_steps=500,
                save_total_limit=1,
                logging_steps=50,
                fp16=self.device == "cuda",  # только для NVIDIA GPU
                optim="adamw_torch",
                learning_rate=5e-5,
                warmup_steps=100,
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                report_to="none",
                dataloader_num_workers=0,
                gradient_checkpointing=True,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset,
            )
            
            logger.info("Начинаем обучение модели...")
            trainer.train()
            
            trainer.save_model()
            tokenizer.save_pretrained(model_save_path)
            
            metadata = {
                "target_user": target_user_name,
                "target_user_id": target_user_id,
                "source_file": dataset_info['name'],
                "source_file_type": dataset_info['type'],
                "training_pairs_count": len(training_data),
                "model_size": model_name,
                "training_device": self.device
            }
            
            with open(os.path.join(model_save_path, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            del model, trainer, dataset
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info(f"Модель обучена и сохранена в {model_save_path}")
            return model_save_path
        
        except Exception as e:
            logger.exception(f"Ошибка при обучении модели: {e}")
            return f"Произошла ошибка при обучении модели: {str(e)}"
    
    def list_trained_models(self) -> List[Dict]:
        models = []
        
        if not os.path.exists(self.model_dir):
            return models
        
        for model_dir in os.listdir(self.model_dir):
            metadata_path = os.path.join(self.model_dir, model_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    models.append({
                        "name": model_dir,
                        "metadata": metadata
                    })
        
        return models
