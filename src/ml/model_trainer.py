import os
import gc
import json
import logging
import torch
import yaml
from typing import List, Tuple, Dict
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from src.utils.data_processor import DataProcessor
from pathlib import Path

class ModelTrainer:
    def __init__(self, config_manager=None, device=None, model=None):
        if config_manager:
            self.config = config_manager.get_full_config()
            ml_config = config_manager.get_ml_config()
            logging_config = config_manager.get_section('logging', {})
            logging.basicConfig(level=getattr(logging, logging_config.get('level', 'INFO')),
                          format=logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        else:
            # Backward compatibility
            config_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / 'config' / 'config.yaml'
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            ml_config = self.config.get('ml', {})
            logging_config = self.config.get('logging', {})
            logging.basicConfig(level=getattr(logging, logging_config.get('level', 'INFO')), 
                             format=logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        self.logger = logging.getLogger(__name__)
        self.model_dir = os.path.join(
            os.path.dirname(__file__), 
            self.config.get('model_dir', 'trained_models')
        )
        
        self.data_processor = DataProcessor(config_manager)
        self.model_name = model or ml_config.get('model') or self.config.get('default_model')
        self.available_devices = self._get_available_devices()
        self.device = device if device and device in self.available_devices else self._get_default_device()
        
        # Set environment variables from config
        if 'device' in self.config.get('inference', {}) and 'cuda_env_vars' in self.config['inference']['device']:
            for key, value in self.config['inference']['device']['cuda_env_vars'].items():
                os.environ[key] = value
        
        self._setup_device()

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _get_available_devices(self) -> Dict[str, bool]:
        return {
            "cpu": True,
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available()
        }

    def _get_default_device(self) -> str:
        device_priority = self.config.get('device_priority', ["cuda", "mps", "cpu"])
        for device in device_priority:
            if self.available_devices.get(device, False):
                return device
        return "cpu"

    def _setup_device(self):
        gc.collect()
        if self.device == "mps":
            torch.mps.set_per_process_memory_fraction(
                self.config.get('mps_memory_fraction', 0.7)
            )
            torch.mps.empty_cache()
            self.logger.info(
                f"Используется Apple MPS ускоритель на {torch.mps.current_allocated_memory()/1024/1024:.2f} МБ памяти GPU"
            )
        elif self.device == "cuda":
            torch.cuda.empty_cache()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(
                f"Используется NVIDIA GPU: {gpu_name} с {gpu_memory:.1f} ГБ памяти"
            )
        else:
            self.logger.info("Используется CPU для обучения")

    def _clear_memory(self):
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    def get_available_devices(self) -> Dict[str, str]:
        device_names = self.config.get('device_names', {
            "cpu": "Процессор (CPU)",
            "cuda": "NVIDIA GPU (CUDA)",
            "mps": "Apple Silicon GPU (MPS)"
        })
        return {
            device: device_names.get(device, device)
            for device, is_available in self.available_devices.items()
            if is_available
        }

    def set_device(self, device: str) -> bool:
        if device in self.available_devices and self.available_devices[device]:
            self.device = device
            self._setup_device()
            return True
        return False

    def get_current_device(self) -> str:
        return self.device

    def train_model(self, dataset_info: Dict, target_user_id: str) -> str:
        self.logger.info(
            f"Начинаем обучение на файле: {dataset_info['name']} для пользователя с ID {target_user_id}"
        )
        self.logger.info(f"Используется устройство: {self.device}")

        data = self.data_processor.load_dataset(dataset_info)
        conversation = self.data_processor.extract_conversation(data)
        participants = self.data_processor.get_chat_participants(data)
        target_user_name = next(
            (user['name'] for user in participants if user['id'] == target_user_id),
            "Неизвестный пользователь"
        )
        training_data = self.data_processor.prepare_training_data(conversation, target_user_id)
        self.logger.info(
            f"Подготовлено {len(training_data)} пар для обучения пользователя {target_user_name}"
        )

        min_pairs = self.config.get('min_training_pairs', 5)
        if len(training_data) < min_pairs:
            self.logger.error(
                f"Недостаточно данных для обучения. Нужно минимум {min_pairs} пар диалогов."
            )
            return f"Недостаточно данных для обучения. Нужно минимум {min_pairs} пар диалогов."

        try:
            self.logger.info(f"Загрузка модели {self.model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model_config = self.config.get('model_loading', {})
            dtype_str = model_config.get('dtype', 'torch.float32')
            dtype = eval(dtype_str) if isinstance(dtype_str, str) and dtype_str.startswith('torch.') else torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                **{k: v for k, v in model_config.items() if k != 'dtype'}
            )
            if hasattr(model, "config"):
                model.config.use_cache = model_config.get('use_cache', False)
            model.to(self.device)

            dataset = ConversationDataset(
                training_data, 
                tokenizer, 
                self.config.get('dataset', {'max_length': 256})
            )
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )

            model_name_format = self.config.get('model_name_format', "{user}_{model}")
            model_save_name = model_name_format.format(
                user=target_user_name.replace(' ', '_'),
                model=self.model_name.split('/')[-1]
            )
            model_save_path = os.path.join(self.model_dir, model_save_name)

            if self.device not in self.config.get('training_args', {}):
                self.logger.warning(
                    f"Нет конфигурации для устройства {self.device}, используются стандартные настройки"
                )
            device_args = self.config.get('training_args', {}).get(self.device, {}).copy()
            batch_sizes = self.config.get('batch_sizes', {'cuda': 2, 'mps': 1, 'cpu': 1})
            batch_size = batch_sizes.get(self.device, 1)
            device_args['output_dir'] = model_save_path
            device_args['per_device_train_batch_size'] = batch_size

            for k, v in device_args.items():
                if isinstance(v, str):
                    if v.startswith('os.'):
                        device_args[k] = eval(v)
                    elif v.startswith('torch.'):
                        device_args[k] = eval(v)

            training_args = TrainingArguments(**device_args)

            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset,
            )
            self.logger.info("Начинаем обучение модели...")
            trainer.train()
            trainer.save_model()
            tokenizer.save_pretrained(model_save_path)

            metadata = {
                "target_user": target_user_name,
                "target_user_id": target_user_id,
                "source_file": dataset_info['name'],
                "source_file_type": dataset_info['type'],
                "training_pairs_count": len(training_data),
                "model_base": self.model_name,
                "training_device": self.device,
                "training_timestamp": self._get_timestamp()
            }
            with open(os.path.join(model_save_path, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            del model, trainer, dataset
            self._clear_memory()
            self.logger.info(f"Модель обучена и сохранена в {model_save_path}")
            return model_save_path

        except Exception as e:
            self.logger.exception(f"Ошибка при обучении модели: {e}")
            return f"Произошла ошибка при обучении модели: {str(e)}"

    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

class ConversationDataset(Dataset):
    def __init__(self, conversations: List[Tuple[str, str]], tokenizer, config: Dict):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = config.get('max_length', 512)
        self.prompt_format = config.get('prompt_format', "Q: {}\nA: {}")

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        question, answer = self.conversations[idx]
        text = self.prompt_format.format(question, answer)
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

