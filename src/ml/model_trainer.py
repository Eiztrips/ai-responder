import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
from src.utils.data_processor import DataProcessor
from typing import List, Tuple, Dict
import json
import logging

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
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
        self.data_processor = DataProcessor()
        
        if torch.backends.mps.is_available():
            self.device = "mps"
            torch.mps.set_per_process_memory_fraction(0.7)
            logger.info(f"Используется Apple MPS ускоритель на {torch.mps.current_allocated_memory()/1024/1024:.2f} МБ памяти GPU")
        else:
            self.device = "cpu"
            logger.info("MPS недоступен, используется CPU")
        
        import gc
        gc.collect()
        torch.mps.empty_cache() if self.device == "mps" else None
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def train_model(self, dataset_info: Dict, target_user_id: str) -> str:
        logger.info(f"Начинаем обучение на файле: {dataset_info['name']} для пользователя с ID {target_user_id}")
        
        data = self.data_processor.load_dataset(dataset_info)
        conversation = self.data_processor.extract_conversation(data)
        
        participants = self.data_processor.get_chat_participants(data)
        target_user_name = next((user['name'] for user in participants if user['id'] == target_user_id), "Неизвестный пользователь")
        
        training_data = self.data_processor.prepare_training_data(conversation, target_user_id)
        
        logger.info(f"Подготовлено {len(training_data)} пар для обучения пользователя {target_user_name}")
        
        if len(training_data) < 5:
            logger.error("Недостаточно данных для обучения.")
            return "Недостаточно данных для обучения. Нужно минимум 5 пар диалогов."
        
        batch_size = 1 if self.device == "mps" else 1
        
        try:
            model_name = "tinkoff-ai/ruDialoGPT-small"
            logger.info(f"Загрузка модели {model_name}...")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                        torch_dtype=torch.float32)
            
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
                num_train_epochs=3,  # Изменить на 4
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=8,
                save_steps=500,
                save_total_limit=1,
                logging_steps=50,
                fp16=False,
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
                "model_size": model_name
            }
            
            with open(os.path.join(model_save_path, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            del model, trainer, dataset
            torch.mps.empty_cache() if self.device == "mps" else None
            
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
