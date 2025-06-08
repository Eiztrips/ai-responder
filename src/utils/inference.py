import os
import json
import torch
import functools
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import logging

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self, model_path=None):
        self.model = None
        self.tokenizer = None
        
        # Определяем лучшее доступное устройство
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"Используется NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            torch.mps.set_per_process_memory_fraction(0.7)
            logger.info(f"Используется устройство MPS (Apple Silicon)")
        else:
            self.device = "cpu"
            logger.info("Используется CPU")
        
        self.models_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
        
        self.response_cache = {}
        self.max_cache_size = 100
        
        if model_path:
            self.load_model(model_path)
        else:
            self.load_latest_model()
    
    def load_model(self, model_path):
        try:
            if self.model:
                del self.model
                del self.tokenizer
                gc.collect()
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
            
            logger.info(f"Загружаем модель из {model_path}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Выбор dtype в зависимости от устройства
            dtype = torch.float16 if self.device == "cuda" else torch.float32
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            
            self.response_cache.clear()
            
            logger.info(f"Модель успешно загружена из {model_path}")
            
            metadata_path = os.path.join(model_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    logger.info(f"Модель для пользователя: {self.metadata.get('target_user', 'Неизвестно')}")
            
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False
    
    def load_latest_model(self):
        if not os.path.exists(self.models_dir):
            logger.warning("Директория с моделями не найдена")
            return False
            
        model_dirs = [os.path.join(self.models_dir, d) for d in os.listdir(self.models_dir) 
                     if os.path.isdir(os.path.join(self.models_dir, d))]
        
        if not model_dirs:
            logger.warning("Обученных моделей не найдено")
            return False
            
        latest_model = max(model_dirs, key=os.path.getmtime)
        return self.load_model(latest_model)
    
    def clean_response(self, text):
        text = re.sub(r'@@[^@]*@@', '', text)
        text = re.sub(r'(?m)^[A-Z]:\s*', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @functools.lru_cache(maxsize=128)
    def _cached_generate_response(self, message_key, max_length):
        if not self.model or not self.tokenizer:
            return "Модель не загружена. Пожалуйста, сначала обучите модель."
        
        input_text = f"Q: {message_key}\nA:"
        
        inputs = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_length=50,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.8,
                top_p=0.5,
                no_repeat_ngram_size=4,
                repetition_penalty=1.4,
                length_penalty=0.6,
                early_stopping=True
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        try:
            answer = response.split("A:")[1].strip()
        except:
            answer = response

        answer = self.clean_response(answer)
        
        return answer
    
    def generate_response(self, message, max_length=150):
        return self._cached_generate_response(message, max_length)

response_generator = None

def generate_response(message):
    global response_generator
    
    if response_generator is None:
        response_generator = ResponseGenerator()
        
    return response_generator.generate_response(message)
