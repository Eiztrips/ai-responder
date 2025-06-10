# MIT License
#
# Copyright (c) 2025 Eiztrips
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import json
import torch
import functools
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import logging
import yaml
from pathlib import Path

class ResponseGenerator:
    def __init__(self, model_path=None, config_manager=None):
        if config_manager:
            self.config = config_manager.get_inference_config()
            logging_config = config_manager.get_section('logging', {})
            # Set environment variables from config
            if 'device' in self.config and 'cuda_env_vars' in self.config['device']:
                for key, value in self.config['device']['cuda_env_vars'].items():
                    os.environ[key] = value
        else:
            # Backward compatibility
            config_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / 'config' / 'config.yaml'
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.config = config['inference']
            logging_config = config['logging']
            # Set environment variable from config
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = self.config['device']['cuda_env_vars']['PYTORCH_MPS_HIGH_WATERMARK_RATIO']

        logging.basicConfig(level=getattr(logging, logging_config.get('level', 'INFO')), 
                         format=logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger = logging.getLogger(__name__)
        
        self.model = None
        self.tokenizer = None
        self.metadata = {}
        self.profile = self.config.get('active_profile', 'default')
        
        self.logger.info(f"Используется профиль генерации: {self.profile}")

        if torch.cuda.is_available():
            self.device = "cuda"
            self.logger.info(f"Используется NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            torch.mps.set_per_process_memory_fraction(self.config['device']['mps_memory_fraction'])
            self.logger.info(f"Используется устройство MPS (Apple Silicon)")
        else:
            self.device = "cpu"
            self.logger.info("Используется CPU")

        self.models_dir = os.path.join(os.path.dirname(__file__), self.config['model']['models_dir'])
        self.response_cache = {}
        self.max_cache_size = self.config['cache']['max_size']

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
            
            self.logger.info(f"Загружаем модель из {model_path}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            dtype = torch.float16 if self.device == "cuda" else torch.float32
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            
            self.response_cache.clear()
            
            self.logger.info(f"Модель успешно загружена из {model_path}")
            
            metadata_path = os.path.join(model_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    self.logger.info(f"Модель для пользователя: {self.metadata.get('target_user', 'Неизвестно')}")
            
            return True
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            return False
    
    def load_latest_model(self):
        if not os.path.exists(self.models_dir):
            return False
            
        model_dirs = [os.path.join(self.models_dir, d) for d in os.listdir(self.models_dir) 
                     if os.path.isdir(os.path.join(self.models_dir, d))]
        
        if not model_dirs:
            self.logger.warning("Обученных моделей не найдено")
            return False
            
        latest_model = max(model_dirs, key=os.path.getmtime)
        return self.load_model(latest_model)
    
    def clean_response(self, text):
        text = re.sub(r'@@[^@]*@@', '', text)
        text = re.sub(r'(?m)^[A-Z]:\s*', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_generation_params(self):
        profiles = self.config['model'].get('generation_profiles', {})

        if self.profile in profiles:
            return profiles[self.profile]

        self.logger.warning(f"Профиль {self.profile} не найден, используем default")
        return self.config['model']['generation']
    
    @functools.lru_cache(maxsize=128)
    def _cached_generate_response(self, message_key, max_length, profile):
        if not self.model or not self.tokenizer:
            return "Модель не загружена. Пожалуйста, сначала обучите модель."
        
        input_text = f"Q: {message_key}\nA:"
        inputs = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        gen_params = self.get_generation_params()
        self.logger.debug(f"Используются параметры генерации: {gen_params}")
        
        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_length=gen_params['max_length'],
                num_return_sequences=gen_params['num_return_sequences'],
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=gen_params['do_sample'],
                temperature=gen_params['temperature'],
                top_p=gen_params['top_p'],
                no_repeat_ngram_size=gen_params['no_repeat_ngram_size'],
                repetition_penalty=gen_params['repetition_penalty'],
                length_penalty=gen_params['length_penalty'],
                early_stopping=gen_params['early_stopping']
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        try:
            answer = response.split("A:")[1].strip()
        except:
            answer = response

        answer = self.clean_response(answer)
        
        return answer
    
    def generate_response(self, message, max_length=150, profile=None):
        if profile is not None:
            original_profile = self.profile
            self.profile = profile
            self.logger.info(f"Временно переключен профиль на: {profile}")
            result = self._cached_generate_response(message, max_length, profile)
            self.profile = original_profile
            return result
        
        return self._cached_generate_response(message, max_length, self.profile)
    
    def set_profile(self, profile_name):
        profiles = self.config['model'].get('generation_profiles', {})
        
        if profile_name in profiles:
            self.profile = profile_name
            self.logger.info(f"Профиль генерации изменен на: {profile_name}")
            return True
        else:
            self.logger.warning(f"Профиль {profile_name} не найден")
            return False
    
    def get_available_profiles(self):
        return list(self.config['model'].get('generation_profiles', {}).keys())

response_generator = None

def generate_response(message, profile=None):
    global response_generator
    
    if response_generator is None:
        response_generator = ResponseGenerator()

    if profile:
        return response_generator.generate_response(message, profile=profile)
        
    return response_generator.generate_response(message)
