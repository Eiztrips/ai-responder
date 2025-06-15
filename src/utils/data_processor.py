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

import logging
import json
import yaml
import csv
import re
import os

from typing import List, Dict, Tuple
from collections import Counter
from pathlib import Path


class DataProcessor:
    def __init__(self, config_manager=None):
        if config_manager:
            self.config = config_manager.get_data_processor_config()
            logging_config = config_manager.get_section('logging', {})
            logging.basicConfig(level=getattr(logging, logging_config.get('level', 'INFO')), 
                             format=logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        else:
            config_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / 'config' / 'config.yaml'
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.config = config['data_processor']
            logging_config = config['logging']
            logging.basicConfig(level=getattr(logging, logging_config['level']), format=logging_config['format'])

        self.logger = logging.getLogger(__name__)
        base_dir = os.path.dirname(__file__)

        directories_config = self.config.get('directories', {})
        self.base_model_dir = os.path.join(base_dir, directories_config.get('base_model_dir', 'model'))
        self.jsonl_dir = os.path.join(self.base_model_dir, directories_config.get('jsonl_dir', 'json'))
        self.csv_dir = os.path.join(self.base_model_dir, directories_config.get('csv_dir', 'csv'))
        
        os.makedirs(self.jsonl_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

        regex_patterns = self.config.get('regex', {})
        emoji_pattern = regex_patterns.get('emoji_pattern', "[\U0001F600-\U0001F64F]")
        self.EMOJI_PATTERN = re.compile(emoji_pattern, flags=re.UNICODE)
    
    def get_available_datasets(self, format_type=None):
        datasets = []
        
        if format_type == 'jsonl' or format_type is None:
            jsonl_files = [f for f in os.listdir(self.jsonl_dir) if f.endswith('.jsonl')]
            datasets.extend([{'name': file, 'type': 'jsonl', 'path': os.path.join(self.jsonl_dir, file)} for file in jsonl_files])
        
        if format_type == 'csv' or format_type is None:
            csv_files = [f for f in os.listdir(self.csv_dir) if f.endswith('.csv')]
            datasets.extend([{'name': file, 'type': 'csv', 'path': os.path.join(self.csv_dir, file)} for file in csv_files])
        
        return datasets
    
    def load_dataset(self, file_info):
        filepath = file_info['path']
        file_type = file_info['type']
        messages = []
        
        try:
            if file_type == 'jsonl':
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            message = json.loads(line.strip())
                            if isinstance(message, dict) and 'author' in message and 'text' in message:
                                messages.append(message)
                        except json.JSONDecodeError:
                            continue
            
            elif file_type == 'csv':
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    messages.extend([row for row in reader if 'author' in row and 'text' in row])
            
            self.logger.info(f"Загружено {len(messages)} сообщений из {filepath}")
            return messages
        except Exception as e:
            self.logger.error(f"Ошибка загрузки файла {filepath}: {e}")
            return []
    
    def is_valid_message(self, text: str) -> bool:
        message_validation = self.config.get('message_validation', {'min_words': 2})
        regex_patterns = self.config.get('regex', {})
        
        if len(text.split()) < message_validation.get('min_words', 2):
            return False
            
        if re.match(regex_patterns.get('system_message_pattern', 
                                      "^(joined|left|pinned|changed|removed|added|created)"), text):
            return False
            
        if not re.search(r'[a-zA-Zа-яА-Я]', text):
            return False
            
        return True
    
    def clean_text(self, text: str) -> str:
        regex_patterns = self.config.get('regex', {})
        text = self.EMOJI_PATTERN.sub(r'', text)
        text = re.sub(regex_patterns.get('url_pattern', "https?://\\S+|www\\.\\S+"), '', text)
        text = re.sub(regex_patterns.get('mention_hashtag_pattern', "@\\S+|#\\S+"), '', text)
        text = re.sub(regex_patterns.get('whitespace_pattern', "\\s+"), ' ', text)
        text = re.sub(regex_patterns.get('control_chars_pattern', "[\\x00-\\x1F\\x7F-\\x9F]"), '', text)
        text = re.sub(regex_patterns.get('html_tags_pattern', "<[^>]+>"), '', text)
        return text.strip()
    
    def extract_conversation(self, messages: List[Dict]) -> List[Dict]:
        conversation = []
        
        for msg in messages:
            if not msg.get('author') or not msg.get('text'):
                continue
                
            text = self.clean_text(msg['text'])
            
            if text and self.is_valid_message(text):
                conversation.append({
                    'from': msg['author'],
                    'text': text,
                    'from_id': msg.get('author_id', msg['author'])
                })
        
        self.logger.info(f"Извлечено {len(conversation)} валидных сообщений из датасета")
        return conversation
    
    def get_chat_participants(self, messages: List[Dict]) -> List[Dict]:
        participants = {}
        author_counts = Counter([msg.get('author') for msg in messages if msg.get('author')])
        
        for author, count in author_counts.items():
            author_id = next((msg.get('author_id') for msg in messages if msg.get('author') == author and msg.get('author_id')), author)
            participants[author] = {
                'name': author,
                'id': author_id,
                'message_count': count
            }
        
        return sorted(participants.values(), key=lambda x: x['message_count'], reverse=True)
    
    def prepare_training_data(self, conversation: List[Dict], target_user_id: str) -> List[Tuple[str, str]]:
        training_data = []
        
        for i in range(1, len(conversation)):
            current_msg = conversation[i]
            prev_msg = conversation[i-1]
            
            if current_msg['from_id'] == target_user_id and prev_msg['from_id'] != target_user_id:
                training_data.append((prev_msg['text'], current_msg['text']))
        
        self.logger.info(f"Создано {len(training_data)} пар для обучения")
        return training_data
    
    def get_target_user(self, messages: List[Dict]) -> str:
        participants = self.get_chat_participants(messages)
        return participants[0]['id'] if participants else ""
    
    def parse_json_to_dataset(self, input_file_path: str, output_file_name: str = None) -> dict:
        try:
            if not os.path.isfile(input_file_path):
                self.logger.error(f"Файл {input_file_path} не существует")
                return None
            
            with open(input_file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    self.logger.info(f"JSON файл успешно загружен: {input_file_path}")
                except json.JSONDecodeError as e:
                    self.logger.error(f"Ошибка при декодировании JSON: {e}")
                    return None
            
            if not output_file_name:
                output_file_name = os.path.splitext(os.path.basename(input_file_path))[0]
            
            csv_path = os.path.join(self.csv_dir, f"{output_file_name}.csv")
            jsonl_path = os.path.join(self.jsonl_dir, f"{output_file_name}.jsonl")
            
            messages = []
            raw_messages = []

            if isinstance(data, list):
                raw_messages = data
            elif isinstance(data, dict):
                if "messages" in data:
                    raw_messages = data["messages"]
                elif "chats" in data and isinstance(data["chats"], list):
                    for chat in data["chats"]:
                        if isinstance(chat, dict) and "messages" in chat:
                            raw_messages.extend(chat["messages"])

            for msg in raw_messages:
                if msg.get("type") != "message":
                    continue
                
                text = msg.get("text", "")
                if not isinstance(text, str) or text.strip() == "":
                    continue
                
                if msg.get("media_type"):
                    continue
                
                messages.append({
                    "author": msg.get("from", msg.get("author", "Unknown")),
                    "text": text.strip()
                })
            
            if not messages:
                self.logger.error("Не удалось извлечь сообщения из JSON файла")
                return None

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["author", "text"])
                writer.writeheader()
                writer.writerows(messages)

            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for msg in messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + '\n')
            
            self.logger.info(f"Данные успешно сохранены в CSV: {csv_path} и JSONL: {jsonl_path}")
            self.logger.info(f"Обработано {len(messages)} сообщений")
            
            return {
                'csv': {'name': f"{output_file_name}.csv", 'type': 'csv', 'path': csv_path},
                'jsonl': {'name': f"{output_file_name}.jsonl", 'type': 'jsonl', 'path': jsonl_path}
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при создании датасета: {e}")
            return None
