import os
import json
import re
import csv
from typing import List, Dict, Tuple, Set
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF" 
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251" 
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u200d"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\u3030"
    "\ufe0f"
    "]+", 
    flags=re.UNICODE
)

class DataProcessor:
    def __init__(self):
        self.base_model_dir = os.path.join(os.path.dirname(__file__), 'model')
        self.jsonl_dir = os.path.join(self.base_model_dir, 'json')
        self.csv_dir = os.path.join(self.base_model_dir, 'csv')
        
        os.makedirs(self.jsonl_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        
    def get_available_datasets(self, format_type=None):
        datasets = []
        
        if format_type == 'jsonl' or format_type is None:
            jsonl_files = [f for f in os.listdir(self.jsonl_dir) if f.endswith('.jsonl')]
            for file in jsonl_files:
                datasets.append({'name': file, 'type': 'jsonl', 'path': os.path.join(self.jsonl_dir, file)})
        
        if format_type == 'csv' or format_type is None:
            csv_files = [f for f in os.listdir(self.csv_dir) if f.endswith('.csv')]
            for file in csv_files:
                datasets.append({'name': file, 'type': 'csv', 'path': os.path.join(self.csv_dir, file)})
        
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
                    for row in reader:
                        if 'author' in row and 'text' in row:
                            messages.append({
                                'author': row['author'],
                                'text': row['text']
                            })
            
            logger.info(f"Загружено {len(messages)} сообщений из {filepath}")
            return messages
        except Exception as e:
            logger.error(f"Ошибка загрузки файла {filepath}: {e}")
            return []
    
    def clean_text(self, text: str) -> str:
        text = EMOJI_PATTERN.sub(r'', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'@\S+|#\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()
    
    def is_valid_message(self, text: str) -> bool:
        if len(text.split()) < 2:
            return False
            
        if re.match(r'^(joined|left|pinned|changed|removed|added|created)', text):
            return False
            
        if not re.search(r'[a-zA-Zа-яА-Я]', text):
            return False
            
        return True
    
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
        
        logger.info(f"Извлечено {len(conversation)} валидных сообщений из датасета")
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
        
        sorted_participants = sorted(
            participants.values(),
            key=lambda x: x['message_count'],
            reverse=True
        )
        
        return sorted_participants
    
    def prepare_training_data(self, conversation: List[Dict], target_user_id: str) -> List[Tuple[str, str]]:
        training_data = []
        
        for i in range(1, len(conversation)):
            current_msg = conversation[i]
            prev_msg = conversation[i-1]
            
            if current_msg['from_id'] == target_user_id and prev_msg['from_id'] != target_user_id:
                training_data.append((prev_msg['text'], current_msg['text']))
        
        logger.info(f"Создано {len(training_data)} пар для обучения")
        return training_data
    
    def get_target_user(self, messages: List[Dict]) -> str:
        participants = self.get_chat_participants(messages)
        
        if participants:
            return participants[0]['id']
        
        return ""
    
    def parse_json_to_dataset(self, input_file_path: str, output_file_name: str = None) -> dict:
        try:
            if not os.path.isfile(input_file_path):
                logger.error(f"Файл {input_file_path} не существует")
                return None
            
            with open(input_file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    logger.info(f"JSON файл успешно загружен: {input_file_path}")
                except json.JSONDecodeError as e:
                    logger.error(f"Ошибка при декодировании JSON: {e}")
                    return None
                    
            if not output_file_name:
                base_name = os.path.basename(input_file_path)
                name_without_ext = os.path.splitext(base_name)[0]
                output_file_name = f"{name_without_ext}"
            
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
                logger.error("Не удалось извлечь сообщения из JSON файла")
                return None
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["author", "text"])
                writer.writeheader()
                writer.writerows(messages)
            
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for msg in messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + '\n')
            
            logger.info(f"Данные успешно сохранены в CSV: {csv_path} и JSONL: {jsonl_path}")
            logger.info(f"Обработано {len(messages)} сообщений")
            
            return {
                'csv': {'name': f"{output_file_name}.csv", 'type': 'csv', 'path': csv_path},
                'jsonl': {'name': f"{output_file_name}.jsonl", 'type': 'jsonl', 'path': jsonl_path}
            }
            
        except Exception as e:
            logger.error(f"Ошибка при создании датасета: {e}")
            return None
