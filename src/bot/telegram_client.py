import os
import logging
import asyncio
from pyrogram import Client
from pyrogram.handlers import MessageHandler
from pyrogram.enums import ChatAction, ChatType

from src.utils.inference import ResponseGenerator
from decouple import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelegramResponder:
    def __init__(self, model_path=None):
        self.api_id = config('API_ID')
        self.api_hash = config('API_HASH')
        self.phone = config('PHONE')
        self.login = config('LOGIN')

        target_user_ids_raw = config('TARGET_USER_IDS', default='-1')
        target_channel_ids_raw = config('TARGET_CHANNEL_IDS', default='-1')

        self.target_user_ids = [int(uid.strip()) for uid in target_user_ids_raw.split('#')[0].split(',') if uid.strip()]
        self.target_channel_ids = [int(cid.strip()) for cid in target_channel_ids_raw.split('#')[0].split(',') if cid.strip()]
        
        self.mode = config('MODE')
        
        self.session_path = os.path.join(os.path.dirname(__file__), 'session', self.login)
        os.makedirs(os.path.dirname(self.session_path), exist_ok=True)
        
        self.client = Client(
            self.session_path,
            api_id=self.api_id,
            api_hash=self.api_hash,
            phone_number=self.phone
        )
        
        self.response_generator = ResponseGenerator(model_path)
        self.model_info = self._get_model_info()
        
        self.is_running = False
        
        logger.info(f"TelegramResponder инициализирован с моделью: {self.model_info}")
        logger.info(f"Режим работы: {self.mode}")
        logger.info(f"Целевые пользователи: {self.target_user_ids}")
        logger.info(f"Целевые каналы/группы: {self.target_channel_ids}")

    def _get_model_info(self):
        if hasattr(self.response_generator, 'metadata'):
            return f"{self.response_generator.metadata.get('target_user', 'Неизвестно')}"
        return "Последняя доступная модель"

    async def _send_response(self, client, message):
        logger.info(f"Получено сообщение: {message.text}")
        
        response = self.response_generator.generate_response(message.text)
        delay = min(len(response) * 0.1, 3)
        
        await client.send_chat_action(message.chat.id, ChatAction.TYPING)
        await asyncio.sleep(delay)
        await message.reply(response)
        
        logger.info(f"Отправлен ответ: {response}")

    async def message_handler(self, client, message):
        if message.outgoing:
            return

        if not message.text:
            return
            
        if self.mode == "only_private_chats":
            if message.chat.type == ChatType.PRIVATE:
                if -1 in self.target_user_ids or message.from_user.id in self.target_user_ids:
                    await self._send_response(client, message)
        
        elif self.mode == "only_channel_messages":
            if message.chat.type in [ChatType.GROUP, ChatType.SUPERGROUP, ChatType.CHANNEL]:
                if -1 in self.target_channel_ids or message.chat.id in self.target_channel_ids:
                    await self._send_response(client, message)
        
        elif self.mode == "stalker":
            if message.from_user and message.from_user.id in self.target_user_ids:
                await self._send_response(client, message)

    async def start(self):
        if not self.response_generator.model:
            logger.error("Нет загруженной модели. Не могу запустить клиент.")
            print("❌ Нет загруженной модели. Сначала обучите модель или выберите существующую.")
            return
        
        self.is_running = True
        
        self.client.add_handler(MessageHandler(self.message_handler))
        
        print("\n=== TELEGRAM RESPONDER ===")
        print(f"Активная модель: {self.model_info}")
        print(f"Режим: {self.mode}")
        print("Запуск клиента...")
        
        await self.client.start()
        
        if self.mode == "only_private_chats":
            target_info = "всех личных чатах" if -1 in self.target_user_ids else f"личных чатах с пользователями {self.target_user_ids}"
        elif self.mode == "only_channel_messages":
            target_info = "всех беседах/группах" if -1 in self.target_channel_ids else f"беседах/группах {self.target_channel_ids}"
        else:
            target_info = f"любых чатах для пользователей {self.target_user_ids}"
            
        print(f"✅ Бот успешно запущен и готов отвечать в {target_info}")
        print("Нажмите Ctrl+C для остановки.")
        
        while self.is_running:
            await asyncio.sleep(1)

    async def stop(self):
        self.is_running = False
        await self.client.stop()
        logger.info("Клиент остановлен")
