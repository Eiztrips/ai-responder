"""Telegram-клиент на Pyrogram.

Подписывается на сообщения и отвечает с помощью :class:`ResponseGenerator`.
RAG используется автоматически, если индекс существует для выбранной модели.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger
from pyrogram import Client
from pyrogram.enums import ChatAction, ChatType
from pyrogram.errors import (
    AuthKeyUnregistered,
    BadRequest,
    FloodWait,
    PhoneCodeInvalid,
    PhoneNumberInvalid,
    Unauthorized,
)
from pyrogram.handlers import MessageHandler
from pyrogram.types import Message

from aiResponder.config.settings import Settings
from aiResponder.inference.generator import ResponseGenerator
from aiResponder.utils.paths import project_root


class TelegramResponder:
    """Pyrogram-обёртка с интеграцией LLM/RAG."""

    def __init__(self, settings: Settings, model_path: str | Path | None = None) -> None:
        self.settings = settings
        telegram = settings.telegram
        if not telegram.api_id or not telegram.api_hash or not telegram.login:
            raise ValueError(
                "Telegram-конфиг неполный: укажите api_id, api_hash и login в config.yaml или .env"
            )

        self.mode = telegram.mode
        self.target_user_ids = list(telegram.target_user_ids)
        self.target_channel_ids = list(telegram.target_channel_ids)

        session_dir = project_root() / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        self.session_path = session_dir / telegram.login

        self.client = Client(
            name=str(self.session_path),
            api_id=telegram.api_id,
            api_hash=telegram.api_hash,
            phone_number=telegram.phone,
            app_version=f"AI Responder {settings.app.version}",
            device_model="PC",
            system_version="Docker/Linux",
            sleep_threshold=10,
        )
        self.response_generator = ResponseGenerator(settings, model_path=model_path)
        self.is_running = False
        logger.info(
            "TelegramResponder готов | модель: {} | режим: {} | users: {} | channels: {}",
            self._model_label(),
            self.mode,
            self.target_user_ids,
            self.target_channel_ids,
        )

    def _model_label(self) -> str:
        meta = self.response_generator.metadata
        return meta.get("target_user") or "—"

    # ------------------------------------------------------------------ #
    # Маршрутизация входящих сообщений
    # ------------------------------------------------------------------ #
    def _allowed(self, message: Message) -> bool:
        if self.mode == "only_private_chats":
            if message.chat.type != ChatType.PRIVATE:
                return False
            uid = message.from_user.id if message.from_user else None
            return uid is not None and (-1 in self.target_user_ids or uid in self.target_user_ids)
        if self.mode == "only_channel_messages":
            if message.chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP, ChatType.CHANNEL):
                return False
            return -1 in self.target_channel_ids or message.chat.id in self.target_channel_ids
        if self.mode == "stalker":
            uid = message.from_user.id if message.from_user else None
            return uid is not None and uid in self.target_user_ids
        logger.warning("Неизвестный режим бота: {}", self.mode)
        return False

    async def _reply(self, client: Client, message: Message) -> None:
        try:
            result = self.response_generator.generate(message.text or "")
            await client.send_chat_action(message.chat.id, ChatAction.TYPING)
            await asyncio.sleep(min(0.1 * len(result.text or ""), 3))
            await message.reply(result.text)
            logger.info(
                "Ответил в {} (rag={}, hits={})",
                message.chat.id,
                result.used_rag,
                len(result.hits),
            )
        except FloodWait as exc:
            logger.warning("FloodWait: ждать {}s", exc.value)
            await asyncio.sleep(exc.value)
            await self._reply(client, message)
        except Exception:  # noqa: BLE001
            logger.exception("Сбой при ответе")

    async def _handler(self, client: Client, message: Message) -> None:
        if message.outgoing or not message.text:
            return
        try:
            if self._allowed(message):
                await self._reply(client, message)
        except Exception:  # noqa: BLE001
            logger.exception("Ошибка обработки входящего сообщения")

    # ------------------------------------------------------------------ #
    # Жизненный цикл
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        if self.response_generator.model is None:
            logger.error("Нет загруженной модели — не запускаю клиент")
            return
        self.is_running = True
        self.client.add_handler(MessageHandler(self._handler))

        logger.info("Подключение к Telegram API…")
        try:
            await asyncio.wait_for(self.client.start(), timeout=60.0)
        except asyncio.TimeoutError:
            logger.error("Тайм-аут подключения к Telegram API")
            self.is_running = False
            return
        except (AuthKeyUnregistered, BadRequest, Unauthorized) as exc:
            logger.error("Ошибка авторизации: {}", exc)
            session_file = Path(str(self.session_path) + ".session")
            if session_file.exists():
                session_file.unlink(missing_ok=True)
                logger.info("Удалён файл сессии {}", session_file)
            self.is_running = False
            return
        except (PhoneNumberInvalid, PhoneCodeInvalid) as exc:
            logger.error("Проблема с телефоном: {}", exc)
            self.is_running = False
            return

        me = await self.client.get_me()
        logger.info("Подключён как: {} {} (@{})", me.first_name or "", me.last_name or "", me.username)
        try:
            while self.is_running:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Получен сигнал остановки")
        finally:
            await self.stop()

    async def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        try:
            await self.client.stop()
            logger.info("Telegram-клиент остановлен")
        except Exception:  # noqa: BLE001
            logger.exception("Ошибка при остановке клиента")
