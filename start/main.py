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
import yaml
import asyncio
import tkinter as tk
from tkinter import filedialog
from typing import Dict, Any

from src.utils.data_processor import DataProcessor
from src.ml.model_trainer import ModelTrainer
from src.utils.inference import ResponseGenerator
from src.bot.telegram_client import TelegramResponder

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
YAML_CONFIG_FILE = os.path.join(BASE_DIR, 'config', 'config.yaml')

class ConfigManager:

    def __init__(self):
        self.config = {}
        self.yaml_config = self._load_yaml_config()

    def _load_yaml_config(self) -> Dict[str, Any]:
        if os.path.exists(YAML_CONFIG_FILE):
            try:
                with open(YAML_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Ошибка загрузки YAML конфигурации: {e}")
        return {}

    def save_yaml_config(self):
        os.makedirs(os.path.dirname(YAML_CONFIG_FILE), exist_ok=True)
        try:
            with open(YAML_CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(self.yaml_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            print(f"✅ Конфигурация успешно сохранена в {YAML_CONFIG_FILE}")
            return True
        except Exception as e:
            print(f"❌ Не удалось сохранить YAML конфигурацию: {e}")
            return False

    def get(self, section: str, key: str, default: Any = None) -> Any:
        try:
            if section in self.yaml_config and key in self.yaml_config[section]:
                return self.yaml_config[section][key]
            return default
        except Exception:
            return default

    def get_section(self, section: str, default: Any = None) -> Any:
        return self.yaml_config.get(section, default)

    def get_app_config(self, key: str, default: Any = None) -> Any:
        return self.yaml_config.get('main_settings', {}).get(key, default)

    def set_app_config(self, key: str, value: Any):
        if 'main_settings' not in self.yaml_config:
            self.yaml_config['main_settings'] = {}
        self.yaml_config['main_settings'][key] = value
        self.save_yaml_config()

    def get_ml_config(self) -> Dict[str, Any]:
        return self.yaml_config.get('ml', {})

    def get_telegram_config(self) -> Dict[str, Any]:
        return self.yaml_config.get('telegram', {})

    def get_inference_config(self) -> Dict[str, Any]:
        return self.yaml_config.get('inference', {})
    
    def get_data_processor_config(self) -> Dict[str, Any]:
        return self.yaml_config.get('data_processor', {})
    
    def get_full_config(self) -> Dict[str, Any]:
        return self.yaml_config

    def update_yaml_setting(self, section: str, key: str, value: Any):
        if section not in self.yaml_config:
            self.yaml_config[section] = {}
        self.yaml_config[section][key] = value

        if section != 'main_settings':
            if 'main_settings' not in self.yaml_config:
                self.yaml_config['main_settings'] = {}

            mapping = {
                'ml': {'model': 'model'},
                'telegram': {'mode': 'telegram_mode'},
                'inference': {'active_profile': 'active_generation_profile'},
                'main_settings': {}
            }
            
            if section in mapping and key in mapping[section]:
                main_key = mapping[section][key]
                self.yaml_config['main_settings'][main_key] = value


def select_json_file():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    root.update()
    file_path = filedialog.askopenfilename(
        title="Выберите JSON файл",
        filetypes=[("JSON files", "*.json")]
    )
    root.destroy()
    return file_path if file_path else None


def change_bot_mode(config_manager: ConfigManager):
    telegram_config = config_manager.get_telegram_config()
    current_mode = telegram_config.get('mode', 'only_private_chats')
    mode_descriptions = telegram_config.get('mode_descriptions', {})

    print("\n===== Изменение режима работы бота =====")
    print(f"Текущий режим: {current_mode}")
    print("\nДоступные режимы:")
    
    modes = list(mode_descriptions.items())
    for i, (mode, description) in enumerate(modes, 1):
        print(f"{i}. {mode} - {description}")

    choice = input("\nВыберите режим (1-{}) или 0 для отмены: ".format(len(modes)))

    if choice.isdigit() and 1 <= int(choice) <= len(modes):
        mode_idx = int(choice) - 1
        new_mode = modes[mode_idx][0]
        print(f"\n✅ Режим успешно изменен на: {new_mode} (измените config.yaml для сохранения)")
        return True
    elif choice == "0":
        print("Изменение режима отменено.")
        return False
    else:
        print("Неверный выбор.")
        return False


def select_training_device(model_trainer, config_manager: ConfigManager):
    current_device = config_manager.get_app_config("training_device", model_trainer.get_current_device())

    print("\n===== Выбор устройства для обучения =====")
    print(f"Текущее устройство: {current_device}")

    available_devices = model_trainer.get_available_devices()

    print("\nДоступные устройства:")
    options = list(available_devices.items())
    for i, (device_id, device_name) in enumerate(options, 1):
        print(f"{i}. {device_name} ({'текущее' if device_id == current_device else 'доступно'})")

    try:
        choice = int(input("\nВыберите устройство (номер) или 0 для отмены: "))

        if choice == 0:
            print("Выбор устройства отменен.")
            return

        if 1 <= choice <= len(options):
            selected_device = options[choice-1][0]
            if model_trainer.set_device(selected_device):
                config_manager.set_app_config("training_device", selected_device)
                print(f"\n✅ Устройство для обучения изменено на: {available_devices[selected_device]}")
            else:
                print("\n❌ Не удалось установить выбранное устройство.")
        else:
            print("Неверный выбор.")

    except ValueError:
        print("Пожалуйста, введите число.")

def settings_menu(config_manager: ConfigManager, model_trainer):
    """Меню настроек с улучшенной читаемостью и функционалом сохранения."""
    while True:
        print("\n" + "="*40)
        print("        НАСТРОЙКИ AI-RESPONDER")
        print("="*40)
        print("1. Профиль генерации текста")
        print("2. Модель по умолчанию")
        print("3. Режим Telegram-бота")
        print("4. Устройство для обучения")
        print("5. Сохранить все настройки")
        print("6. Вернуться в главное меню")
        print("="*40)
        
        choice = input("Выберите опцию (1-6): ")

        if choice == "1":
            _handle_generation_profile_settings(config_manager)
        elif choice == "2":
            _handle_default_model_settings(config_manager)
        elif choice == "3":
            _handle_telegram_mode_settings(config_manager)
        elif choice == "4":
            if hasattr(model_trainer, "get_available_devices"):
                select_training_device(model_trainer, config_manager)
            else:
                print("❌ Не поддерживается в вашей версии.")
        elif choice == "5":
            if config_manager.save_yaml_config():
                print("\n✅ Все настройки успешно сохранены в файл конфигурации!")
            else:
                print("\n❌ Не удалось сохранить настройки. Проверьте права доступа к файлу.")
        elif choice == "6":
            if _prompt_for_save_if_needed(config_manager):
                config_manager.save_yaml_config()
            break
        else:
            print("❌ Неверный выбор. Пожалуйста, введите число от 1 до 6.")
            

def _handle_generation_profile_settings(config_manager: ConfigManager):
    print("\n" + "-"*40)
    print("     ПРОФИЛИ ГЕНЕРАЦИИ ТЕКСТА")
    print("-"*40)

    profiles = config_manager.yaml_config.get("inference", {}) \
        .get("model", {}) \
        .get("generation_profiles", {})

    if not profiles:
        print("❌ Не найдено ни одного профиля генерации в конфиге.")
        print("DEBUG: inference.model.generation_profiles =", config_manager.yaml_config.get("inference", {}).get("model", {}).get("generation_profiles"))
        print("DEBUG: Полный config['inference'] =", config_manager.yaml_config.get("inference"))
        return

    active = config_manager.yaml_config.get("main_settings", {}).get("active_generation_profile") \
        or config_manager.yaml_config.get("inference", {}).get("active_profile", "creative")

    print("Доступные профили генерации:")
    for i, key in enumerate(profiles, 1):
        current = "✓" if key == active else " "
        profile = profiles[key]
        print(f"{i}. [{current}] {key}")
        print(f"   Длина: {profile.get('max_length', 'Не указано')}, "
              f"Температура: {profile.get('temperature', 'Не указано')}")

    idx = input("\nВыберите профиль (номер) или 0 для отмены: ")

    if idx.isdigit() and 1 <= int(idx) <= len(profiles):
        selected = list(profiles.keys())[int(idx)-1]
        config_manager.update_yaml_setting('inference', 'active_profile', selected)
        print(f"\n✅ Активный профиль изменен на: {selected}")
    elif idx == "0":
        print("Выбор профиля отменен.")
    else:
        print("❌ Неверный выбор.")


def _handle_default_model_settings(config_manager: ConfigManager):
    print("\n" + "-"*40)
    print("     НАСТРОЙКА МОДЕЛИ ПО УМОЛЧАНИЮ")
    print("-"*40)
    
    ml_cfg = config_manager.get_ml_config()
    current_model = ml_cfg.get("model", "gpt2")
    
    print(f"Текущая модель: {current_model}")
    print("\nПримеры доступных моделей:")
    print("- gpt2")
    print("- facebook/opt-125m")
    print("- EleutherAI/pythia-70m")
    print("- sberbank-ai/rugpt3small_based_on_gpt2")
    
    new_model = input("\nВведите название модели (или Enter для отмены): ").strip()
    
    if new_model:
        config_manager.update_yaml_setting('ml', 'model', new_model)
        print(f"\n✅ Модель изменена на: {new_model}")
    else:
        print("Изменение модели отменено.")


def _handle_telegram_mode_settings(config_manager: ConfigManager):
    print("\n" + "-"*40)
    print("     НАСТРОЙКА РЕЖИМА TELEGRAM БОТА")
    print("-"*40)
    
    yaml_cfg = config_manager.yaml_config
    telegram_cfg = yaml_cfg.get("telegram", {})
    mode_descriptions = telegram_cfg.get("mode_descriptions", {})

    current_mode = yaml_cfg.get("main_settings", {}).get("telegram_mode") \
        or telegram_cfg.get("mode", "only_private_chats")
    
    print(f"Текущий режим: {current_mode}")
    print("\nДоступные режимы:")
    
    modes = list(mode_descriptions.items())
    for i, (mode, description) in enumerate(modes, 1):
        current = "✓" if mode == current_mode else " "
        print(f"{i}. [{current}] {mode}")
        print(f"   {description}")
    
    choice_mode = input("\nВыберите режим (1-{}) или 0 для отмены: ".format(len(modes)))
    
    if choice_mode.isdigit() and 1 <= int(choice_mode) <= len(modes):
        mode_idx = int(choice_mode) - 1
        new_mode = modes[mode_idx][0]

        config_manager.update_yaml_setting('telegram', 'mode', new_mode)
        print(f"\n✅ Режим успешно изменен на: {new_mode}")
    elif choice_mode == "0":
        print("Изменение режима отменено.")
    else:
        print("❌ Неверный выбор.")


def _prompt_for_save_if_needed(config_manager: ConfigManager) -> bool:
    response = input("\nСохранить изменения в файл конфигурации? (д/н): ")
    return response.lower() in ['д', 'y', 'yes', 'да']


def display_menu():
    print("\n" + "="*40)
    print("          AI-RESPONDER")
    print("="*40)
    print("1. Обучить модель")
    print("2. Конвертировать JSON в датасет")
    print("3. Список моделей")
    print("4. Выбрать модель")
    print("5. Запустить Telegram-бота")
    print("6. Настройки")
    print("7. Выход")
    print("="*40)
    return input("Выберите опцию (1-7): ")

async def main():
    config_manager = ConfigManager()
    data_processor = DataProcessor(config_manager)
    ml_config = config_manager.get_ml_config()
    model_name = ml_config.get('model', 'gpt2')
    try:
        model_trainer = ModelTrainer(
            config_manager,
            device=config_manager.get_app_config("training_device"),
            model=model_name
        )
    except TypeError:
        print("Предупреждение: Ваша версия ModelTrainer не поддерживает выбор устройства. Используется CPU.")
        model_trainer = ModelTrainer(config_manager, model=model_name)

    while True:
        choice = display_menu()

        if choice == "1":
            datasets = data_processor.get_available_datasets()

            if not datasets:
                print("Датасеты не найдены. Добавьте их через опцию 2.")
                continue

            print("\nДоступные датасеты:")
            for i, dataset in enumerate(datasets, 1):
                print(f"{i}. {dataset['name']} ({dataset['type'].upper()})")

            try:
                file_idx = int(input("\nВыберите датасет для обучения (номер): ")) - 1
                if 0 <= file_idx < len(datasets):
                    selected_dataset = datasets[file_idx]

                    data = data_processor.load_dataset(selected_dataset)
                    participants = data_processor.get_chat_participants(data)

                    print("\nДоступные пользователи для имитации:")
                    for i, user in enumerate(participants, 1):
                        print(f"{i}. {user['name']} (Сообщений: {user['message_count']})")

                    user_idx = int(input("\nВыберите пользователя для имитации (номер): ")) - 1
                    if 0 <= user_idx < len(participants):
                        selected_user = participants[user_idx]
                        print(f"\nВыбран пользователь: {selected_user['name']} с ID {selected_user['id']}")

                        current_device = model_trainer.get_current_device()
                        available_devices = model_trainer.get_available_devices()
                        print(f"\nТекущее устройство для обучения: {available_devices.get(current_device, current_device)}")

                        if input("Начать обучение? (д/н): ").lower() in ['д', 'y', 'yes', 'да']:
                            model_path = model_trainer.train_model(selected_dataset, selected_user['id'])
                            print(f"Модель успешно обучена и сохранена: {model_path}")
                    else:
                        print("Неверный выбор пользователя")
                else:
                    print("Неверный выбор файла")
            except ValueError:
                print("Пожалуйста, введите число")

        elif choice == "2":
            print("\nВыберите JSON файл для конвертации в датасет")

            json_file_path = select_json_file()

            if not json_file_path:
                print("Выбор файла отменен")
                continue

            print(f"Выбран файл: {json_file_path}")

            output_name = input("Введите имя для выходного файла (оставьте пустым для автоматической генерации): ")
            output_name = output_name.strip() if output_name.strip() else None

            result = data_processor.parse_json_to_dataset(json_file_path, output_name)

            if result:
                print(f"Датасет успешно создан в форматах CSV и JSONL")
                print(f"CSV: {result['csv']['path']}")
                print(f"JSONL: {result['jsonl']['path']}")
            else:
                print("Ошибка при создании датасета")

        elif choice == "3":
            models = model_trainer.list_trained_models()

            if not models:
                print("Обученных моделей не найдено")
                continue

            print("\nДоступные модели:")
            for i, model in enumerate(models, 1):
                metadata = model["metadata"]
                print(f"{i}. {model['name']}")
                print(f"   Пользователь: {metadata.get('target_user', 'Неизвестно')}")
                print(f"   Исходный файл: {metadata.get('source_file', 'Неизвестно')} ({metadata.get('source_file_type', 'unknown').upper()})")
                print(f"   Обучающих пар: {metadata.get('training_pairs_count', 'Неизвестно')}")
                print(f"   Устройство обучения: {metadata.get('training_device', 'Неизвестно')}")

                if config_manager.get_app_config("selected_model") == os.path.join(model_trainer.model_dir, model['name']):
                    print("   ✅ ТЕКУЩАЯ МОДЕЛЬ")

        elif choice == "4":
            models = model_trainer.list_trained_models()

            if not models:
                print("Обученных моделей не найдено. Сначала обучите модель.")
                continue

            print("\nВыберите модель для использования:")
            for i, model in enumerate(models, 1):
                metadata = model["metadata"]
                print(f"{i}. {model['name']} (Пользователь: {metadata.get('target_user', 'Неизвестно')})")

                if config_manager.get_app_config("selected_model") == os.path.join(model_trainer.model_dir, model['name']):
                    print("   ✅ ТЕКУЩАЯ МОДЕЛЬ")

            try:
                model_idx = int(input("\nВыберите модель (номер) или 0 для отмены: "))
                if 1 <= model_idx <= len(models):
                    selected_model = models[model_idx-1]
                    model_path = os.path.join(model_trainer.model_dir, selected_model['name'])

                    test_generator = ResponseGenerator()
                    if test_generator.load_model(model_path):
                        config_manager.set_app_config("selected_model", model_path)
                        print(f"\n✅ Модель успешно выбрана: {selected_model['name']}")
                        print(f"   Пользователь: {selected_model['metadata'].get('target_user', 'Неизвестно')}")
                    else:
                        print("\n❌ Не удалось загрузить модель. Проверьте файлы модели.")
                elif model_idx == 0:
                    print("Отмена выбора модели.")
                else:
                    print("Неверный выбор модели.")
            except ValueError:
                print("Пожалуйста, введите число")

        elif choice == "5":
            print("Запуск Telegram клиента...")

            model_path = config_manager.get_app_config("selected_model")
            if not model_path:
                models = model_trainer.list_trained_models()
                if models:
                    print("\n⚠️ Модель не выбрана. Будет использована последняя обученная модель.")
                    print("Для выбора конкретной модели используйте пункт 4 в главном меню.")
                else:
                    print("\n❌ Нет доступных моделей. Сначала обучите модель.")
                    continue

            try:
                inference_config = config_manager.get_inference_config()
                responder = TelegramResponder(config_manager, model_path)
                try:
                    await responder.start()
                except ValueError as e:
                    print(f"\n❌ Ошибка конфигурации: {e}")
                    print("Проверьте config.yaml.")
                except KeyboardInterrupt:
                    print("\n⚠️ Остановка клиента...")
                    await responder.stop()
                    print("Клиент остановлен")
                except Exception as e:
                    print(f"\n❌ Ошибка при работе клиента: {e}")
                    if hasattr(responder, "stop"):
                        await responder.stop()

            except Exception as e:
                print(f"\n❌ Не удалось запустить Telegram клиент: {e}")

        elif choice == "6":
            settings_menu(config_manager, model_trainer)

        elif choice == "7":
            print("Выход из программы...")
            break

        else:
            print("Неверный выбор. Пожалуйста, выберите 1-7")


if __name__ == "__main__":
    asyncio.run(main())
