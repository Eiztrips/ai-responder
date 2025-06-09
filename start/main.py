import os
from src.utils.data_processor import DataProcessor
from src.ml.model_trainer import ModelTrainer
from src.utils.inference import ResponseGenerator
from src.bot.telegram_client import TelegramResponder
import asyncio
from tkinter import filedialog
import tkinter as tk
import json
from decouple import config, UndefinedValueError

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config', 'config.json')
ENV_FILE = os.path.join(os.path.dirname(__file__), '.env')

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Ошибка в формате JSON файла конфигурации: {e}")
        except Exception as e:
            print(f"Не удалось загрузить конфигурацию: {e}")
    return {"selected_model": None, "training_device": None}

def save_config(config):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Не удалось сохранить конфигурацию: {e}")

def update_env_value(key, new_value):
    if not os.path.exists(ENV_FILE):
        print(f"Файл .env не найден по пути {ENV_FILE}")
        return False

    with open(ENV_FILE, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}=") or line.startswith(f"{key} ="):
            comment = ""
            if "#" in line:
                comment = line[line.find("#"):]

            lines[i] = f"{key}={new_value} {comment}" if comment else f"{key}={new_value}\n"
            updated = True
            break

    if updated:
        with open(ENV_FILE, 'w', encoding='utf-8') as file:
            file.writelines(lines)
        return True
    else:
        print(f"Ключ {key} не найден в файле .env")
        return False

def display_menu():
    print("\n===== AI-Responder Menu =====")
    print("1. Обучить новую модель")
    print("2. Спарсить JSON файл в датасет")
    print("3. Показать доступные модели")
    print("4. Выбрать модель для использования")
    print("5. Изменить режим работы Telegram клиента")
    print("6. Запустить Telegram клиент")
    print("7. Выбрать устройство для обучения")
    print("8. Настроить Telegram API")
    print("9. Выход")
    return input("Выберите опцию (1-9): ")

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

def change_bot_mode():
    current_mode = config('MODE', default='only_private_chats')

    print("\n===== Изменение режима работы бота =====")
    print(f"Текущий режим: {current_mode}")
    print("\nДоступные режимы:")
    print("1. only_private_chats - отвечать только в личных чатах")
    print("2. only_channel_messages - отвечать только в беседах/группах")
    print("3. stalker - отвечать пользователям из списка TARGET_USER_IDS в любых чатах")

    choice = input("\nВыберите режим (1-3) или 0 для отмены: ")

    mode_map = {
        "1": "only_private_chats",
        "2": "only_channel_messages",
        "3": "stalker"
    }

    if choice in mode_map:
        new_mode = mode_map[choice]
        if update_env_value("MODE", new_mode):
            print(f"\n✅ Режим успешно изменен на: {new_mode}")
            return True
        else:
            print("\n❌ Не удалось изменить режим. Проверьте файл .env")
            return False
    elif choice == "0":
        print("Изменение режима отменено.")
        return False
    else:
        print("Неверный выбор.")
        return False

def select_training_device(model_trainer):
    config = load_config()
    current_device = config.get("training_device", model_trainer.get_current_device())

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
                config["training_device"] = selected_device
                save_config(config)
                print(f"\n✅ Устройство для обучения изменено на: {available_devices[selected_device]}")
            else:
                print("\n❌ Не удалось установить выбранное устройство.")
        else:
            print("Неверный выбор.")

    except ValueError:
        print("Пожалуйста, введите число.")

def configure_telegram_api():
    """Позволяет настроить параметры Telegram API"""
    print("\n===== Настройка Telegram API =====")

    if not os.path.exists(ENV_FILE):
        with open(ENV_FILE, 'w', encoding='utf-8') as f:
            f.write("# Telegram API Settings\n")
            f.write("API_ID=\n")
            f.write("API_HASH=\n")
            f.write("PHONE=\n")
            f.write("LOGIN=user\n\n")
            f.write("# Bot Settings\n")
            f.write("MODE=only_private_chats\n")
            f.write("TARGET_USER_IDS=-1\n")
            f.write("TARGET_CHANNEL_IDS=-1\n")
        print("Создан новый файл .env с шаблоном настроек")

    try:
        current_api_id = config('API_ID', default='')
        current_api_hash = config('API_HASH', default='')
        current_phone = config('PHONE', default='')
        current_login = config('LOGIN', default='user')
    except:
        current_api_id = ''
        current_api_hash = ''
        current_phone = ''
        current_login = 'user'

    print(f"\nТекущие настройки:")
    print(f"1. API_ID: {'*'*len(current_api_id) if current_api_id else 'Не задан'}")
    print(f"2. API_HASH: {'*'*len(current_api_hash) if current_api_hash else 'Не задан'}")
    print(f"3. PHONE: {current_phone if current_phone else 'Не задан'}")
    print(f"4. LOGIN: {current_login}")

    print("\nВыберите параметр для изменения (1-4) или 0 для возврата в меню:")
    choice = input("> ")

    if choice == "1":
        new_api_id = input("Введите API_ID (числовой ID): ")
        if new_api_id.strip():
            update_env_value("API_ID", new_api_id)
            print("✅ API_ID успешно обновлен")

    elif choice == "2":
        new_api_hash = input("Введите API_HASH (строка): ")
        if new_api_hash.strip():
            update_env_value("API_HASH", new_api_hash)
            print("✅ API_HASH успешно обновлен")

    elif choice == "3":
        new_phone = input("Введите номер телефона (с кодом страны, например +7xxxxxxxxxx): ")
        if new_phone.strip():
            update_env_value("PHONE", new_phone)
            print("✅ PHONE успешно обновлен")

    elif choice == "4":
        new_login = input("Введите логин (используется для имени файла сессии): ")
        if new_login.strip():
            update_env_value("LOGIN", new_login)
            print("✅ LOGIN успешно обновлен")

    elif choice == "0":
        return

    else:
        print("Неверный выбор")

    # Удаляем существующий файл сессии, если были изменены API_ID, API_HASH или PHONE
    if choice in ["1", "2", "3"]:
        session_dir = os.path.join(os.path.dirname(__file__), 'src', 'bot', 'session')
        if os.path.exists(session_dir):
            for file in os.listdir(session_dir):
                if file.endswith(".session"):
                    os.remove(os.path.join(session_dir, file))
                    print(f"Файл сессии {file} удален для применения новых настроек")

async def main():
    data_processor = DataProcessor()
    config_data = load_config()

    # Безопасная инициализация ModelTrainer с проверкой поддержки device
    try:
        model_trainer = ModelTrainer(device=config_data.get("training_device"))
    except TypeError:
        print("Предупреждение: Ваша версия ModelTrainer не поддерживает выбор устройства. Используется CPU.")
        model_trainer = ModelTrainer()  # Пробуем без аргумента device

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
                file_idx = int(input("\nВ��берите датасет для обучения (номер): ")) - 1
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

                        # Показываем текущее устройство
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
            print("\nВыберите JSON файл для конвертации в дат��сет")

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

                if config_data.get("selected_model") == os.path.join(model_trainer.model_dir, model['name']):
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

                if config_data.get("selected_model") == os.path.join(model_trainer.model_dir, model['name']):
                    print("   ✅ ТЕКУЩАЯ МОДЕЛЬ")

            try:
                model_idx = int(input("\nВыберите модель (номер) или 0 для отмены: "))
                if 1 <= model_idx <= len(models):
                    selected_model = models[model_idx-1]
                    model_path = os.path.join(model_trainer.model_dir, selected_model['name'])

                    test_generator = ResponseGenerator()
                    if test_generator.load_model(model_path):
                        config_data["selected_model"] = model_path
                        save_config(config_data)
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
            change_bot_mode()

        elif choice == "6":
            print("Запуск Telegram клиента...")

            model_path = config_data.get("selected_model")
            if not model_path:
                models = model_trainer.list_trained_models()
                if models:
                    print("\n⚠️ Модель не выбрана. Будет использована последняя обученная модель.")
                    print("Для выбора конкретной модели используйте пункт 4 в главном меню.")
                else:
                    print("\n❌ Нет доступных моделей. Сначала обучите модель.")
                    continue

            try:
                # Проверка минимальной конфигурации перед запуском
                try:
                    api_id = config('API_ID')
                    api_hash = config('API_HASH')
                    phone = config('PHONE')
                except UndefinedValueError as e:
                    print(f"\n❌ Отсутствует обязательный параметр в .env файле: {e}")
                    print("Используйте пункт 8 для настройки Telegram API.")
                    continue

                responder = TelegramResponder(model_path)
                try:
                    await responder.start()
                except ValueError as e:
                    print(f"\n❌ Ошибка конфигурации: {e}")
                    print("Используйте пункт 8 для настройки Telegram API.")
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

        elif choice == "7":
            # Проверяем поддержку выбора устройства
            if hasattr(model_trainer, "get_available_devices") and callable(getattr(model_trainer, "get_available_devices")):
                select_training_device(model_trainer)
            else:
                print("\n❌ Ваша версия программы не поддерживает выбор устройства обучения.")
                print("Пожалуйста, убедитесь, что установлена последняя верс��я.")

        elif choice == "8":
            configure_telegram_api()

        elif choice == "9":
            print("Выход из программы...")
            break

        else:
            print("Неверный выбор. Пожалуйста, выберите 1-9")

if __name__ == "__main__":
    asyncio.run(main())
