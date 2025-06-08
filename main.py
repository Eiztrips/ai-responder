import os
from src.utils.data_processor import DataProcessor
from src.ml.model_trainer import ModelTrainer
from src.utils.inference import ResponseGenerator
from src.bot.telegram_client import TelegramResponder
import asyncio
from tkinter import filedialog
import tkinter as tk
import json
import re
from decouple import config

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config', 'config.json')
ENV_FILE = os.path.join(os.path.dirname(__file__), '.env')

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"selected_model": None, "training_device": None}

def save_config(config):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def update_env_value(key, new_value):
    """Обновляет значение переменной в файле .env"""
    if not os.path.exists(ENV_FILE):
        print(f"Файл .env не найден по пути {ENV_FILE}")
        return False
        
    with open(ENV_FILE, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    updated = False
    for i, line in enumerate(lines):
        # Ищем строку с нужным ключом
        if line.startswith(f"{key}=") or line.startswith(f"{key}="):
            # ��охраняем комментарий, если есть
            comment = ""
            if "#" in line:
                comment = line[line.find("#"):]
            
            # Обновляем значение
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
    print("8. Выход")
    return input("Выберите опцию (1-8): ")

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
    """Меню выбора устройства для обучения"""
    config = load_config()
    current_device = config.get("training_device", model_trainer.get_current_device())
    
    print("\n===== Выбор устройства для обуче��ия =====")
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
        print("Пожалуйста, введит�� число.")

async def main():
    data_processor = DataProcessor()
    config = load_config()
    model_trainer = ModelTrainer(device=config.get("training_device"))
    
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
                        print(f"{i}. {user['name']} (Со��бщений: {user['message_count']})")
                    
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
                
                if config.get("selected_model") == os.path.join(model_trainer.model_dir, model['name']):
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
                
                if config.get("selected_model") == os.path.join(model_trainer.model_dir, model['name']):
                    print("   ✅ ТЕКУЩАЯ МОДЕЛЬ")
            
            try:
                model_idx = int(input("\nВыберите модель (номер) или 0 для отмены: "))
                if 1 <= model_idx <= len(models):
                    selected_model = models[model_idx-1]
                    model_path = os.path.join(model_trainer.model_dir, selected_model['name'])
                    
                    test_generator = ResponseGenerator()
                    if test_generator.load_model(model_path):
                        config["selected_model"] = model_path
                        save_config(config)
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
            
            model_path = config.get("selected_model")
            if not model_path:
                models = model_trainer.list_trained_models()
                if models:
                    print("\n⚠️ Модель не выбрана. Будет использована последняя обученная модель.")
                    print("Для выбора конкретной модели используйте пункт 4 в главном меню.")
                else:
                    print("\n❌ Нет доступных моделей. Сначала обучите модель.")
                    continue
                    
            responder = TelegramResponder(model_path)
            try:
                await responder.start()
            except KeyboardInterrupt:
                await responder.stop()
                print("Клиент остановлен")
        
        elif choice == "7":
            select_training_device(model_trainer)
        
        elif choice == "8":
            print("Выход из программы...")
            break
        
        else:
            print("Неверный выбор. Пожалуйста, выберите 1-8")

if __name__ == "__main__":
    asyncio.run(main())
