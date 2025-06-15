#!/bin/bash

set -e

cd "$(dirname "$0")/../../"

if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 не найден. Установите его через пакетный менеджер вашей системы (например, sudo apt install python3)"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Создаю виртуальное окружение в директории проекта..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Устанавливаю зависимости из requirements.txt..."
pip install --upgrade pip
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "⚠️ Файл requirements.txt не найден в директории проекта, пропускаю установку зависимостей."
fi

echo "Запуск start_linux.py..."
python3 start/linux/start_linux.py