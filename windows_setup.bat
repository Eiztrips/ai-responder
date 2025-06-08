@echo off
setlocal enabledelayedexpansion

echo === AI Telegram Responder - Установка для Windows ===

REM Проверка наличия Python
python --version 2>NUL
if errorlevel 1 (
    echo Ошибка: Python не установлен
    echo Пожалуйста, установите Python 3.8 или выше с сайта python.org
    pause
    exit /b 1
)

REM Проверка версии Python
for /f "tokens=2 delims=." %%I in ('python -c "import sys; print(sys.version.split('.')[0])"') do set PYTHON_VER=%%I
if %PYTHON_VER% LSS 8 (
    echo Ошибка: Требуется Python 3.8 или выше
    echo Текущая версия: Python 3.%PYTHON_VER%
    pause
    exit /b 1
)

REM Создание виртуального окружения
if not exist venv (
    echo Создание виртуального окружения...
    python -m venv venv
) else (
    echo Виртуальное окружение уже существует
)

REM Активация виртуального окружения и установка зависимостей
echo Активация виртуального окружения...
call venv\Scripts\activate.bat

echo Установка зависимостей...
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

REM Проверка CUDA
python -c "import torch; print('CUDA доступен' if torch.cuda.is_available() else 'CUDA недоступен')"

REM Создание конфигурационных файлов
if not exist config (
    echo Создание директории config...
    mkdir config
)

if not exist config\config.json (
    echo Создание config.json...
    echo {"selected_model": "", "training_device": "cuda"} > config\config.json
)

if not exist .env (
    echo Создание файла .env...
    copy .env_example .env
    echo Пожалуйста, настройте параметры в файле .env
)

echo.
echo Установка завершена успешно!
echo Пожалуйста, не забудьте:
echo 1. Настроить параметры в файле .env
echo 2. Добавить свои датасеты в директорию src/utils/datasets
echo.
echo Для запуска используйте: python start_windows.py
echo.

pause