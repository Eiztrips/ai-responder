# AI Telegram Responder

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Персональный Telegram-бот, имитирующий стиль общения выбранного пользователя на
основе истории переписки. Поддерживает RAG (Retrieval-Augmented Generation),
LoRA-обучение и автоматический выбор устройства (NVIDIA CUDA, Apple Silicon MPS,
CPU).

## Что нового в 3.0

- Полный отказ от папки `start/` — единая точка входа `python -m aiResponder`.
- Запуск через **Docker Compose** (профили `cpu` и `cuda`) + нативный режим
  для macOS Apple Silicon с MPS (Docker не пропускает Metal внутрь контейнера).
- **RAG** на FAISS + `intfloat/multilingual-e5-small` — индекс строится по
  обучающим парам, во время инференса подмешивается top-K похожих диалогов как
  few-shot контекст.
- **LoRA** через PEFT — опционально, ускоряет тренировку и снижает требования
  к памяти, что критично для MPS/CPU.
- Конфигурация на **pydantic-settings**: один источник истины
  `config/config.yaml` + переменные окружения. Никаких `eval()` над строками.
- **Улучшенные никнеймы**: NFKC-нормализация, удаление эмодзи / zero-width,
  объединение алиасов по `author_id` и fuzzy-merge через `rapidfuzz`.
- **Конфигурируемый фильтр мусора**: эмодзи-only, URL-only, флуд символов,
  дубли в окне, системные сообщения. По умолчанию мягкие настройки —
  «человеческие» короткие реплики («ок», «+», «да») сохраняются.

## Архитектура

```
config/config.yaml ─┐
.env ───────────────┴──► pydantic Settings ──┐
                                             │
                                             ▼
data/raw/*.json ─► DataProcessor ─► (filters + nicknames) ─► data/datasets/*.{csv,jsonl}
                                             │
                                             ▼
                                       TrainingPair[…]
                                       │            │
                                       ▼            ▼
                              ModelTrainer    RagIndexBuilder
                              (HF Trainer +   (FAISS + e5)
                               LoRA)
                                       │            │
                                       ▼            ▼
                                models/<name>   rag_index/<name>.faiss
                                       │            │
                                       └──┬─────────┘
                                          ▼
                                   ResponseGenerator
                                          │
                                          ▼
                                   TelegramResponder (Pyrogram)
```

## Быстрый старт

### Подготовка

```bash
git clone https://github.com/Eiztrips/ai-responder.git
cd ai-responder
cp .env-example .env
# заполните API_ID, API_HASH, PHONE, LOGIN
```

Сложите выгрузку Telegram (`result.json`) в `data/raw/`.

### Docker Compose (Linux / Windows)

| GPU                | Команда                                                          |
| ------------------ | ---------------------------------------------------------------- |
| NVIDIA CUDA        | `docker compose --profile cuda run --rm app-cuda`                |
| CPU (x86_64/ARM64) | `docker compose --profile cpu run --rm app`                      |
| Сборка             | `make build-cpu` или `make build-cuda`                            |

CUDA-профиль требует установленного [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).

### macOS Apple Silicon (M1/M2/M3) — нативный запуск с MPS

Apple Metal недоступен внутри Docker — для MPS используем `.venv` на хосте.

```bash
make run-mac
```

Эта цель:
1. Создаёт `.venv` (Python 3.11+).
2. Ставит зависимости из `requirements-mac.txt`.
3. Выставляет `AI_RESPONDER_DEVICE=mps` и запускает `python -m aiResponder`.

## Использование

После запуска вы увидите меню:

```
        AI-RESPONDER
========================================
1. Обучить модель
2. Конвертировать JSON в датасет
3. Список моделей
4. Выбрать модель
5. Запустить Telegram-бота
6. Настройки
7. Выход
========================================
```

Типичный рабочий процесс:

1. Положите `result.json` в `data/raw/`.
2. Меню → **2** — сконвертируйте JSON в датасет (попадёт в `data/datasets/`).
3. Меню → **1** — выберите датасет и пользователя для имитации.
4. После обучения модель окажется в `models/`, RAG-индекс — в `rag_index/`.
5. Меню → **4** — выберите модель.
6. Меню → **5** — запустите Telegram-клиент.

## Конфигурация

Все параметры — в [`config/config.yaml`](config/config.yaml). Чувствительные
данные (api_id/hash, phone, login, целевые id) можно переопределить через `.env`
или переменные окружения.

Ключевые блоки:

| Блок            | Назначение                                                                 |
| --------------- | -------------------------------------------------------------------------- |
| `telegram`      | API-доступ к Telegram и режим работы.                                      |
| `training`      | Базовая модель, гиперпараметры по устройствам, LoRA, early stopping.       |
| `inference`     | Профили генерации (`creative`, `precise`, `balanced`, `chatty`) и кэш.     |
| `rag`           | Включение RAG, эмбеддинг-модель, top-K, similarity threshold.              |
| `filters`       | Эвристики фильтрации мусора (пороги, белый список коротких реплик).        |
| `nicknames`     | Чистка ника, fuzzy-merge алиасов.                                          |
| `data_processor`| Регулярки для чистки текста, путь к датасетам.                             |

### RAG

```yaml
rag:
  enabled: true
  embeddings_model: intfloat/multilingual-e5-small
  top_k: 4
  similarity_threshold: 0.55
  index_dir: rag_index
  rebuild_on_train: true
  max_pairs_in_prompt: 4
```

После обучения индекс автоматически сохраняется как
`rag_index/<имя_модели>.faiss` + `<имя_модели>.meta.jsonl`. Во время инференса
сообщение собеседника матчится с базой пар, и top-K похожих подмешиваются в
промпт как few-shot:

```
Q: <похожий вопрос 1>
A: <ответ пользователя 1>
Q: <похожий вопрос 2>
A: <ответ пользователя 2>
Q: <новое сообщение>
A:
```

### LoRA

```yaml
training:
  lora:
    enabled: true
    r: 8
    alpha: 16
    dropout: 0.05
```

Target-модули автодетектятся по архитектуре модели (`gpt2`, `gpt_neox`,
`llama`, `qwen2`, и т.п.).

### Фильтры

Дефолты не убивают короткие человеческие ответы. Если хочется агрессивнее —
поднимите `min_words`, отключите `allow_short_replies` или расширьте
`short_reply_whitelist`.

## Структура проекта

```
ai-responder/
├── config/
│   └── config.yaml
├── docker/
│   ├── Dockerfile.cpu
│   ├── Dockerfile.cuda
│   └── entrypoint.sh
├── docker-compose.yml
├── Makefile
├── scripts/
│   ├── install_native.sh
│   └── run_native.sh
├── src/aiResponder/
│   ├── bot/telegram.py
│   ├── cli.py
│   ├── config/settings.py
│   ├── data/
│   │   ├── filters.py
│   │   ├── nicknames.py
│   │   └── processor.py
│   ├── inference/generator.py
│   ├── ml/
│   │   ├── dataset.py
│   │   ├── device.py
│   │   ├── lora.py
│   │   └── trainer.py
│   ├── rag/
│   │   ├── embeddings.py
│   │   ├── index.py
│   │   └── retriever.py
│   └── utils/
│       ├── logging.py
│       └── paths.py
├── requirements.txt
├── requirements-mac.txt
└── pyproject.toml
```

## Требования

- Python 3.10+
- Docker / Docker Compose (для CPU и CUDA-профилей)
- (опц.) NVIDIA Container Toolkit для CUDA
- (опц.) macOS 12+ с Apple Silicon для MPS

## Полезные команды Makefile

```bash
make help        # все доступные цели
make build-cuda  # собрать CUDA-образ
make run-cpu     # меню в CPU-контейнере
make run-mac     # нативный запуск на Apple Silicon
make shell       # bash внутри контейнера
make logs        # tail logs/ai-responder.log
make clean       # удалить кэши
```

## Лицензия

MIT — см. [LICENSE](LICENSE).
