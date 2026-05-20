#!/usr/bin/env bash
# Нативный запуск AI-Responder на macOS с использованием Apple MPS.
# Docker не пропускает Metal внутрь контейнера, поэтому для маков мы
# используем локальный virtualenv и запускаем приложение напрямую.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ ! -d .venv ]]; then
    echo "→ Создаю .venv (Python 3.11)..."
    python3.11 -m venv .venv 2>/dev/null || python3 -m venv .venv
fi

source .venv/bin/activate

if ! python -c "import aiResponder" 2>/dev/null; then
    echo "→ Устанавливаю зависимости (один раз)..."
    pip install --upgrade pip wheel >/dev/null
    pip install -r requirements-mac.txt
fi

export AI_RESPONDER_DEVICE="${AI_RESPONDER_DEVICE:-mps}"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

exec python -m aiResponder
