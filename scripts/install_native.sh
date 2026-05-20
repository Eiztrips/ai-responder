#!/usr/bin/env bash
# Установка зависимостей в локальный .venv для macOS (Apple Silicon).
# Используется только когда нужен реальный MPS — Docker такого не умеет.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ ! -d .venv ]]; then
    python3.11 -m venv .venv 2>/dev/null || python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements-mac.txt

echo
echo "✓ Зависимости установлены в .venv"
echo "  Запуск: scripts/run_native.sh   (или make run-mac)"
