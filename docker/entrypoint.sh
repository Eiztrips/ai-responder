#!/usr/bin/env bash
# Entrypoint для контейнера AI-Responder.
# Прокидывает переменные окружения под выбранное устройство и запускает CMD.

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

DEVICE="${AI_RESPONDER_DEVICE:-auto}"
case "${DEVICE}" in
    cuda)
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        ;;
    cpu)
        export CUDA_VISIBLE_DEVICES=""
        ;;
    auto|*)
        # выбор делает Python через aiResponder.ml.device.detect_device
        :
        ;;
esac

mkdir -p /app/data/raw /app/data/datasets /app/models /app/rag_index /app/logs /app/sessions /app/.hf-cache

exec "$@"
