SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

COMPOSE ?= docker compose

.PHONY: help build build-cpu build-cuda up run run-cpu run-cuda run-mac shell logs clean install-mac venv-mac

help: ## Показать список команд
	@awk 'BEGIN {FS = ":.*##"; printf "\nЦели:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

build: build-cpu ## Собрать CPU-образ (алиас)

build-cpu: ## Собрать CPU-образ Docker
	$(COMPOSE) --profile cpu build app

build-cuda: ## Собрать CUDA-образ Docker
	$(COMPOSE) --profile cuda build app-cuda

run: run-cpu ## Запустить меню (CPU-профиль)

run-cpu: ## Запустить интерактивное меню в CPU-контейнере
	$(COMPOSE) --profile cpu run --rm app

run-cuda: ## Запустить интерактивное меню в CUDA-контейнере (нужен nvidia-container-toolkit)
	$(COMPOSE) --profile cuda run --rm app-cuda

run-mac: install-mac ## macOS / Apple Silicon: запустить нативно с MPS (вне Docker)
	@./scripts/run_native.sh

install-mac: venv-mac ## Установить зависимости в .venv для маков
	@./scripts/install_native.sh

venv-mac: ## Создать .venv (Python 3.11+)
	@if [ ! -d .venv ]; then python3.11 -m venv .venv || python3 -m venv .venv; fi

shell: ## Открыть bash в CPU-контейнере
	$(COMPOSE) --profile cpu run --rm --entrypoint /bin/bash app

logs: ## Tail логов в logs/ai-responder.log
	@tail -f logs/ai-responder.log

clean: ## Удалить кеши/PyCache (без данных и моделей)
	@find . -type d \( -name __pycache__ -o -name .mypy_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	@rm -rf .hf-cache
