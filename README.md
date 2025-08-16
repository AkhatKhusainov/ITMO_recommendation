# itmo-ai-programs-chatbot

Чат-бот для программ ИТМО: парсер страниц, диалоговый движок, Telegram-бот и FastAPI API.

## Краткое описание

- Парсит 2 страницы магистерских программ ИТМО.
- Отвечает на вопросы о программах через диалоговый движок.
- Telegram-бот (python-telegram-bot 20.x).
- Опционально — веб-сервис через FastAPI.

## Как запустить локально

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e .
# или
pip install -r requirements.txt
cp .env.example .env
pre-commit install
make run-bot      # Запуск Telegram-бота
make run-api      # Запуск FastAPI API
```

## Переменные окружения

- `TELEGRAM_TOKEN` — токен Telegram-бота (обязателен для бота)
- `PORT` — порт для API (по умолчанию 8000)

## Запуск API

```sh
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

## Запуск Docker

```sh
docker compose up --build
```

## Команды Makefile

- `make install` — установка зависимостей
- `make fmt` — автоформатирование (black/isort)
- `make lint` — статический анализ (ruff/black/isort)
- `make type` — проверка типов (mypy)
- `make test` — тесты (pytest)
- `make run-api` — запуск API
- `make run-bot` — запуск Telegram-бота
- `make docker-build` — сборка Docker-образа
- `make docker-run` — запуск контейнера
