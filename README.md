# Multi-Agent Verification System

Продвинутая система верификации ответов LLM с использованием множественных агентов, каждый из которых выполняет специфическую роль в процессе проверки качества и надежности информации.

## Архитектура

```
main.py          - Главная точка входа, управление рабочим процессом
config.py        - Конфигурация (модели, параметры)
models.py        - Интеграция с LLM (Yandex, OpenRouter)
agents.py        - Агенты (Истец, Критик, Библиотекарь, Присяжные, Судья)
search.py        - Веб-поиск и поиск источников
utils.py         - Вспомогательные функции, логирование
conversation_log.json - Лог всех взаимодействий
results.txt      - Итоговый результат
```

## Агенты

1. **PLAINTIFF (Истец)** - Генерирует подробный, обоснованный ответ на вопрос
2. **CRITIC (Критик)** - Анализирует ответ и выявляет слабые места
3. **LIBRARIAN (Библиотекарь)** - Проводит веб-поиск для проверки фактов
4. **JURY (Жюри)** - Несколько независимых присяжных дают свое мнение
5. **JUDGE (Судья)** - Выносит финальное решение (0-10 баллов, STOP/CONTINUE)

## Установка

```bash
# Зависимости
pip install langgraph langchain langchain-community requests

# Для Yandex Cloud (уже в коде)
pip install yandex-cloud-ml-sdk

# Для веб-поиска (опционально)
pip install ddgs
```

## Использование

### С Yandex Cloud (по умолчанию)

```python
from main import run_system

topics = [
    "Ваш вопрос здесь"
]

run_system(topics)
```

### С OpenRouter (бесплатные модели)

```python
import os
os.environ["OPENROUTER_API_KEY"] = "ваш_api_key"

from main import run_system

topics = ["Ваш вопрос"]

run_system(topics, provider="openrouter", model="meta-llama/llama-2-7b-chat")
```

### Изменение параметров в config.py

```python
CONFIG = {
    "model_provider": "yandex",  # или "openrouter"
    "max_iterations": 3,  # Число итераций верификации
    "use_jury": True,  # Использовать жюри
    "use_librarian": True,  # Использовать поиск источников
    # ...
}
```

## Бесплатные модели OpenRouter

- `meta-llama/llama-2-7b-chat`
- `mistralai/mistral-7b-instruct`
- `gryphe/mythomist-7b`

## Выходные данные

1. **results.txt** - Полный итоговый ответ с красивым форматированием
2. **conversation_log.json** - Детальный лог всех агентов и токенов

## Пример

```bash
python main.py
```

Система обработает темы, выведет весь процесс верификации в реальном времени и сохранит результаты в текстовый файл.

## Особенности

✅ Полные промпты без урезания
✅ Надежный веб-поиск через DuckDuckGo и arXiv
✅ Поддержка нескольких LLM провайдеров
✅ Красивый вывод в консоль
✅ Сохранение всех результатов в файлы
✅ Автоматическое ранее остановление при высокой уверенности
✅ Детальное логирование токенов и вызовов
