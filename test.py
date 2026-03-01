#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick test script to verify the system works"""

from main import run_system

# Test topics
topics = [
    "Что такое искусственный интеллект и какие его основные применения?",
    "Решить уравнение: 3x - 5 = 10"
]

if __name__ == "__main__":
    print("\n🚀 Запуск Multi-Agent Verification System...\n")
    
    # Run with Yandex (default)
    try:
        run_system(topics)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\n💡 Проверьте:")
        print("1. Установлены ли все зависимости?")
        print("2. Доступны ли credentials Yandex?")
        print("3. Или попробуйте с OpenRouter:")
        print("\n   run_system(topics, provider='openrouter', model='meta-llama/llama-2-7b-chat')")
