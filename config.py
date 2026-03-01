# -*- coding: utf-8 -*-
"""Configuration module for Multi-Agent Verification System"""

import os
from typing import Dict, Any

# Default configuration
CONFIG = {
    # Model configuration
    "model_provider": "openrouter",  # "yandex" or "openrouter"
    
    # Yandex Cloud settings
    "yandex": {
        "folder_id": os.getenv("YANDEX_FOLDER_ID"),
        "auth_token": os.getenv("YANDEX_AUTH"),
        "model": "yandexgpt",
        "temperature": 0.6,
    },
    
    # OpenRouter settings (free models)
    "openrouter": {
        "model": "google/gemma-3n-e4b-it:free",  # Free model
       # "model": "z-ai/glm-4.5-air:free",
        # Other free models:
        # - "mistralai/mistral-7b-instruct"
        # - "meta-llama/llama-2-7b-chat"
        # - "gryphe/mythomist-7b"
        "temperature": 0.6,
    },

    # LangSearch settings (web search)
    "langsearch": {
        # Prefer storing a real key in the environment variable LANGSEARCH_API_KEY
    },
    
    # System configuration
    "use_jury": True,
    "use_librarian": True,
    "max_iterations": 3,  # Reduced for faster execution
    "target_confidence": 0.85,
    "use_rl_reward": True,
    
    # Web search configuration
    "web_search": {
        "enabled": True,
        "max_results": 5,
        "timeout": 10,
    },

    # Rate limiting / retry defaults for external HTTP calls (seconds)
    "rate_limit": {
        "delay": 0.5,           # default pause after each LLM call
        "max_retries": 4,       # retry attempts for transient errors
        "backoff_factor": 2.0,  # exponential backoff multiplier
    },
    
    # Output configuration
    "output": {
        "save_txt": True,
        "save_json": True,
        "txt_filename": "results.txt",
        "json_filename": "conversation_log.json",
    },
}

def get_config() -> Dict[str, Any]:
    """Get current configuration"""
    return CONFIG

def update_config(key: str, value: Any):
    """Update configuration"""
    if key in CONFIG:
        CONFIG[key] = value
    else:
        print(f"Warning: Config key '{key}' not found")

def set_model_provider(provider: str, model: str = None):
    """Change LLM provider and optionally model"""
    if provider not in ["yandex", "openrouter"]:
        raise ValueError(f"Unknown provider: {provider}")
    CONFIG["model_provider"] = provider
    if model and provider in CONFIG:
        CONFIG[provider]["model"] = model
