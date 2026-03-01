# -*- coding: utf-8 -*-
"""LLM integration module - supports Yandex Cloud and OpenRouter"""

import requests
import json
import time
from typing import Optional, Dict, Any
from config import get_config

class LLMProvider:
    """Base LLM provider interface"""
    
    def __call__(self, prompt: str) -> str:
        raise NotImplementedError


class YandexLLM(LLMProvider):
    """Yandex Cloud LLM provider"""
    
    def __init__(self):
        try:
            from yandex_cloud_ml_sdk import YCloudML
            self.YCloudML = YCloudML
            self.client = None
        except ImportError:
            raise ImportError("yandex-cloud-ml-sdk not installed")
    
    def __call__(self, prompt: str) -> str:
        config = get_config()
        yandex_cfg = config["yandex"]
        
        try:
            sdk = self.YCloudML(
                folder_id=yandex_cfg["folder_id"],
                auth=yandex_cfg["auth_token"],
            )
            model = sdk.models.completions(yandex_cfg["model"])
            model = model.configure(temperature=yandex_cfg["temperature"])
            result = model.run(prompt)
            answer = result.alternatives[0].text
            answer = answer.replace("*", "")
            return answer
        except Exception as e:
            # Fallback with error message
            print(f"❌ Yandex LLM Error: {e}")
            raise


class OpenRouterLLM(LLMProvider):
    """OpenRouter LLM provider (supports free models)"""
    
    def __init__(self):
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def __call__(self, prompt: str) -> str:
        config = get_config()
        or_cfg = config["openrouter"]
        
        if not or_cfg["api_key"]:
            raise ValueError("OPENROUTER_API_KEY not set. Set environment variable or update config.")
        
        headers = {
            "Authorization": f"Bearer {or_cfg['api_key']}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": or_cfg["model"],
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": or_cfg["temperature"],
        }
        
        # Retry with exponential backoff on 429 / 5xx / transient network errors
        rl_cfg = config.get("rate_limit", {})
        max_retries = int(rl_cfg.get("max_retries", 4))
        backoff = float(rl_cfg.get("backoff_factor", 2.0))
        base_delay = float(rl_cfg.get("delay", 0.5))

        attempt = 0
        last_exc = None
        while attempt <= max_retries:
            try:
                response = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
                # Handle explicit 429 / 5xx with backoff
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    wait = base_delay * (backoff ** attempt)
                    print(f"⚠️ OpenRouter returned {response.status_code}. Backing off {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                    attempt += 1
                    continue

                response.raise_for_status()
                data = response.json()

                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                else:
                    raise ValueError(f"Unexpected response format: {data}")

            except requests.exceptions.RequestException as e:
                last_exc = e
                # network-level error — backoff and retry
                if attempt >= max_retries:
                    print(f"❌ OpenRouter Error (final): {e}")
                    raise
                wait = base_delay * (backoff ** attempt)
                print(f"⚠️ OpenRouter request error: {e}. Retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                attempt += 1

        # If we exit loop without returning, raise the last exception
        if last_exc:
            raise last_exc
        raise RuntimeError("OpenRouter request failed after retries")


class LLMFactory:
    """Factory for creating LLM providers"""
    
    providers = {
        "yandex": YandexLLM,
        "openrouter": OpenRouterLLM,
    }
    
    @staticmethod
    def create(provider_name: str = None) -> LLMProvider:
        """Create LLM provider by name"""
        if provider_name is None:
            provider_name = get_config()["model_provider"]
        
        if provider_name not in LLMFactory.providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        return LLMFactory.providers[provider_name]()


# Global LLM instance
_llm_instance: Optional[LLMProvider] = None

def get_llm() -> LLMProvider:
    """Get global LLM instance"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMFactory.create()
    return _llm_instance

def set_llm_provider(provider_name: str):
    """Set LLM provider and reset instance"""
    global _llm_instance
    _llm_instance = None
    from config import set_model_provider
    set_model_provider(provider_name)

def call_llm(prompt: str) -> str:
    """Call LLM with current provider"""
    llm = get_llm()
    result = llm(prompt)

    # Small pause between LLM calls to avoid rate limits
    try:
        rl_delay = float(get_config().get("rate_limit", {}).get("delay", 0))
        if rl_delay and rl_delay > 0:
            time.sleep(rl_delay)
    except Exception:
        pass

    return result
