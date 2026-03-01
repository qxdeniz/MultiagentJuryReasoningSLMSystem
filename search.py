# -*- coding: utf-8 -*-
"""Web search module with reliable sources"""

import requests
from typing import List, Dict, Optional
from config import get_config

class WebSearcher:
    """Reliable web search using multiple sources"""
    
    def __init__(self):
        self.config = get_config()["web_search"]
        self.timeout = self.config["timeout"]
    
    def search_langsearch(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search using LangSearch Web Search API.

        Falls back to a lightweight DuckDuckGo HTML scrape if LangSearch fails or
        the API key is not configured.
        """
        results: List[Dict[str, str]] = []

        api_key = get_config().get("langsearch", {}).get("api_key")
        if not api_key:
            print("⚠️ LangSearch API key not configured, falling back to DuckDuckGo")
            return self._duckduckgo_html_fallback(query, max_results)

        url = "https://api.langsearch.com/v1/web-search"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "freshness": "noLimit",
            "summary": True,
            "count": max_results,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            # Return the raw response text (JSON string) as requested
            return resp.text
        except Exception as e:
            print(f"⚠️ LangSearch request failed: {e}")
            return ""

    # NOTE: OpenRouter-specific extractor removed — LangSearch is used now.

    def _duckduckgo_html_fallback(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Lightweight HTML scrape fallback for DuckDuckGo (used when OpenRouter is unavailable)."""
        results: List[Dict[str, str]] = []
        try:
            url = "https://html.duckduckgo.com/"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            params = {"q": query}
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            import re
            snippets = re.findall(r'<a.*?href="([^"]*)".*?>([^<]*)</a>.*?<div[^>]*>([^<]*)<', response.text)
            for i, (href, title, snippet) in enumerate(snippets[:max_results]):
                if href and title and snippet:
                    results.append({
                        "title": title.strip(),
                        "url": href.strip(),
                        "body": snippet.strip(),
                        "source": "DuckDuckGo-HTML"
                    })
            return results
        except Exception as e:
            print(f"⚠️ DuckDuckGo HTML fallback also failed: {e}")
            return []

    def search(self, query: str, max_results: int = None) -> str:
        """Perform comprehensive web search using LangSearch with DuckDuckGo fallback."""
        if max_results is None:
            max_results = self.config.get("max_results", 5)

        print(f"🔍 Searching (LangSearch): {query}")
        results = self.search_langsearch(query, max_results)
        # If the langsearch method returned a raw string, return it directly
        if isinstance(results, str):
            return results

        if not results:
            return "No search results found. Please verify the query."

        formatted = []
        for i, result in enumerate(results[:max_results], 1):
            title = result.get("title", "No title")
            body = result.get("body", "No description")
            source = result.get("source", "Web")
            formatted.append(f"{i}. [{source}] {title}\n   {body[:200]}")

        return "\n\n".join(formatted)


# Global searcher instance
_searcher: Optional[WebSearcher] = None

def get_searcher() -> WebSearcher:
    """Get global searcher instance"""
    global _searcher
    if _searcher is None:
        _searcher = WebSearcher()
    return _searcher

def web_search(query: str, max_results: int = 5) -> str:
    """Perform web search"""
    searcher = get_searcher()
    return searcher.search(query, max_results)


res = web_search("машина")
print(res)

