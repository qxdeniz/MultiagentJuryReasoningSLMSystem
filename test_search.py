# -*- coding: utf-8 -*-
import unittest
from unittest.mock import patch, MagicMock
import config
from search import WebSearcher


class TestLangSearch(unittest.TestCase):
    def setUp(self):
        # Ensure a langsearch key is present to avoid fallback
        config.CONFIG.setdefault("langsearch", {})["api_key"] = "testkey"

    @patch("requests.post")
    def test_search_langsearch_basic(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "results": [
                {
                    "title": "Apple ESG 2024 highlights",
                    "url": "https://apple.com/esg",
                    "snippet": "Key points about Apple's 2024 ESG report..."
                }
            ]
        }
        mock_post.return_value = mock_resp

        ws = WebSearcher()
        out = ws.search("tell me the highlights from Apple's 2024 ESG report", max_results=1)

        self.assertIn("Apple ESG 2024 highlights", out)
        self.assertIn("https://apple.com/esg", out)
        self.assertIn("LangSearch", out)


if __name__ == "__main__":
    unittest.main()
