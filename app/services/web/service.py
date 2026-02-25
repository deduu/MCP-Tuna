"""Web service — fetch URLs and perform web searches."""

import re
from typing import Any, Dict, Optional

import httpx


class WebService:
    """Provides web fetch and search operations for MCP tool consumption."""

    def __init__(self, search_api_key: Optional[str] = None):
        self.search_api_key = search_api_key

    async def fetch(self, url: str, max_length: int = 50000) -> Dict[str, Any]:
        """Fetch a URL and return its content as text.

        HTML is simplified to readable text by stripping tags.
        """
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            text = resp.text

            # Simple HTML → text conversion
            if "html" in content_type:
                text = self._html_to_text(text)

            if len(text) > max_length:
                text = text[:max_length] + f"\n\n[Truncated at {max_length} chars]"

            return {
                "success": True,
                "url": str(resp.url),
                "status": resp.status_code,
                "content_type": content_type,
                "content": text,
                "length": len(text),
            }
        except Exception as e:
            return {"success": False, "url": url, "error": str(e)}

    async def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search the web using DuckDuckGo HTML (no API key required)."""
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "Mozilla/5.0 (compatible; AgentY/1.0)"},
                )
                resp.raise_for_status()

            results = self._parse_ddg_results(resp.text, max_results)
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results),
            }
        except Exception as e:
            return {"success": False, "query": query, "error": str(e)}

    def _html_to_text(self, html: str) -> str:
        """Simple HTML to text conversion."""
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _parse_ddg_results(self, html: str, max_results: int) -> list:
        """Parse DuckDuckGo HTML search results."""
        results = []
        # Match result links and snippets
        pattern = r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</span>'
        matches = re.findall(pattern, html, re.DOTALL)

        for url, title, snippet in matches[:max_results]:
            results.append({
                "title": re.sub(r"<[^>]+>", "", title).strip(),
                "url": url,
                "snippet": re.sub(r"<[^>]+>", "", snippet).strip(),
            })
        return results
