from tavily import TavilyClient
from config import TAVILY_API_KEY

from tavily import TavilyClient
from config import TAVILY_API_KEY

# ==============================
# NEW SEARCH SERVICE (YOUR STYLE)
# ==============================

class SearchService:
    def __init__(self):
        self.client = TavilyClient(api_key=TAVILY_API_KEY)

    def search(self, query: str):
        try:
            response = self.client.search(
                query=query,
                max_results=5,
                search_depth="advanced",
                include_raw_content=False
            )

            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0)
                })

            return results

        except Exception as e:
            print(f"  ⚠️ Search failed for '{query}': {str(e)}")
            return []


# Create shared client
search_service = SearchService()

# ==========================================================
# KEEP YOUR EXISTING FUNCTION NAMES (for compatibility)
# ==========================================================

def tavily_search(search_queries):
    """
    Wrapper around SearchService so your pipeline doesn't break.
    """
    all_results = []

    for q in search_queries:
        results = search_service.search(q)

        all_results.append({
            "query": q,
            "results": results
        })

    return all_results


def compress_tavily_results(web_results, max_chars=4000):
    """
    Trim Tavily results to avoid token overflow.
    Keeps only short snippets + URLs.
    """
    compressed = []

    for item in web_results:
        snippets = []

        for r in item["results"]:
            text = (r.get("content") or "")[:800]  # safe slice
            url = r.get("url")

            snippets.append({
                "snippet": text,
                "url": url
            })

        compressed.append({
            "query": item["query"],
            "snippets": snippets
        })

    return compressed
