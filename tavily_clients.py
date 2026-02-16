import os
from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()


# ==========================================================
# ENHANCED TAVILY SEARCH SERVICE (REFERENCE + CONTEXT READY)
# ==========================================================

class SearchService:
    """
    Central Tavily client that now:
    - Fetches normal snippets
    - Fetches raw content for recommendation/suggestion use
    - Is reusable across the whole project
    """

    def __init__(self):
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def search(self, query: str):
        try:
            response = self.client.search(
                query=query,
                max_results=2,              
                search_depth="advanced",
                include_raw_content=True   
            )
            # max_results means it Limits the number of search results returned per queryMeaning:
            #max_results=1 → only top 1 result, max_results=5 → top 5 results

            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "raw_content": result.get("raw_content", ""), 
                    "score": result.get("score", 0.0)
                })

            return results

        except Exception as e:
            print(f"  Search failed for '{query}': {str(e)}")
            return []


# Create one shared Tavily client for the whole app
search_service = SearchService()

# ==========================================================
# KEEP YOUR EXISTING FUNCTION NAMES (FOR COMPATIBILITY)
# ==========================================================

def tavily_search(search_queries):
    """
    Wrapper around SearchService so your pipeline does NOT break.
    Returns structure compatible with your current code.
    """

    all_results = []

    for q in search_queries:
        results = search_service.search(q)

        all_results.append({
            "query": q,
            "results": results
        })

    return all_results


def compress_tavily_results(web_results, max_chars=2000):
    """
    Now returns:
    - short snippets (for display)
    - urls (for references)
    - full_context (for LLM recommendations/suggestions)

    This enables:
    - cleaner references
    - smarter recommendations
    - contextual LLM reasoning
    """

    compressed = []


    for item in web_results:
        snippets = []

        for r in item["results"]:
            snippets.append({
                "title": r.get("title", "Untitled Source"),
                "snippet": (r.get("content") or "")[:800],
                "url": r.get("url")
            })

        compressed.append({
            "query": item["query"],
            "snippets": snippets
        })

    return compressed
