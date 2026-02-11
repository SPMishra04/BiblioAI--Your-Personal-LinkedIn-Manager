# from tavily import TavilyClient
# from config import TAVILY_API_KEY

# from tavily import TavilyClient
# from config import TAVILY_API_KEY


# class SearchService:
#     def __init__(self):
#         self.client = TavilyClient(api_key=TAVILY_API_KEY)

#     def search(self, query: str):
#         try:
#             response = self.client.search(
#                 query=query,
#                 max_results=5,
#                 search_depth="advanced",
#                 include_raw_content=False
#             )

#             results = []
#             for result in response.get("results", []):
#                 results.append({
#                     "title": result.get("title", ""),
#                     "url": result.get("url", ""),
#                     "content": result.get("content", ""),
#                     "score": result.get("score", 0.0)
#                 })

#             return results

#         except Exception as e:
#             print(f"  ⚠️ Search failed for '{query}': {str(e)}")
#             return []


# # Create shared client
# search_service = SearchService()


# def tavily_search(search_queries):
#     if not search_queries:
#         return []      # <-- prevents crashes

#     all_results = []


#     for q in search_queries:
#         results = search_service.search(q)

#         all_results.append({
#             "query": q,
#             "results": results
#         })

#     return all_results


# def compress_tavily_results(web_results, max_chars=4000):
#     """
#     Trim Tavily results to avoid token overflow.
#     Keeps only short snippets + URLs.
#     """
#     compressed = []

#     for item in web_results:
#         snippets = []

#         for r in item["results"]:
#             text = (r.get("content") or "")[:800]  # safe slice
#             url = r.get("url")

#             snippets.append({
#                 "snippet": text,
#                 "url": url
#             })

#         compressed.append({
#             "query": item["query"],
#             "snippets": snippets
#         })

#     return compressed




from tavily import TavilyClient
from config import TAVILY_API_KEY

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
        self.client = TavilyClient(api_key=TAVILY_API_KEY)

    def search(self, query: str):
        try:
            response = self.client.search(
                query=query,
                max_results=5,
                search_depth="advanced",
                include_raw_content=True   # <-- IMPORTANT (for LLM recs later)
            )

            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "raw_content": result.get("raw_content", ""),  # NEW FIELD
                    "score": result.get("score", 0.0)
                })

            return results

        except Exception as e:
            print(f"  ⚠️ Search failed for '{query}': {str(e)}")
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
        combined_text = ""

        for r in item["results"]:
            text = (r.get("content") or "")[:800]      # short snippet
            raw = (r.get("raw_content") or "")[:1200] # richer context

            combined_text += raw + "\n\n"

            snippets.append({
                "snippet": text,
                "url": r.get("url")
            })

        compressed.append({
            "query": item["query"],
            "snippets": snippets,
            "full_context": combined_text[:max_chars]  # NEW FIELD
        })

    return compressed
