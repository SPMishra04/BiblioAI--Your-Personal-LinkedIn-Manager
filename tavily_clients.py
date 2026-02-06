from tavily import TavilyClient
from config import TAVILY_API_KEY

tavily = TavilyClient(api_key=TAVILY_API_KEY)

def tavily_search(search_queries):
    all_results = []

    for q in search_queries:
        response = tavily.search(query=q, max_results=3)
        all_results.append({
            "query": q,
            "results": response.get("results", [])
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
            text = r.get("content", "")[:800]   # keep first 800 chars only
            url = r.get("url")
            snippets.append({"snippet": text, "url": url})

        compressed.append({
            "query": item["query"],
            "snippets": snippets
        })

    return compressed

