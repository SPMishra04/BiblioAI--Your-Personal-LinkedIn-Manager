# import ast  # ✅ FIX 1: needed to safely parse LLM list output

# from intent_analyzer import analyze_intent
# from schema import LinkedInPostOutput

# from intelligence import *
# from chains import *
# from tavily import TavilyClient
# from config import *
# from utils import collect_sufficient_input
# from llm import *


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
#             for result in response.get('results', []):
#                 results.append({
#                     'title': result.get('title', ''),
#                     'url': result.get('url', ''),
#                     'content': result.get('content', ''),
#                     'score': result.get('score', 0.0)
#                 })

#             return results

#         except Exception as e:
#             print(f"  ⚠️ Search failed for '{query}': {str(e)}")
#             return []


# def compress_search_results(search_results, max_items=5, max_chars=400):
#     """
#     Reduce search results size to fit LLM token limits
#     """
#     if not search_results:
#         return []

#     compressed = []
#     for item in search_results[:max_items]:
#         compressed.append({
#             "title": item.get("title", ""),
#             "url": item.get("url", ""),
#             "score": item.get("score", 0.0),
#             "content": item.get("content", "")[:max_chars]
#         })

#     return compressed


# tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


# def run_tavily_search_and_display(fact_checked_search_queries, max_results=5):
#     """
#     Runs Tavily search on fact-checked queries and displays
#     query, score, url, and content.
#     """

#     # ✅ FIX 2: defensive conversion if LLM returns stringified list
#     if isinstance(fact_checked_search_queries, str):
#         try:
#             fact_checked_search_queries = ast.literal_eval(fact_checked_search_queries)
#         except Exception:
#             print("⚠️ Invalid fact-checked search queries format.")
#             return []

#     all_search_outputs = []

#     for query in fact_checked_search_queries:
#         if not isinstance(query, str) or len(query.strip()) < 2:
#             continue  # ✅ guard against bad queriesj

#         print(f"\n🔍 Searching Tavily for: {query}")
#         print("-" * 80)

#         response = tavily_client.search(
#             query=query,
#             max_results=max_results
#         )

#         results = response.get("results", [])

#         for idx, result in enumerate(results, start=1):
#             output = {
#                 "search_query": query,
#                 "score": result.get("score"),
#                 "url": result.get("url"),
#                 "content": result.get("content")
#             }

#             all_search_outputs.append(output)

#             print(f"\nResult {idx}")
#             print(f"Score   : {output['score']}")
#             print(f"URL     : {output['url']}")
#             print(f"Content : {output['content'][:500]}")
#             print("-" * 80)

#     # return all_search_outputs


# # ✅ Slightly smarter factual detector
# def is_factual_question(text: str) -> bool:
#     text = text.lower()
#     return (
#         text.strip().endswith("?")
#         and any(k in text for k in ["is", "are", "does", "do", "which", "what", "compare", "difference"])
#     )


# def is_vague_question(text: str) -> bool:
#     vague_phrases = [
#         "can you fix",
#         "fix this",
#         "help me",
#         "what do you think",
#         "is this okay",
#         "can you explain",
#         "suggest something",
#         "can you help"
#     ]
#     text = text.lower()
#     return any(p in text for p in vague_phrases)


# def main():
#     print("\n--- LinkedIn AI Generator ---\n")

#     user_query = input("Enter your idea/topic: ").strip()

#     # ===============================
#     # 1️⃣ Decide clarification path
#     # ===============================
#     if is_factual_question(user_query) and not is_vague_question(user_query):
#         final_input = f"Explain this clearly with professional insight for LinkedIn:\n{user_query}"
#     else:
#         result = collect_sufficient_input(user_query, clarity_chain)

#         if result["status"] == "failed":
#             print("\n❌ Cannot proceed:", result["reason"])
#             return

#         if result["status"] == "reject":
#             print("\n❌ Rejected:", result["reason"])
#             return

#         final_input = result["final_input"]

#     # ===============================
#     # 2️⃣ Safety check loop
#     # ===============================
#     MAX_RETRIES = 3
#     attempts = 0

#     while attempts < MAX_RETRIES:
#         safety = check_restricted_content(final_input)

#         if safety.allowed:
#             break

#         print(f"\n❌ {safety.message}")
#         attempts += 1

#         if attempts < MAX_RETRIES:
#             final_input = input("🔁 Please try again with a different topic: ").strip()
#         else:
#             print("\n🚫 Too many invalid attempts. Exiting safely.")
#             return

#     print("\n✅ Accepted input:\n", final_input)

#     # ===============================
#     # 3️⃣ Intent analysis
#     # ===============================
#     intent = analyze_intent(final_input)

#     search_results = None
#     sq = None
#     fsq = []

#     if intent.websearch:
#         search_query = format_search_query(
#             user_query=final_input,
#             websearch_key=intent.websearch_key,
#             reasoning=intent.reasoning
#         )

#         sq = search_query

#         # ✅ FIX 3: parse LLM output safely
#         fact_search_queries_raw = fact_check_search_queries(
#             search_query=search_query,
#             reasoning=intent.reasoning
#         )

#         try:
#             fsq = ast.literal_eval(fact_search_queries_raw)
#         except Exception:
#             print("⚠️ Failed to parse fact-checked queries.")
#             fsq = []

#         search_service = SearchService()
#         search_results = []
#         for q in fsq:
#             search_results.extend(search_service.search(q))

#     print("\nIntent Analyzer Output:", intent)
#     print("\nSearch Query:", sq)
#     print("\nFact Checked Search Queries:", fsq)

#     sr = run_tavily_search_and_display(fsq)
#     sr

#     # ===============================
#     # 4️⃣ LinkedIn Trainer
#     # ===============================
#     trainer = linkedin_trainer(
#         user_query=final_input,
#         intent_reasoning=intent.reasoning,
#         search_results=search_results
#     )

#     print("\nLinkedIn Trainer:", trainer)

#     # ===============================
#     # 5️⃣ Pre-call payload
#     # ===============================
#     compressed_results = compress_search_results(search_results)

#     pre_call = {
#     "user_query": final_input,
#     "reasoning": intent.reasoning,
#     "body_guidance": trainer.body_guidance,
#     "word_count": trainer.word_count,
#     "search_result": compressed_results or "No external search was required.",
#     "hook_guidance": trainer.hook_guidance
# }
#     parallel_output = run_parallel_llms(pre_call)

#     hookcta_obj = parallel_output["hookcta"]
#     body_obj = parallel_output["body"]

#     hook = hookcta_obj.hook
#     cta = getattr(hookcta_obj, "cta", None) or getattr(hookcta_obj, "call_to_action", "")

#     body = body_obj.content
#     final_linkedin_post = format_post(hook, body, cta)

#     print("\n\nFinal Post:\n")
#     print(final_linkedin_post.content)


# if __name__ == "__main__":
#     main()



import ast

from intent_analyzer import analyze_intent
from schema import LinkedInPostOutput

from intelligence import (
    check_restricted_content,
    format_search_query,
    linkedin_trainer
)
from chains import *
from tavily import TavilyClient
from config import TAVILY_API_KEY
from utils import collect_sufficient_input


# --------------------------------
# Tavily Search Service
# --------------------------------
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

            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0)
                }
                for r in response.get("results", [])
            ]

        except Exception as e:
            print(f"⚠️ Tavily search failed for '{query}': {e}")
            return []


# --------------------------------
# Helpers
# --------------------------------
def compress_search_results(results, max_items=5, max_chars=400):
    if not results:
        return []

    return [
        {
            "title": r["title"],
            "url": r["url"],
            "score": r["score"],
            "content": r["content"][:max_chars]
        }
        for r in results[:max_items]
    ]


def is_factual_question(text: str) -> bool:
    text = text.lower()
    return (
        text.strip().endswith("?")
        and any(k in text for k in ["is", "are", "does", "do", "what", "which", "compare"])
    )


def is_vague_question(text: str) -> bool:
    vague_phrases = [
        "can you fix",
        "fix this",
        "help me",
        "what do you think",
        "is this okay",
        "can you explain",
        "suggest something",
        "can you help"
    ]
    return any(p in text.lower() for p in vague_phrases)


# --------------------------------
# MAIN
# --------------------------------
def main():
    print("\n--- LinkedIn AI Generator ---\n")

    user_query = input("Enter your idea/topic: ").strip()

    # ===============================
    # 1️⃣ Clarification logic
    # ===============================
    if is_factual_question(user_query) and not is_vague_question(user_query):
        final_input = f"Explain this clearly with professional insight for LinkedIn:\n{user_query}"
    else:
        result = collect_sufficient_input(user_query, clarity_chain)

        if result["status"] != "ready":
            print(f"\n❌ {result.get('reason', 'Unable to proceed')}")
            return

        final_input = result["final_input"]

    # ===============================
    # 2️⃣ Safety check
    # ===============================
    for _ in range(3):
        safety = check_restricted_content(final_input)
        if safety.allowed:
            break
        print(f"\n❌ {safety.message}")
        final_input = input("🔁 Please try again with a different topic: ").strip()
    else:
        print("\n🚫 Too many invalid attempts.")
        return

    print("\n✅ Accepted input:\n", final_input)

    # ===============================
    # 3️⃣ Intent analysis
    # ===============================
    intent = analyze_intent(final_input)

    search_results = []

    if intent.websearch:
        raw_queries = format_search_query(
            user_query=final_input,
            websearch_key=intent.websearch_key,
            reasoning=intent.reasoning
        )

        try:
            queries = ast.literal_eval(raw_queries)
        except Exception:
            queries = []

        search_service = SearchService()
        for q in queries:
            search_results.extend(search_service.search(q))

    print("\nIntent Analyzer Output:", intent)
    print("\nSearch Results Count:", len(search_results))

    # ===============================
    # 4️⃣ LinkedIn Trainer
    # ===============================
    trainer = linkedin_trainer(
        user_query=final_input,
        intent_reasoning=intent.reasoning,
        search_results=search_results
    )

    # ===============================
    # 5️⃣ Parallel generation
    # ===============================
    compressed_results = compress_search_results(search_results)

    pre_call = {
        "user_query": final_input,
        "reasoning": intent.reasoning,
        "body_guidance": trainer.body_guidance,
        "word_count": trainer.word_count,
        "search_result": compressed_results or "No external search was required.",
        "hook_guidance": trainer.hook_guidance
    }

    outputs = run_parallel_llms(pre_call)

    hook = outputs["hookcta"].hook
    cta = getattr(outputs["hookcta"], "cta", "")
    body = outputs["body"].content

    final_post = format_post(hook, body, cta)

    print("\n\nFinal Post:\n")
    print(final_post.content)


if __name__ == "__main__":
    main()
