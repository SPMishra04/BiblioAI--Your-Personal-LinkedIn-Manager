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
            print(f" Tavily search failed for '{query}': {e}")
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
            print(f"\n {result.get('reason', 'Unable to proceed')}")
            return

        final_input = result["final_input"]

    # ===============================
    # 2️⃣ Safety check
    # ===============================
    for _ in range(3):
        safety = check_restricted_content(final_input)
        if safety.allowed:
            break
        print(f"\n {safety.message}")
        final_input = input(" Please try again with a different topic: ").strip()
    else:
        print("\n Too many invalid attempts.")
        return

    print("\n Accepted input:\n", final_input)

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
