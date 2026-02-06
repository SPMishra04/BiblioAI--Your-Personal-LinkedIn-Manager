from intent_analyzer import analyze_intent
from chains import  extract_facts, CTA_Hook_prompt
from memory import memory_store
from tavily import TavilyClient
from config import GROQ_API_KEY, MODEL_NAME, TAVILY_API_KEY
from chains import *


class SearchService:
    def __init__(self):
        self.client = TavilyClient(api_key=TAVILY_API_KEY)

    def search(self, query: str, max_results: int = 5):
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
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
            print(f"⚠️ Tavily search failed: {e}")
            return []

def main():
    user_id = "user_1"
    search_service = SearchService()

    print("\n--- LinkedIn AI Generator ---\n")
    user_query = input("Enter your idea/topic: ")

    # Reset memory
    if user_query.lower() == "reset":
        memory_store.clear_user_memory(user_id)
        print("✅ Memory cleared. Start fresh.")
        return
    
    past_memories = memory_store.get_relevant_memory(
    user_id=user_id,
    query=user_query,
    k=7
    )

    memory_context = "\n".join(past_memories) if past_memories else ""
    # print("\n[DEBUG] Retrieved Memory:")
    # print(f"\nMemory Content::{memory_context}\n" if memory_context else "No past memory")

    intent_result = analyze_intent(user_query)
    print("\n[DEBUG] Intent Analyzer Output:")
    print(intent_result)

    
    if intent_result.websearch:
        search_results = search_service.search(user_query)

        pre_call = {
            "user_query": user_query,
            "reasoning": intent_result.reasoning,
            "websearch_results": search_results,
            "memory_context": memory_context   # REQUIRED for prompt
        }
        web_facts = extract_facts(search_results)
        memory_store.add_memory(
        user_id=user_id,
        text=web_facts,
        memory_type="web_facts"
    )
        

        # print("\n[DEBUG] Websearch used. Payload =", pre_call)

    else:
        pre_call = {
            "user_query": user_query,
            "reasoning": intent_result.reasoning,
            "websearch_results": "",  # NEVER None for prompts
            "memory_context": memory_context      # REQUIRED for prompt
        }
        # print("\n[DEBUG] Websearch NOT used. Payload =", pre_call)

    parallel_output = run_parallel_llms(pre_call)

    hook = parallel_output["hook"]
    body = parallel_output["body"]
    print("\n[DEBUG] Hook Output:\n", hook)
    print("\n[DEBUG] Body Output:\n", body)

    final_post = run_formatting(hook, body)

    print("----"*20)
    print(f"\nfinal_post: \n{final_post.content}")

    facts = extract_facts(user_query)
    memory_store.add_memory(user_id=user_id, text=facts)

    print("\n[DEBUG] Memory saved successfully")

    facts = extract_facts(body)

    memory_store.add_memory(
    user_id=user_id,
    text=facts,
    memory_type="facts"
)
    

if __name__ == "__main__":
    main()
