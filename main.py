import os
import tiktoken
from agents import *
from tavily_clients import tavily_search, compress_tavily_results
from config import (
    MAX_CLARIFICATION_TURNS,
    NUM_SEARCH_QUERIES,
    MAX_LINKS_PER_TOPIC_FACTCHECK,
    MAX_LINKS_PER_TOPIC_NORMAL
)


os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY", "")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT", "")
os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")


def extract_urls(web_results):
    urls = []
    for item in web_results:
        for snippet in item.get("snippets", []):
            url = snippet.get("url")
            if url:
                urls.append(url)
    return urls


def count_tokens(text, model="gpt-4o-mini"):
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))


def main():
    print("\nLinkedIn Post Generator (CLI)\n")

    user_query = input("Enter your LinkedIn topic: ")

    # ===== TEMP MEMORY FOR Q&A =====
    qa_history = []
    turn = 0

    while turn < MAX_CLARIFICATION_TURNS:

        print(f"\n🔹 Running Fact-check decision (Turn {turn+1})...\n")

        context_text = "\n".join(qa_history) if qa_history else "None"

        combined_input = f"""
Original topic:
{user_query}

Conversation so far:
{context_text}
"""

        gate_keeper_output = run_gate_keeper(combined_input, NUM_SEARCH_QUERIES, qa_history)

        # =====================================================
        # BLOCK HANDLING WITH ONE-TIME SUGGESTION RETRY
        # =====================================================
        if gate_keeper_output.get("allowed") is False:

            print("\n BLOCK TRIGGERED")
            print("Reason:", gate_keeper_output.get("message"))

            suggestions = gate_keeper_output.get("suggestion")

            if suggestions:
                print("\n Suggestions available for correction.")
                print("You get ONE chance to select a valid alternative.\n")

                for idx, s in enumerate(suggestions, 1):
                    print(f"{idx}. {s}")

                user_choice = input("\nEnter the number of your choice (or press Enter to cancel): ").strip()

                if user_choice.isdigit() and 1 <= int(user_choice) <= len(suggestions):

                    selected_topic = suggestions[int(user_choice) - 1]
                    print(f"\n You selected: {selected_topic}")
                    print(" Re-running LLM with selected suggestion...\n")

                    gate_keeper_output["suggestion"] = None
                    gate_keeper_output = run_gate_keeper(selected_topic, NUM_SEARCH_QUERIES, None)

                    if gate_keeper_output.get("allowed") is False:
                        print("\n Still blocked after correction. Exiting.\n")
                        return

                    user_query = selected_topic

                else:
                    print("\n Invalid or no selection made.")
                    print(" Blocking permanently.\n")
                    gate_keeper_output["suggestion"] = None
                    return

            else:
                print("\n No suggestions available. Blocking immediately.\n")
                return

        clarification_q = gate_keeper_output.get("clarification_question")

        # ===== CASE B — SAFE BUT VAGUE → ASK QUESTION =====
        if clarification_q:
            print("\nInput unclear — asking for clarification.\n")
            print("->", clarification_q)

            user_reply = input("\nYour answer: ").strip()

            qa_history.append(f"AI: {clarification_q}")
            qa_history.append(f"YOU: {user_reply}")

            turn += 1
            continue

        print("\nInput is clear enough — proceeding.\n")
        break

    # ================= REWRITE FINAL QUERY =================
    if qa_history:
        print("\nRewriting final user query from conversation history...\n")

        user_query = f"""
Create a clear LinkedIn topic based on this conversation:

Original topic:
{user_query}

Conversation:
{qa_history}
"""
        print("Revised User Query:\n", user_query)

    qa_history.clear()

    # ================= PHASE 2 — FACT CHECK =================
    print("\nRe-running Gatekeeper LLM...\n")
    gate_keeper_output = run_gate_keeper(user_query, NUM_SEARCH_QUERIES, None)

    tokens_in_llm_request = 0
    fact_checker_output = {}
    web_results = None

    if gate_keeper_output.get("fact_check_required", False):

        print("\nRunning Tavily Web Search...\n")
        raw_results = tavily_search(gate_keeper_output.get("search_queries", []))
        web_results = compress_tavily_results(raw_results)

        print("\nRunning fact_checker_llm (Fact Verification)...\n")
        fact_checker_output = run_fact_checker(user_query, web_results)

        tokens_in_llm_request = count_tokens(user_query)

        if not fact_checker_output.get("is_true", False):

            print("\nCLAIM REJECTED BY fact_checker_llm\n")
            print("LLM VERDICT & REASON:\n")
            print(fact_checker_output.get("correction_if_any", "No explanation provided"))

            print("\nReferences:\n")
            seen_urls = set()

            for item in web_results:
                for snip in item.get("snippets", []):
                    title = snip.get("title", "Untitled Source")
                    url = snip.get("url")

                    if url and url not in seen_urls:
                        print(f"• {title}")
                        print(f"  {url}")
                        seen_urls.add(url)

            return

        print("\nFact Verified — Proceeding to Post Generation\n")

        post_generator_output = run_post_generator(
            final_query=user_query,
            source="tavily",
            websearch_context=web_results,
            verified_facts=fact_checker_output.get("verified_facts"),
            NUM_SEARCH_QUERIES=NUM_SEARCH_QUERIES
        )

    else:
        print("\nNo fact check needed — using Gatekeeper LLM.\n")

        tokens_in_llm_request = count_tokens(user_query)

        post_generator_output = run_post_generator(
            final_query=user_query,
            source="gate_keeper",
            gate_keeper_understanding=gate_keeper_output.get("Gatekeeper LLM understanding"),
            user_intent=gate_keeper_output.get("user_intent")
        )

    # ================= FINAL OUTPUT =================
    output_text = []

    output_text.append("\n===== FINAL LINKEDIN POST =====\n")

    post = post_generator_output.get("formatted_post", "")
    tokens_in_final_post = count_tokens(post)

    output_text.append(f"Tokens in LLM request: {tokens_in_llm_request}")
    output_text.append(f"Tokens in final post: {tokens_in_final_post}\n")
    output_text.append(post)

    decision = post_generator_output.get("REF_DECISION", "").upper()
    queries = post_generator_output.get("search_queries", [])

    # ===== CASE 1: FACT-CHECK URLs =====
    if web_results:

        if decision == "REF":
            header = " REFERENCES & EVIDENCE:"
        elif decision == "REC":
            header = " RECOMMENDATIONS WITH SOURCES:"
        elif decision == "SUGG":
            header = " SUGGESTIONS WITH SOURCES:"
        else:
            header = " RELATED RESOURCES:"

        output_text.append(f"\n{header}\n")

        seen_urls = set()

        count = 0
        for item in web_results:
            
            for snip in item.get("snippets", []):
                title = snip.get("title", "Untitled Source")
                url = snip.get("url")

                if url and url not in seen_urls:
                    output_text.append(f" • {title}")
                    output_text.append(f"   {url}")
                    seen_urls.add(url)
                    count += 1

                if count == MAX_LINKS_PER_TOPIC_FACTCHECK:
                    break

            if count == MAX_LINKS_PER_TOPIC_FACTCHECK:
                break

            output_text.append("")

    # ===== CASE 2: NORMAL TAVILY FLOW =====
    elif decision and queries:

        raw_results = tavily_search(queries)
        tavily_results = compress_tavily_results(raw_results)

        if decision == "REF":
            header = " REFERENCES & EVIDENCE:"
        elif decision == "REC":
            header = " RECOMMENDATIONS WITH SOURCES:"
        elif decision == "SUGG":
            header = " SUGGESTIONS WITH SOURCES:"
        else:
            header = " RELATED RESOURCES:"

        output_text.append(f"\n{header}\n")

        seen_urls = set()


        count = 0
        for item in tavily_results:
            for snip in item.get("snippets", []):
                title = snip.get("title", "Untitled Source")
                url = snip.get("url")

                if url and url not in seen_urls:
                    output_text.append(f" • {title}")
                    output_text.append(f"   {url}")
                    seen_urls.add(url)
                    count += 1

                if count == MAX_LINKS_PER_TOPIC_NORMAL:
                    break

            if count == MAX_LINKS_PER_TOPIC_NORMAL:
                break

            output_text.append("")

    else:
        output_text.append("\n No references/recommendations/suggestions required.\n")

    print("\n".join(output_text))


if __name__ == "__main__":
    main()
