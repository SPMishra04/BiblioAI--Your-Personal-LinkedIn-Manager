from agents import run_llm1, run_llm2, run_llm3
from tavily_clients import tavily_search, compress_tavily_results
from config import MAX_CLARIFICATION_TURNS, NUM_SEARCH_QUERIES
import tiktoken



def count_tokens(text, model="gpt-4o-mini"):
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))



def main():
    print("\nLinkedIn Post Generator (CLI)\n")

    user_query = input("Enter your LinkedIn topic: ")

    # ===== TEMP MEMORY FOR Q&A (YOUR REQUIRED FORMAT) =====
    qa_history = []   # will store: ["AI: ...", "YOU: ..."]

    # ============================
    # PHASE 1 — CLARIFICATION LOOP
    # ============================
    turn = 0

    while turn < MAX_CLARIFICATION_TURNS:

        print(f"\n🔹 Running Fact-check decision (Turn {turn+1})...\n")

        # Build combined input from memory
        context_text = "\n".join(qa_history) if qa_history else "None"

        combined_input = f"""
Original topic:
{user_query}

Conversation so far:
{context_text}
"""

        llm1_output = run_llm1(combined_input, NUM_SEARCH_QUERIES, qa_history)

        # -------- TRUE UNSAFE CONTENT ONLY --------
        if llm1_output.get("allowed") is False and not llm1_output.get("clarification_question"):
            print("\nBLOCKED:")
            print(llm1_output.get("message"))
            return

        

        # ===== CASE 1 — NEED MORE INFO (VAGUE INPUT) =====
        clarification_q = llm1_output.get("clarification_question")

# ===== CASE A — UNSAFE CONTENT (IMMEDIATE BLOCK) =====
        if llm1_output.get("allowed") is False:
            print("\nBLOCKED (UNSAFE CONTENT):")
            print(llm1_output.get("message"))
            return

        # ===== CASE B — SAFE BUT VAGUE → ASK QUESTION =====
        if clarification_q:
            print("\nInput unclear — asking for clarification.\n")
            print("->", clarification_q)

            user_reply = input("\nYour answer: ").strip()

            qa_history.append(f"AI: {clarification_q}")
            qa_history.append(f"YOU: {user_reply}")

            turn += 1
            continue

        # ===== CASE C — CLEAR AND SAFE =====
        print("\nInput is clear enough — proceeding.\n")
        break


    # ============================
    # REWRITE FINAL USER QUERY FROM HISTORY
    # ============================
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

    # Clear memory after rewrite (your requirement)
    qa_history.clear()

    # ============================
    # PHASE 2 — FACT CHECK DECISION AGAIN
    # ============================
    print("\nRe-running Gatekeeper LLM...\n")
    llm1_output = run_llm1(user_query, NUM_SEARCH_QUERIES, None)

    tokens_in_llm_request = 0

    # ===== FACT CHECK PATH =====
    if llm1_output.get("fact_check_required", False):

        print("\nRunning Tavily Web Search...\n")
        raw_results = tavily_search(llm1_output.get("search_queries", []))
        web_results = compress_tavily_results(raw_results)

        print("\nRunning fact_checker_llm (Fact Verification)...\n")
        llm2_output = run_llm2(user_query, web_results)

        tokens_in_llm_request = count_tokens(user_query)

        # ---- NEW SAFE HANDLING ----
        if not llm2_output.get("is_true", False):

            print("\nCLAIM REJECTED BY fact_checker_llm\n")

            print("LLM VERDICT & REASON:\n")
            print(llm2_output.get("correction_if_any", "No explanation provided"))

            # print("\n🔗 Evidence URLs used by LLM:\n")
            for url in llm2_output.get("evidence_urls", []):
                print(" •", url)

            return


        print("\nFact Verified — Proceeding to Post Generation\n")

        tokens_in_llm_request = count_tokens(user_query)

        llm3_output = run_llm3(
            final_query=user_query,
            source="tavily",
            tavily_context=web_results,
            verified_facts=llm2_output.get("verified_facts")
        )

    # ===== NO FACT CHECK PATH =====
    else:
        print("\nNo fact check needed — using Gatekeeper LLM.\n")

        tokens_in_llm_request = count_tokens(user_query)

        llm3_output = run_llm3(
        final_query=user_query,
        source="llm1",
        llm1_understanding=llm1_output.get("Gatekeeper LLM understanding"),
        user_intent=llm1_output.get("user_intent")
    )


    # ============================
    # PRINT FINAL POST
    # ============================
    # print("\n===== FINAL LINKEDIN POST =====\n")
    # post = llm3_output.get("formatted_post", "")
    # print(post.encode("utf-8", errors="replace").decode("utf-8"))


    print("\n===== FINAL LINKEDIN POST =====\n")

    post = llm3_output.get("formatted_post", "")

    tokens_in_final_post = count_tokens(post)

    print(f"Tokens in LLM request: {tokens_in_llm_request}")
    print(f"Tokens in final post: {tokens_in_final_post}\n")

    print(post.encode("utf-8", errors="replace").decode("utf-8"))


    # ===== NEW: CONDITIONAL REFERENCES PHASE =====
    if llm3_output.get("need_references", False):

        print("\nFetching references from Tavily...\n")

        queries = llm3_output.get("reference_queries", [])

        raw_results = tavily_search(queries)
        web_results = compress_tavily_results(raw_results)

        print("\nREFERENCES & FURTHER READING:\n")

        for item in web_results:
            print(f"Topic: {item['query']}")
            for snip in item["snippets"]:
                print(f" • {snip['url']}")
            print("\n")

        # Optional: show LLM recommendations too
        recs = llm3_output.get("recommendations", [])
        if recs:
            print("RECOMMENDATIONS:\n")
            for r in recs:
                print(" •", r)

    else:
        print("\npost_generator_llm decided no references were required.\n")


if __name__ == "__main__":
    main()




















# from agents import run_llm1, run_llm2, run_llm3
# from tavily_clients import tavily_search, compress_tavily_results
# from config import MAX_CLARIFICATION_TURNS, NUM_SEARCH_QUERIES


# def main():
#     print("\n🚀 LinkedIn Post Generator (CLI)\n")

#     user_query = input("Enter your LinkedIn topic: ").strip()

#     # ===== TEMP MEMORY FOR Q&A (AI / YOU format) =====
#     qa_history = []   # ["AI: ...", "YOU: ..."]

#     # ============================
#     # PHASE 1 — CLARIFICATION LOOP
#     # ============================
#     turn = 0

#     while turn < MAX_CLARIFICATION_TURNS:

#         print(f"\n🔹 Running LLM1 (Turn {turn+1})...\n")

#         context_text = "\n".join(qa_history) if qa_history else "None"

#         combined_input = f"""
# Original topic:
# {user_query}

# Conversation so far:
# {context_text}
# """

#         llm1_output = run_llm1(combined_input, NUM_SEARCH_QUERIES, qa_history)

#         # ---- HARD BLOCK: truly unsafe content ----
#         if llm1_output.get("allowed") is False and not llm1_output.get("clarification_question"):
#             print("\n❌ BLOCKED (UNSAFE CONTENT):")
#             print(llm1_output.get("message"))
#             return

#         clarification_q = llm1_output.get("clarification_question")

#         # ---- SAFE BUT VAGUE → ASK USER ----
#         if clarification_q:
#             print("\n⚠️ Input unclear — asking for clarification.\n")
#             print("👉", clarification_q)

#             user_reply = input("\nYour answer: ").strip()

#             qa_history.append(f"AI: {clarification_q}")
#             qa_history.append(f"YOU: {user_reply}")

#             turn += 1
#             continue

#         # ---- CLEAR AND SAFE ----
#         print("\n✅ Input is clear enough — proceeding.\n")
#         break

#     # ============================
#     # REWRITE QUERY FROM HISTORY
#     # ============================
#     if qa_history:
#         print("\n🧠 Rewriting final user query from conversation history...\n")

#         user_query = f"""
# Create a clear LinkedIn topic based on this conversation:

# Original topic:
# {user_query}

# Conversation:
# {qa_history}
# """

#         print("✅ Revised User Query:\n", user_query)

#     qa_history.clear()   # as you wanted

#     # ============================
#     # PHASE 2 — FACT CHECK DECISION
#     # ============================
#     print("\n🔁 Re-running fact-check decision...\n")
#     llm1_output = run_llm1(user_query, NUM_SEARCH_QUERIES, None)

#     if llm1_output.get("fact_check_required", False):

#         print("\n🌐 Running Tavily Web Search...\n")
#         raw_results = tavily_search(llm1_output.get("search_queries", []))
#         web_results = compress_tavily_results(raw_results)

#         print("\n🔹 Running LLM2 (Fact Verification)...\n")
#         llm2_output = run_llm2(user_query, web_results)

#         if not llm2_output.get("is_true", False):

#             print("\n❌ CLAIM REJECTED BY LLM2\n")
#             print("📌 Reason:\n", llm2_output.get("correction_if_any", "No explanation"))
#             print("\n🔗 Evidence URLs:\n")

#             for url in llm2_output.get("evidence_urls", []):
#                 print(" •", url)

#             return

#         print("\n✅ Fact Verified — Proceeding to Post Generation\n")

#         llm3_output = run_llm3(
#             final_query=user_query,
#             source="tavily",
#             tavily_context=web_results,
#             verified_facts=llm2_output.get("verified_facts")
#         )

#     else:
#         print("\n➡️ No fact check needed — using LLM1 context.\n")

#         llm3_output = run_llm3(
#             final_query=user_query,
#             source="llm1",
#             llm1_understanding=llm1_output.get("llm1_understanding")
#         )

#     # ============================
#     # PRINT FINAL POST
#     # ============================
#     print("\n===== FINAL LINKEDIN POST =====\n")
#     post = llm3_output.get("formatted_post", "")
#     print(post.encode("utf-8", errors="replace").decode("utf-8"))

#     # ============================
#     # CONDITIONAL REFERENCES PHASE
#     # ============================
#     if llm3_output.get("needs_references", False):

#         print("\n🔎 Fetching references from Tavily...\n")

#         topic = llm3_output.get("reference_topic")
#         raw_results = tavily_search([topic])
#         web_results = compress_tavily_results(raw_results)

#         print("\n📚 REFERENCES & FURTHER READING:\n")

#         for item in web_results:
#             print(f"🔹 Topic: {item['query']}")
#             for snip in item["snippets"][:3]:
#                 print(f" • {snip['url']}")
#             print()

#     else:
#         print("\nℹ️ LLM decided no references were required.\n")


# if __name__ == "__main__":
#     main()
