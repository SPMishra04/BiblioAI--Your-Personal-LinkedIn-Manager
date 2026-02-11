from agents import *
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

        gate_keeper_output = run_gate_keeper(combined_input, NUM_SEARCH_QUERIES, qa_history)

        # -------- TRUE UNSAFE CONTENT ONLY --------
        
        
        if gate_keeper_output.get("allowed") is False:
            print("\nBLOCKED (UNSAFE CONTENT):")
            # print(gate_keeper_output.get("message"))
            print(gate_keeper_output)
            return

        

        # ===== CASE 1 — NEED MORE INFO (VAGUE INPUT) =====
        clarification_q = gate_keeper_output.get("clarification_question")

# ===== CASE A — UNSAFE CONTENT (IMMEDIATE BLOCK) =====
        if gate_keeper_output.get("allowed") is False and not gate_keeper_output.get("clarification_question"):
            print("\nBLOCKED:")
            # print(gate_keeper_output.get("message"))
            print(gate_keeper_output)
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
    gate_keeper_output = run_gate_keeper(user_query, NUM_SEARCH_QUERIES, None)

    tokens_in_llm_request = 0

    # ===== FACT CHECK PATH =====
    fact_checker_output = {}
    if gate_keeper_output.get("fact_check_required", False):

        print("\nRunning Tavily Web Search...\n")
        raw_results = tavily_search(gate_keeper_output.get("search_queries", []))
        web_results = compress_tavily_results(raw_results)

        print("\nRunning fact_checker_llm (Fact Verification)...\n")
        fact_checker_output = run_fact_checker(user_query, web_results)

        tokens_in_llm_request = count_tokens(user_query)

        # ---- NEW SAFE HANDLING ----
        if not fact_checker_output.get("is_true", False):

            print("\nCLAIM REJECTED BY fact_checker_llm\n")

            print("LLM VERDICT & REASON:\n")
            print(fact_checker_output.get("correction_if_any", "No explanation provided"))

            # print("\n🔗 Evidence URLs used by LLM:\n")
            for url in fact_checker_output.get("evidence_urls", []):
                print(" •", url)

            return


        print("\nFact Verified — Proceeding to Post Generation\n")

        tokens_in_llm_request = count_tokens(user_query)

        post_generator_output = run_post_generator(
            final_query=user_query,
            source="tavily",
            tavily_context=web_results,
            verified_facts=fact_checker_output.get("verified_facts")
        )

    # ===== NO FACT CHECK PATH =====
    else:
        print("\nNo fact check needed — using Gatekeeper LLM.\n")

        tokens_in_llm_request = count_tokens(user_query)

        post_generator_output = run_post_generator(
        final_query=user_query,
        source="gate_keeper",
        gate_keeper_understanding=gate_keeper_output.get("Gatekeeper LLM understanding"),
        user_intent=gate_keeper_output.get("user_intent")
    )

    
    print("\nGate_Keeper Output:" , gate_keeper_output , "\n")
    print("\nFact_Checker Output" , fact_checker_output , "\n")


    print("\n===== FINAL LINKEDIN POST =====\n")

    post = post_generator_output.get("formatted_post", "")

    tokens_in_final_post = count_tokens(post)

    print(f"Tokens in LLM request: {tokens_in_llm_request}")
    print(f"Tokens in final post: {tokens_in_final_post}\n")

    print(post.encode("utf-8", errors="replace").decode("utf-8"))

        # ===== NEW: CONDITIONAL OUTPUT PHASE =====

    show_refs = post_generator_output.get("show_references", False)
    show_recs = post_generator_output.get("show_recommendations", False)
    show_sugs = post_generator_output.get("show_suggestions", False)

    # ---------------- CASE 1: REFERENCES ----------------
    if show_refs:

        print("\n🔎 Fetching references from Tavily...\n")

        queries = post_generator_output.get("reference_queries", [])

        raw_results = tavily_search(queries)
        web_results = compress_tavily_results(raw_results)

        print("\n📚 REFERENCES & FURTHER READING:\n")

        for item in web_results:
            print(f"🔹 Topic: {item['query']}")
            for snip in item["snippets"]:
                print(f" • {snip['url']}")
            print("\n")

    # ---------------- CASE 2: RECOMMENDATIONS ----------------
    elif show_recs:

        recs = post_generator_output.get("recommendations", [])

        print("\n💡 RECOMMENDATIONS:\n")
        for r in recs:
            print(" •", r)

    # ---------------- CASE 3: SUGGESTIONS ----------------
    elif show_sugs:

        sugs = post_generator_output.get("suggestions", [])

        print("\n🧠 ALTERNATIVE TOPIC SUGGESTIONS:\n")
        for s in sugs:
            print(" •", s)

    # ---------------- CASE 4: NOTHING NEEDED ----------------
    else:
        print("\nℹ️ LLM decided no references, recommendations, or suggestions were required.\n")



#     print("\n===== FINAL LINKEDIN POST =====\n")

#     post = post_generator_output.get("formatted_post", "")

#     tokens_in_final_post = count_tokens(post)

#     print(f"Tokens in LLM request: {tokens_in_llm_request}")
#     print(f"Tokens in final post: {tokens_in_final_post}\n")

#     print(post.encode("utf-8", errors="replace").decode("utf-8"))


#     # ===== NEW: CONDITIONAL REFERENCES PHASE =====
#     if post_generator_output.get("need_references", False):

#         print("\nFetching references from Tavily...\n")

#         queries = post_generator_output.get("reference_queries", [])

#         raw_results = tavily_search(queries)
#         web_results = compress_tavily_results(raw_results)

#         print("\nREFERENCES & FURTHER READING:\n")

#         for item in web_results:
#             print(f"Topic: {item['query']}")
#             for snip in item["snippets"]:
#                 print(f" • {snip['url']}")
#             print("\n")

#         # Optional: show LLM recommendations too
#         recs = post_generator_output.get("recommendations", [])
#         if recs:
#             print("RECOMMENDATIONS:\n")
#             for r in recs:
#                 print(" •", r)

#     else:
#         print("\npost_generator_llm decided no references were required.\n")


# if __name__ == "__main__":
#     main()




















# from agents import gate_keeper, fact_checker, post_generator
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

#         print(f"\n🔹 Running gate_keeper (Turn {turn+1})...\n")

#         context_text = "\n".join(qa_history) if qa_history else "None"

#         combined_input = f"""
# Original topic:
# {user_query}

# Conversation so far:
# {context_text}
# """

#         gate_keeper_output = gate_keeper(combined_input, NUM_SEARCH_QUERIES, qa_history)

#         # ---- HARD BLOCK: truly unsafe content ----
#         if gate_keeper_output.get("allowed") is False and not gate_keeper_output.get("clarification_question"):
#             print("\n❌ BLOCKED (UNSAFE CONTENT):")
#             print(gate_keeper_output.get("message"))
#             return

#         clarification_q = gate_keeper_output.get("clarification_question")

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
#     gate_keeper_output = gate_keeper(user_query, NUM_SEARCH_QUERIES, None)

#     if gate_keeper_output.get("fact_check_required", False):

#         print("\n🌐 Running Tavily Web Search...\n")
#         raw_results = tavily_search(gate_keeper_output.get("search_queries", []))
#         web_results = compress_tavily_results(raw_results)

#         print("\n🔹 Running fact_checker (Fact Verification)...\n")
#         fact_checker_output = fact_checker(user_query, web_results)

#         if not fact_checker_output.get("is_true", False):

#             print("\n❌ CLAIM REJECTED BY fact_checker\n")
#             print("📌 Reason:\n", fact_checker_output.get("correction_if_any", "No explanation"))
#             print("\n🔗 Evidence URLs:\n")

#             for url in fact_checker_output.get("evidence_urls", []):
#                 print(" •", url)

#             return

#         print("\n✅ Fact Verified — Proceeding to Post Generation\n")

#         post_generator_output = post_generator(
#             final_query=user_query,
#             source="tavily",
#             tavily_context=web_results,
#             verified_facts=fact_checker_output.get("verified_facts")
#         )

#     else:
#         print("\n➡️ No fact check needed — using gate_keeper context.\n")

#         post_generator_output = post_generator(
#             final_query=user_query,
#             source="gate_keeper",
#             gate_keeper_understanding=gate_keeper_output.get("gate_keeper_understanding")
#         )

#     # ============================
#     # PRINT FINAL POST
#     # ============================
#     print("\n===== FINAL LINKEDIN POST =====\n")
#     post = post_generator_output.get("formatted_post", "")
#     print(post.encode("utf-8", errors="replace").decode("utf-8"))

#     # ============================
#     # CONDITIONAL REFERENCES PHASE
#     # ============================
#     if post_generator_output.get("needs_references", False):

#         print("\n🔎 Fetching references from Tavily...\n")

#         topic = post_generator_output.get("reference_topic")
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


if __name__ == "__main__":
    main()
