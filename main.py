from agents import run_llm1, run_llm2, run_llm3
from tavily_clients import tavily_search, compress_tavily_results
from config import MAX_CLARIFICATION_TURNS, NUM_SEARCH_QUERIES

def llm_context_check(full_text, llm1_output):
    """
    Decide whether LLM thinks we have enough clarity.
    If LLM1 is still asking a clarification question,
    we treat the input as NOT sufficient yet.
    """
    return not bool(llm1_output.get("clarification_question"))

def main():
    print("\n🚀 LinkedIn Post Generator (CLI)\n")

    user_query = input("Enter your LinkedIn topic: ")

    # ===== TEMP MEMORY FOR Q&A =====
    answers = []   # Local list-based memory

    # ========== PHASE 1: CLARIFICATION LOOP ==========
    turn = 0

    while turn < MAX_CLARIFICATION_TURNS:

        print(f"\n🔹 Running LLM1 (Turn {turn+1})...\n")

        details = "\n".join(f"- {a}" for a in answers) if answers else "None"

        combined_input = f"""
Original topic:
{user_query}

Additional details:
{details}
"""

        llm1_output = run_llm1(combined_input, NUM_SEARCH_QUERIES, answers)
        # If LLM1 rejected after rewrite, stop gracefully
        if not llm1_output.get("allowed", True):
            print("\n❌ BLOCKED AFTER REWRITE:")
            print(llm1_output.get("message"))
            return


        # -------- SAFE HANDLING (DO NOT BLOCK VAGUE INPUT) --------
        if llm1_output is None:
            print("\n❌ System error: LLM returned empty response.")
            return

        if "allowed" in llm1_output and not llm1_output["allowed"]:

            if llm1_output.get("clarification_question"):
                print("\n⚠️ Input unclear — treating as clarification case, not a block.")
            else:
                print("\n❌ BLOCKED:", llm1_output.get("message"))
                return

        clarification_q = llm1_output.get("clarification_question")

        # ---- CASE 1: READY ----
        if not clarification_q:
            print("\n✅ Input is now clear enough — proceeding.\n")
            break

        # ---- CASE 2: ASK USER ----
        print("\n🤔 I need a bit more info:")
        print("👉", clarification_q)

        user_reply = input("\nYour answer: ").strip()
        answers.append(user_reply)
        turn += 1

        full_text = user_query + "\n" + "\n".join(answers)

        if llm_context_check(full_text, llm1_output):
            print("\n✅ LLM says we have enough context now.\n")
            break

    # ===== REWRITE FINAL QUERY BASED ON ANSWERS =====
    if answers:
        print("\n🧠 Rewriting final user query based on your answers...\n")

        user_query = (
            f"Write a LinkedIn post about GenAI, "
            f"focusing on LangChain prompts, "
            f"from a learning perspective. "
            f"User clarifications: {answers}"
        )

        print("✅ Revised User Query:", user_query)

    # Clear temp memory as required
    answers.clear()

    # ========== PHASE 2: FACT CHECK DECISION AGAIN ==========
    print("\n🔁 Re-running fact-check decision...\n")
    llm1_output = run_llm1(user_query, NUM_SEARCH_QUERIES, None)

    # ===== FACT CHECK PATH =====
    if llm1_output.get("fact_check_required"):

        print("\n🌐 Running Tavily Web Search...\n")
        raw_results = tavily_search(llm1_output.get("search_queries", []))
        web_results = compress_tavily_results(raw_results)

        print("\n🔹 Running LLM2 (Fact Verification)...\n")
        llm2_output = run_llm2(user_query, web_results)

        if not llm2_output.get("is_true", False):
            print("\n❌ CLAIM REJECTED BY LLM2\n")
            print("📌 Reason:", llm2_output.get("correction_if_any"))
            print("\n🔗 Evidence:")
            for url in llm2_output.get("evidence_urls", []):
                print(" •", url)
            return

        print("\n✅ Fact Verified — Proceeding to Post Generation\n")

        llm3_output = run_llm3(
            final_query=user_query,
            source="tavily",
            tavily_context=web_results,
            verified_facts=llm2_output.get("verified_facts")
        )

    # ===== NO FACT CHECK PATH =====
    else:
        print("\n➡️ No fact check needed — using LLM1 context.\n")

        llm3_output = run_llm3(
            final_query=user_query,
            source="llm1",
            llm1_understanding=llm1_output.get("llm1_understanding")
        )

    # ===== PRINT FINAL POST (WINDOWS SAFE) =====
    print("\n===== FINAL LINKEDIN POST =====\n")
    post = llm3_output.get("formatted_post", "")
    print(post.encode("utf-8", errors="replace").decode("utf-8"))

if __name__ == "__main__":
    main()
