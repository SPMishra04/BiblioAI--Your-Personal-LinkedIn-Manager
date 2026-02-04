MAX_ATTEMPTS = 3


def llm_context_check(text: str, clarity_chain) -> bool:
    """
    Ask the LLM itself if the context is sufficient.
    No hardcoding. No keyword rules.
    """
    prompt = f"""
You are validating input for a LinkedIn post generator.

Decide ONLY one thing:
Is the following information sufficient to write a meaningful LinkedIn post?

Respond strictly in JSON:
{{ "sufficient": true/false }}

Input:
{text}
"""

    result = clarity_chain.invoke({
        "user_input": prompt
    })

    # Defensive parsing
    return bool(result.get("sufficient", False))


def collect_sufficient_input(initial_input, clarity_chain, max_turns=MAX_ATTEMPTS):
    answers = []   # ✅ SIMPLE LIST
    turn = 0

    while turn < max_turns:
        details = "\n".join(f"- {a}" for a in answers) if answers else "None"

        combined_input = f"""
Original topic:
{initial_input}

Additional details:
{details}
"""

        # Step 1: Ask LLM if clarification needed
        result = clarity_chain.invoke({
            "user_input": combined_input
        })

        status = result.get("status")

        # ✅ LLM says ready
        if status == "ready":
            return {
                "status": "ready",
                "final_input": combined_input
            }

        # ❌ Hard reject
        if status == "reject":
            return {
                "status": "reject",
                "reason": result.get("reason", "Rejected by clarity check")
            }

        # 🤔 Ask follow-up
        print("\n🤔 I need a bit more info:")
        print("👉", result.get("question", "Can you add more context?"))

        user_reply = input("\nYour answer: ").strip()
        answers.append(user_reply)
        turn += 1

        # ✅ OVERRIDE: Let LLM decide sufficiency
        full_text = initial_input + "\n" + "\n".join(answers)
        if llm_context_check(full_text, clarity_chain):
            return {
                "status": "ready",
                "final_input": combined_input
            }

    # ✅ FINAL FALLBACK — PROCEED WITH WHAT USER GAVE
    combined_input = f"""
Original topic:
{initial_input}

Additional details:
{chr(10).join(f"- {a}" for a in answers) if answers else "None"}
"""

    return {
        "status": "ready",
        "final_input": combined_input
    }
