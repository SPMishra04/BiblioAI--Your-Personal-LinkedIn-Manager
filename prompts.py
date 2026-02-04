from langchain.prompts import ChatPromptTemplate

clarity_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert content strategist. "
        "Your job is to decide whether the user input is sufficient to generate a professional LinkedIn post."
    ),
    (
        "user",
        """
User Input:
{user_input}

Task:
- Decide if the information is sufficient to generate a LinkedIn post.
- If insufficient, ask ONE clear follow-up question.
- If sufficient, mark it ready.
- If content is disallowed or meaningless, reject.

Respond ONLY in JSON with this schema:
{{
  "status": "clarify | ready | reject",
  "reason": "short explanation",
  "question": "follow-up question or null"
}}
"""
    )
])
