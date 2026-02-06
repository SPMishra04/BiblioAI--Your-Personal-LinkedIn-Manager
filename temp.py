from tavily import TavilyClient
from memory import memory_store
from config import GROQ_API_KEY, MODEL_NAME, TAVILY_API_KEY
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=0
)

tavily = TavilyClient(api_key=TAVILY_API_KEY)

def main():
    user_id = "user_1"

    print("\n--- LinkedIn AI Generator (Single LLM Call) ---\n")
    user_query = input("Enter your idea/topic: ")

    if user_query.lower() == "reset":
        memory_store.clear_user_memory(user_id)
        print("✅ Memory cleared.")
        return

    # -------- MEMORY --------
    past_memories = memory_store.get_relevant_memory(
        user_id=user_id,
        query=user_query,
        k=5
    )
    memory_context = "\n".join(past_memories) if past_memories else ""

    # -------- HEURISTIC INTENT (NO LLM) --------
    websearch_needed = any(
        word in user_query.lower()
        for word in ["latest", "news", "update", "trend", "report"]
    )

    web_results = ""
    if websearch_needed:
        response = tavily.search(
            query=user_query,
            max_results=5,
            search_depth="advanced"
        )
        web_results = response.get("results", [])

    # -------- SINGLE PROMPT --------
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a expert LinkedIn content creator .
You specialize in writing high-performing LinkedIn posts that feel:
- Human, thoughtful, and experience-driven
- Professional but conversational
- Insightful rather than promotional

You MUST perform ALL tasks below in ONE response.

━━━━━━━━━━━━━━
CORE RESPONSIBILITIES
━━━━━━━━━━━━━━
1. Internally analyze the user's intent and context.
2. Decide whether the post is:
   - Educational
   - Opinion-based
   - Experience-based
   - Story-driven
   - Trend or insight-led
3. Maintain narrative clarity and reader flow.

━━━━━━━━━━━━━━
MEMORY & CONTEXT RULES (CRITICAL)
━━━━━━━━━━━━━━
- "Past Context" represents the user's REAL previous experiences, learnings, or posts.
- If the user query implies:
  - continuation (e.g., "day 2", "following up", "next part", "as mentioned earlier")
  - reflection on past experience
  - iteration on previous ideas  
  → You MUST use Past Context accurately.

- NEVER ask the user to repeat known information.
- NEVER contradict Past Context.
- Maintain logical and timeline continuity.
- If Past Context is irrelevant to the query, safely ignore it.

━━━━━━━━━━━━━━
WEB SEARCH USAGE
━━━━━━━━━━━━━━
- Use Web Search Results ONLY if provided.
- Use them as factual support or grounding.
- Do NOT invent statistics, studies, or claims.
- Do NOT mention sources explicitly unless naturally required.

━━━━━━━━━━━━━━
CONTENT GENERATION TASK
━━━━━━━━━━━━━━
More Important **Write ONE complete, publish-ready LinkedIn post based on the user query.Include hook,body,cta in a single post **

1️⃣ HOOK  
- 1–2 short lines
- Scroll-stopping and curiosity-driven
- Professional, not clickbait
- No hashtags
- No emojis overload

2️⃣ BODY  
- Value-driven and insight-rich
- Written for LinkedIn professionals
- Short paragraphs (1–3 lines)
- Clear logic and smooth transitions
- Can include bullet points ONLY if they improve clarity
- No hashtags
- No CTA here

3️⃣ CTA  
- Exactly 1 line
- Soft and thoughtful (invite engagement)
- Encourages reflection, discussion, or sharing
- Not sales-oriented
- Must feel natural to the post

━━━━━━━━━━━━━━
FINAL FORMATTING RULES
━━━━━━━━━━━━━━
- Output must be PLAIN TEXT only
- NO markdown
- NO JSON
- NO explanations
- Add light, tasteful emojis (sparingly)
- Add a maximum of 5 relevant hashtags at the end
- The post must look ready to publish on LinkedIn

━━━━━━━━━━━━━━
FINAL OUTPUT STRUCTURE (MANDATORY)
━━━━━━━━━━━━━━
Hook

Body (spaced paragraphs)

CTA

Hashtags
"""
        ),
        (
            "user",
            """
User Query:
{user_query}

Past Context:
{memory_context}

Web Search Results:
{web_results}
"""
        )
    ]
)


    final_post = llm.invoke(
        prompt.format(
            user_query=user_query,
            memory_context=memory_context,
            web_results=web_results
        )
    )

    print("\n---------------- FINAL POST ----------------\n")
    print(final_post.content)

    # -------- MEMORY WRITE BACK --------
    memory_store.add_memory(
        user_id=user_id,
        text=final_post.content,
        memory_type="linkedin_post"
    )

if __name__ == "__main__":
    main()
