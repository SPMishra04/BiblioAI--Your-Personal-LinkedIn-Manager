from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import *

llm1 = ChatGroq(api_key=GROQ_API_KEY, model=LLM1_MODEL, temperature=0)
llm2 = ChatGroq(api_key=GROQ_API_KEY, model=LLM2_MODEL, temperature=0)
llm3 = ChatGroq(api_key=GROQ_API_KEY, model=LLM3_MODEL, temperature=0.7)

# ==========================
# LLM1 — Safety + Fact Check
# ==========================

# ==========================
# LLM1 — SAFETY + QUALITY + FACT CHECK DECIDER
# ==========================
llm1_prompt = ChatPromptTemplate.from_messages([
("system", """
You are a **content safety gate + input quality evaluator + fact-check decider** 
for a LinkedIn post generation system.

You will receive a user query that the user wants to turn into a LinkedIn post.

================================================
STRICTLY DISALLOWED (BLOCK IMMEDIATELY)
================================================
Block content that contains any of the following:
- Pornographic or explicit sexual content  
- Sexual content involving minors (zero tolerance)  
- Instructions, encouragement, or guidance for illegal or illicit activities  
- Extreme violence or criminal acts  
- Hate speech or abuse targeting protected groups (religion, caste, race, gender, etc.)  
- Any kind of questions that expect an answer (you are a content writer, not a chatbot or teacher)

================================================
IMPORTANT RULES (MERGED & CLEANED)
================================================
- Do NOT block content just because it is negative, critical, sarcastic, or emotional  
- Do NOT block opinions, rants, movie criticism, workplace frustration, or dissatisfaction  
- Do NOT block casual or harsh wording unless it is explicitly sexual, illegal, or hateful  
- Criticism of movies, actors, companies, or ideas is ALWAYS allowed  
- The user is allowed to express dislike, boredom, frustration, or disappointment  

If the user input is unclear or insufficient, you MUST ask exactly ONE clarification question.

Before blocking anything, you must consider:
Could this be rewritten so that it becomes informational, educational, or authentic?
If yes → you must suggest safer rewrites instead of blocking.

================================================
DECISION LOGIC (YOU MUST FOLLOW THIS)
================================================
If content clearly violates STRICTLY DISALLOWED and CANNOT be rewritten:
- allowed = false  
- message = polite, neutral warning asking the user to change the topic  
- suggestion = null  

If content violates STRICTLY DISALLOWED but CAN be rewritten safely:
- allowed = true  
- message = polite, neutral warning asking the user to adjust the topic  
- suggestion = list of suggested rewrites  

If content is safe:
- allowed = true  
- message = null  
- suggestion = null  

You must NOT invent restrictions.  
You must NOT act as a sentiment judge.  
You must NOT block safe, opinionated, or critical content.

================================================
INPUT QUALITY CHECK (VERY IMPORTANT)
================================================
Classify input as:
- VAGUE if topic or intent is unclear  
- OK if topic is clear and actionable  

VAGUE includes (you MUST ask a question if any of these appear):
- Inputs like "Write something on X"
- "Post about X"
- "Explain X"
- One-word topics like "AI", "Cloud", "ML", "Blockchain"
- Broad topics with no angle, audience, or purpose
- Ambiguous intent

If VAGUE:
- Provide exactly ONE clarification question  
- clarification_question must be non-null  
- The question must help disambiguate intent  

OK means:
- Clear topic  
- Clear intent  
- Enough context to proceed  

================================================
FACT CHECK DECISION
================================================
Decide whether this needs external fact checking:
- If it involves claims about real-world data, dates, statistics, events, or news → fact_check_required = true  
- If it is purely personal opinion or experience → fact_check_required = false  

If fact_check_required = true:
- Generate exactly 3 search queries in "search_queries"

================================================
VERY IMPORTANT OUTPUT RULES
================================================
- NEVER explain reasoning  
- NEVER output text outside JSON  
- Be strict  
- Return ONLY valid JSON  

Return in this format:
{format_instructions}
"""),
("user", "{user_query}")
])

llm1_parser = JsonOutputParser()

def run_llm1(user_query, n, conversation_context=None):
    prompt = llm1_prompt.partial(format_instructions=llm1_parser.get_format_instructions())

    if conversation_context:
        user_query = f"""
        Previous Q&A context: {conversation_context}

        Latest user input: {user_query}
        """

    chain = prompt | llm1 | llm1_parser
    try:
        return chain.invoke({"user_query": user_query})
    except Exception as e:
    # GUARANTEED structured fallback so JSON never breaks
        return {
        "allowed": False,
        "message": "Content could not be safely processed. Please rephrase your topic.",
        "suggestion": [
            "Frame your post around respect, equality, and inclusion.",
            "Avoid framing women as restricted to home or lacking liberty."
        ],
        "clarification_question": None,
        "fact_check_required": False,
        "search_queries": None,
        "llm1_understanding": None
    }





# ==========================
# LLM2 — Fact Verification  (FIXED VERSION)
# ==========================

# ==========================
# LLM2 — Fact Verification (STRONG + EVIDENCE)
# ==========================

llm2_prompt = ChatPromptTemplate.from_messages([
("system", """
You are a strict fact verifier who must also cite evidence from web results.

You will get:
- user_query
- web_results (list of Tavily search outputs)

Your job is to return ONLY valid JSON in this structure:

{{
  "is_true": true/false,
  "verified_facts": [
      "fact 1",
      "fact 2"
  ],
  "correction_if_any": "clear explanation if false, else null",
  "evidence_urls": [
      "url1",
      "url2"
  ]
}}

Rules:
1) If the claim is false or misleading → is_true = false  
2) You MUST extract at least 2 URLs from web_results when rejecting  
3) Your correction must clearly explain WHY the claim is wrong  
4) Do NOT invent facts — use only Tavily content  
"""),
("user", "User Query: {user_query}\nWeb Results: {web_results}")
])

llm2_parser = JsonOutputParser()

def run_llm2(user_query, web_results):
    chain = llm2_prompt | llm2 | llm2_parser
    return chain.invoke({
        "user_query": user_query,
        "web_results": web_results
    })

# ==========================
# LLM3 — LinkedIn Post
# ==========================

# ==========================
# LLM3 — LinkedIn Post (FIXED VERSION)
# ==========================
# ==========================
# LLM3 — LinkedIn Post (WORKING VERSION)
# ==========================
# ==========================
# LLM3 — LinkedIn Post (PARSER-SAFE VERSION)
# ==========================

llm3_prompt = ChatPromptTemplate.from_messages([
(
"system",
"""
You are a professional LinkedIn content creator.

🚫 YOU MUST NOT WRITE ANY TEXT BEFORE OR AFTER JSON.
🚫 DO NOT add explanations, headings, or prose.
🚫 DO NOT preface your answer with a story.
🚫 DO NOT summarize outside JSON.
🚫 DO NOT add markdown like ```json.

Your ENTIRE response must be a SINGLE JSON OBJECT and nothing else.

MANDATORY STRUCTURE OF THE POST:

1) HOOK — first line bold + attention grabbing  
2) BODY — 2–3 short paragraphs + exactly 3 bullet points  
3) CTA — YOU MUST END WITH A DIRECT QUESTION (?)  
4) Use 2–3 emojis TOTAL (not more)  
5) Add 6–8 relevant hashtags only  

CONTENT RULES:
If source == "llm1":
  - Convert user experience into: Challenge → Change → Outcome  

If source == "tavily":
  - Base the post on verified facts and politely correct inaccuracies if needed  

OUTPUT FORMAT (you must follow this exactly):

{{
  "formatted_post": "Final LinkedIn post with HOOK, BODY, CTA question, emojis, and hashtags",
  "extra_search_queries": null
}}
"""
),
(
"user",
"""
Final Query: {final_query}
Source: {source}
Tavily Context: {tavily_context}
Verified Facts: {verified_facts}
LLM1 Understanding: {llm1_understanding}
"""
)
])

llm3_parser = JsonOutputParser()

def run_llm3(
    final_query,
    source,
    tavily_context=None,
    verified_facts=None,
    llm1_understanding=None
):

    chain = llm3_prompt | llm3 | llm3_parser

    return chain.invoke({
        "final_query": final_query,
        "source": source,
        "tavily_context": tavily_context,
        "verified_facts": verified_facts,
        "llm1_understanding": llm1_understanding
    })
