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
 
STRICTLY DISALLOWED — QUESTION HANDLING (CRITICAL FIX)
If the user input is phrased as a DIRECT QUESTION (starts with:
"why", "how", "what", "is", "can", "should", "when", "where", "does", "do"),

you MUST NOT try to answer it as a post.

Instead, you MUST:
allowed = false  
clarification_question = null  
message = polite rejection like:

"I can’t answer questions directly because I generate LinkedIn posts, not explanations.  
If you want a post on this topic, please rephrase as a statement instead."

You MUST also provide 2–3 safer LinkedIn topic suggestions.

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
USER INTENT CLASSIFICATION (NEW — VERY IMPORTANT)
================================================
You MUST classify the user's intent into exactly ONE of these categories:

1) post_original_text  
   - If the user provides a poem, quote, story, lyrics, or personal writing  
   - AND says they want to "post this", "share this", or "publish this"

2) rewrite_as_linkedin_post  
   - If the user gives an idea, opinion, experience, or topic  
   - AND wants you to transform it into a professional LinkedIn post  

3) explain_or_discuss  
   - If the user is asking a question, requesting explanation, or discussion  

You MUST return this in a new field:
"user_intent": "post_original_text" OR "rewrite_as_linkedin_post" OR "explain_or_discuss"

 

================================================
NEW RULE — NONSENSE vs VAGUE (GENERALIZED)
================================================

You must classify the input into **three categories**:

1️NONSENSE (reject, do NOT ask a question)
Examples of patterns (not hardcoded cases):
- Only numbers  
- Random characters  
- Broken sentence fragments  
- Less than 3 meaningful words with no topic  
- Inputs that cannot logically form a LinkedIn post  

If NONSENSE:
allowed = false  
clarification_question = null  
message = polite rejection asking for a meaningful topic  
suggestion = 2–3 reasonable LinkedIn topic ideas **generated BY YOU based on the user’s domain**  

================================================
 NEW RULE — FIX FOR YOUR BUG 
================================================
Even if the input is an opinion, you MUST still ask a clarification question when:

- The statement is broad, emotional, or judgmental (e.g., "X is bad", "X is terrible")
- The statement does NOT specify:
  • audience  
  • angle  
  • context  
  • purpose  

Examples you must treat as VAGUE:
- "AI is bad"
- "Cloud is useless"
- "GenAI is dangerous"
- "Women are oppressed"
- "Corporate jobs are toxic"

In these cases:
allowed = true  
clarification_question = ONE tailored question based on the user’s exact wording.

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
- Broad negative opinions like “X is bad”

If VAGUE:
- Provide exactly ONE clarification question  
- clarification_question must be non-null  
- The question must be tailored to the user's exact wording, not a generic template.  

OK means:
- Clear topic  
- Clear intent  
- Enough context to proceed  
 
** If the user has already answered your question in previous turns,
DO NOT repeat it. Ask a different question or proceed.

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
        return {
        "allowed": False,
        "message": 
            "I can’t process this as a LinkedIn post because it looks like a question or unclear input. "
            "Please rephrase as a clear topic you want to post about.",
        "suggestion": [
            "The impact of AI on jobs",
            "Lessons from my first internship",
            "Why Python matters in data science"
        ],
        "clarification_question": None,
        "fact_check_required": False,
        "search_queries": None,
        "user_intent": "rewrite_as_linkedin_post",
        "llm1_understanding": None
    }




llm2_prompt = ChatPromptTemplate.from_messages([
("system", """
You are a strict fact verifier using Tavily results.

You must decide if the claim is true or false based ONLY on web_results.

Return your answer in this EXACT text format (not JSON):

VERDICT: true or false

VERIFIED FACTS:
- fact 1
- fact 2

CORRECTION (if false):
<explain clearly why claim is wrong OR write 'None'>

EVIDENCE URLS:
- url1
- url2
"""),
("user", "User Query: {user_query}\nWeb Results: {web_results}")
])

def run_llm2(user_query, web_results):
    chain = llm2_prompt | llm2

    raw_text = chain.invoke({
        "user_query": user_query,
        "web_results": web_results
    })

    text = raw_text.content  # <-- IMPORTANT FIX

    # -------- SAFE PARSING IN PYTHON (NO JSON PARSER) --------

    is_true = "VERDICT: true" in text.lower()

    urls = []
    for line in text.split("\n"):
        if line.strip().startswith("http"):
            urls.append(line.strip())

    return {
        "is_true": is_true,
        "verified_facts": [],
        "correction_if_any": text,
        "evidence_urls": urls
    }

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

# ==========================================================
# LLM3 — LINKEDIN POST + CONDITIONAL REFERENCES (PRODUCTION SAFE)
# ==========================================================

llm3_prompt = ChatPromptTemplate.from_messages([
(
"system",
"""
You are a professional LinkedIn content creator.Your job is to:
1)generate a polished LinkedIn post  
2) Decide whether external references would add value  

 YOU MUST NOT WRITE ANY TEXT BEFORE OR AFTER JSON.
 DO NOT add explanations, headings, or prose.
 DO NOT preface your answer with a story.
 DO NOT summarize outside JSON.
 DO NOT add markdown like ```json.

Your ENTIRE response must be a SINGLE JSON OBJECT and nothing else.

================================================
MANDATORY STRUCTURE OF THE POST
================================================

1) HOOK — a strong first line bold + attention grabbing  
2) BODY — 2–3 short paragraphs + exactly 3 bullet points  
3) CTA — YOU MUST END WITH A DIRECT QUESTION (?)  
4) Use 2–3 emojis TOTAL (not more)  
5) Add 6–8 relevant hashtags only  

================================================
CONTENT RULES
================================================

If source == "llm1":
  - Convert user experience into: Challenge → Change → Outcome  

If source == "tavily":
  - Base the post on verified facts and politely correct inaccuracies if needed  

  
CONTENT RULES (CRITICAL — LET THE LLM DECIDE)

If user_intent == "post_original_text":
  - DO NOT rewrite or paraphrase the user's text  
  - Keep the original wording as much as possible  
  - Only format it for LinkedIn (spacing, light emojis, hashtags, CTA)  
  - A strong hook should be given which must be attention grabbing

If user_intent == "rewrite_as_linkedin_post":
  - Transform the idea into a professional LinkedIn post  
  - Follow Challenge → Change → Outcome style  

    
DECISION RULE (VERY IMPORTANT):
After writing the post, decide:

- needs_references = true IF:
  - Topic involves careers, learning, research, policy, industry trends, or future planning  
  - OR the reader would benefit from trusted sources  

- needs_references = false IF:
  - Pure personal story, poem, or reflection  
  - Creative writing  
  - Simple opinion  
  

If needs_references = true, also provide:
reference_topic = a short topic Tavily can search, e.g.
  - "AI careers for graduates"
  - "AI workplace productivity"
  - "AI job market trends"


================================================
DECISION RULE FOR REFERENCES (CRITICAL)
================================================

You must decide whether external references are needed.

Set:
  "need_references": true ONLY IF:
   - The post contains real-world facts, statistics, trends, dates, or claims
   - OR the topic is educational, technical, or controversial
   - OR the user intent suggests learning, research, or evidence

Set:
  "need_references": false IF:
   - It is purely personal opinion
   - It is motivational / reflective / storytelling
   - It is emotional or subjective
   - It is generic leadership or inspiration

If need_references = true:
  - Generate exactly 3 high-quality Tavily search queries in:
      "reference_queries"

If need_references = false:
  - Set "reference_queries": []

================================================
OPTIONAL RECOMMENDATIONS
================================================

You may add 1–3 helpful recommendations ONLY if they add real value.
Otherwise set:
  "recommendations": []

================================================
FINAL REQUIRED JSON FORMAT (YOU MUST FOLLOW)
================================================

{{
  "formatted_post": "Final LinkedIn post with HOOK, BODY, CTA question, emojis, and hashtags",
  "need_references": true/false,
  "reference_queries": [
      "query 1"
  ],
  "recommendations": [
      "Optional suggestion 1",
      "Optional suggestion 2"
  ]
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
User Intent: {user_intent}
"""
)
])

llm3_parser = JsonOutputParser()

def run_llm3(
    final_query,
    source,
    tavily_context=None,
    verified_facts=None,
    llm1_understanding=None,
    user_intent=None      # ✅ <-- comma added
):

    chain = llm3_prompt | llm3 | llm3_parser

    return chain.invoke({
        "final_query": final_query,
        "source": source,
        "tavily_context": tavily_context,
        "verified_facts": verified_facts,
        "llm1_understanding": llm1_understanding,
        "user_intent": user_intent          # ✅ <-- now actually passed to LLM
    })
