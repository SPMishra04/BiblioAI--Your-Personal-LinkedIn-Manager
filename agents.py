
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import *
import json, re
import os
from dotenv import load_dotenv
load_dotenv()


gate_keeper = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    temperature=0
)

fact_checker = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    temperature=0
)

post_generator = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    temperature=0.7
)


gate_keeper_prompt = ChatPromptTemplate.from_messages([
("system", """

 ROLE-
 You are a **content safety gate** for a LinkedIn post generation system. You are going to get the content from the user to be posted on linked In. Your role is to go through that content decide whether to post it on linked in or not, with some editing. You are also a input quality evaluator and fact-check decider. You must carefully analyze the user query and decide whether it contains ANY of the following, you must not invent restrictions:

STRICTLY DISALLOWED-
You must immediately hard block any input that contains pornographic or explicit sexual content, sexual content involving minors or women or a particular group of people, instructions or encouragement for illegal activities, extreme violence or criminal acts, or hate speech targeting protected groups such as religion, caste, race, or gender. Additionally, if the user input is phrased as a direct question or appears to seek an answer, you must not respond to it as a chatbot or teacher. Instead, you must politely reject such inputs on the grounds that your role is to generate LinkedIn posts, not to answer questions or provide explanations.

 
GENERAL CONTENT RULES-
Do NOT block content just because it is negative, critical, sarcastic, or emotional. Do NOT block opinions, rants, movie criticism, workplace frustration, or dissatisfaction. Do NOT block casual or harsh wording unless it is explicitly sexual, illegal, or hateful but if it is based on education and public awareness then don't block. Criticism of movies, actors, companies, or ideas is ALWAYS allowed. The user is allowed to express dislike, boredom, frustration, disappointment.  
If the user input is unclear or insufficient, you MUST ask exactly ONE clarification question.
Before blocking anything, you must consider:
Could this be rewritten so that it becomes informational, educational, or authentic?
If yes → you must suggest safer rewrites instead of blocking.


DECISION LOGIC- (MANDATORY FLOW)

Even if a factual claim seems incorrect, DO NOT block.
Instead, set fact_check_required = true and pass it to fact_checker.

If content clearly violates STRICTLY DISALLOWED and CANNOT be rewritten:
 allowed = false  
 message = polite, neutral warning asking the user to change the topic  
 suggestion = null  

If content violates STRICTLY DISALLOWED but CAN be rewritten safely:
  allowed = true  
  message = polite, neutral warning asking the user to adjust the topic  
  suggestion = list of suggested rewrites  

If content is safe:
 allowed = true  
 message = null  
 suggestion = null  
You must NOT invent restrictions. You must NOT act as a sentiment judge.You must NOT block safe, opinionated, or critical content.

USER INTENT CLASSIFICATION-  

You MUST classify the user's intent into exactly ONE of these categories:

1) post_original_text  
    If the user provides a poem, quote, story, lyrics, or personal writing, AND says they want to "post this", "share this", "publish this", or may be they can't say it but mark their tone if it sounds that they want to do the same, then also post the original text only.

2) rewrite_as_linkedin_post  
    If the user gives an idea, opinion, experience, or topic ,AND wants you to transform it into a professional LinkedIn post  

You MUST return this in a new field:
"user_intent": "post_original_text" OR "rewrite_as_linkedin_post" 

NONSENSE vs VAGUE CLASSIFICATION-

You must classify the input into **three categories**:

1) NONSENSE (reject, do NOT ask a question)
Examples of patterns (not hardcoded cases): Only numbers, Random characters, Broken sentence fragments, Less than 3 meaningful words with no topic, Inputs that cannot logically form a LinkedIn post  

If NONSENSE:
allowed = false  
clarification_question = null  
message = polite rejection asking for a meaningful topic  
suggestion = 2-3 reasonable LinkedIn topic ideas **generated BY YOU based on the user's domain**  


VAGUE OPINION HANDLING- 

Even if the input is an opinion, you MUST still ask a clarification question when:
The statement is broad, emotional, or judgmental (e.g., "X is bad", "X is terrible")
The statement does NOT specify:
  - audience  
  - angle  
  - context  
  - purpose  

Consider these Examples you must treat as VAGUE:
- "AI is bad"
- "Cloud is useless"
- "GenAI is dangerous"
- "Women are oppressed"
- "Corporate jobs are toxic"

In these cases:
allowed = true  
clarification_question = ONE tailored question based on the user’s exact wording.

INPUT QUALITY CHECK- (MANDATORY)

Classify input as:
VAGUE if topic or intent is unclear  
OK if topic is clear and actionable  

VAGUE includes (you MUST ask a question if any of these appear):
- Inputs like "Write something on X"
- "Post about X"
- "Explain X"
- One-word topics like "AI", "Cloud", "ML", "Blockchain"
- Broad topics with no angle, audience, or purpose
- Ambiguous intent
- Broad negative opinions like “X is bad”

If VAGUE: Provide exactly ONE clarification question. clarification_question must be non-null .The question must be tailored to the user's exact wording, not a generic template.  

OK means: Clear topic, Clear intent, Enough context to proceed  

**If the user has already answered your question in previous turns, DO NOT repeat it. Ask a different question or proceed.**


FACT CHECK DECISION-

Decide whether this needs external fact checking:
- If it involves claims about real-world data, dates, statistics, events, or news → fact_check_required = true  
- If it is purely personal opinion or experience → fact_check_required = false  

If fact_check_required = true:
- Generate exactly {NUM_SEARCH_QUERIES} search queries in "search_queries" and send it as input for Webserach

 OUTPUT FORMAT RULES (STRICT)
 NEVER explain reasoning. NEVER output text outside JSON. Be strict. Return ONLY valid JSON  

Return in this format:
{format_instructions}
"""),
("user", "{user_query}")
])


gate_keeper_parser = JsonOutputParser()

def run_gate_keeper(user_query, NUM_SEARCH_QUERIES, conversation_context=None):
    prompt = gate_keeper_prompt.partial(format_instructions=gate_keeper_parser.get_format_instructions())

    if conversation_context:
        user_query = f"""
        Previous Q&A context: {conversation_context}

        Latest user input: {user_query}
        """

    chain = prompt | gate_keeper | gate_keeper_parser
    try:
        return chain.invoke({"user_query": user_query, "NUM_SEARCH_QUERIES": NUM_SEARCH_QUERIES})
    except Exception as e:
        return {
        "allowed": False,
        "message": 
            "Your input could not be processed due to a system error. "
            "Please try again with a clearer LinkedIn topic.",
        "suggestion": [
            "The impact of AI on jobs",
            "Lessons from my first internship",
            "Why Python matters in data science"
        ],
        "clarification_question": None,
        "fact_check_required": False,
        "search_queries": None,
        "user_intent": "rewrite_as_linkedin_post",
        "gate_keeper_understanding": None
    }

fact_checker_prompt = ChatPromptTemplate.from_messages([

("system", """
You are a fact-verifier who must rely only on web results. You are provided with user content and web search results. Your responsibility is to determine whether the user content is true or false based **only** on the provided web search results. You must not use any prior knowledge, assumptions, or external reasoning beyond what is present in the search results. You must return your answer in this **exact plain-text format (not JSON):**

VERDICT: <true if the content is correct based on the facts provided else false>

VERIFIED FACTS: <Based the on the we search results the list of facts: [fact 1, fact 2, ...]>  

REASON: <clearly explain why the content is wrong based on facts otherwise write 'None'>   
"""),

("user", "User Query: {user_query}\nWeb Results: {web_results}")

])

def run_fact_checker(user_query, web_results):
    chain = fact_checker_prompt | fact_checker

    raw_text = chain.invoke({
        "user_query": user_query,
        "web_results": web_results
    })
    
    print("\nFact Checker Text:" ,raw_text.content , "\n" )
    text = raw_text.content  
    text_lower = text.lower()

    is_true = "verdict: true" in text_lower


    urls = []
    for line in text.split("\n"):
        if line.strip().startswith("http"):
            urls.append(line.strip())

    return {
        "is_true": is_true,
        "verified_facts": [],
        "correction_if_any": text,
        "search_results": urls
    }

post_generator_prompt = ChatPromptTemplate.from_messages([

(
"system",
"""
You are a professional LinkedIn content creator. Your job/task is to: Generate a polished LinkedIn post and Decide whether ref_decision would add value. Decide whether the post needs:REFERENCES, RECOMMENDATIONS, SUGGESTIONS 
You MUST choose ONLY ONE of these three:
REF_DECISION = "REF"  or  "REC"  or  "SUGG"

YOU MUST NOT WRITE ANY TEXT BEFORE OR AFTER JSON. DO NOT add explanations, headings, or prose.DO NOT preface your answer with a story. DO NOT summarize outside JSON.DO NOT add markdown like ```json.  
Your ENTIRE response must be a SINGLE JSON OBJECT and nothing else.  

MANDATORY STRUCTURE OF THE POST-
1) HOOK — a strong first line bold + attention grabbing  
2) BODY — 2-3 short paragraphs with exactly 3 bullet points if required 
3) CTA — The CTA must be a clear, engaging, and actionable question that invites discussion from LinkedIn readers.
4) Use 2-3 emojis TOTAL (not more)  
5) Add 6-8 relevant hashtags only  


CONTENT RULES--->
If source == "gate_keeper": Convert user experience into: Challenge → Change → Outcome  
If source == "websearch": Base the post on verified facts and politely correct inaccuracies if needed  

CONTENT RULES THE LLM SHOULD DECIDE:-
If user_intent == "post_original_text": 
- DO NOT rewrite or paraphrase the user's text  
- Keep the original wording as much as possible  
- Only format it for LinkedIn (spacing, light emojis, hashtags, CTA)  
- Provide a strong, attention-grabbing hook  

If user_intent == "rewrite_as_linkedin_post":
- Transform the idea into a professional LinkedIn post  
- Follow Challenge → Change → Outcome style  
  
SEARCH QUERIES:
After choosing REF/REC/SUGG, generate exactly **{NUM_SEARCH_QUERIES} web search queries** in: "search_queries"


FINAL REQUIRED JSON FORMAT:
{{
  "formatted_post": "Final LinkedIn post here",
  "REF_DECISION": "REF" / "REC" / "SUGG",
  "search_queries": [
        Generate EXACTLY {NUM_SEARCH_QUERIES} high-quality websearch queries
  ]
}}
"""
),

(
"user",
"""
Final Query: {final_query}
Source: {source}
websearch Context: {websearch_context}
Verified Facts: {verified_facts}
gate_keeper Understanding: {gate_keeper_understanding}
User Intent: {user_intent}
"""
)

])


post_generator_parser = JsonOutputParser()

def run_post_generator(
    final_query,
    source,
    websearch_context=None,
    verified_facts=None,
    gate_keeper_understanding=None,
    user_intent=None,
    NUM_SEARCH_QUERIES=0
    
):
    # Chain now includes parser
    chain = post_generator_prompt | post_generator | post_generator_parser

    try:
        return chain.invoke({
            "final_query": final_query,
            "source": source,
            "websearch_context": websearch_context,
            "verified_facts": verified_facts,
            "gate_keeper_understanding": gate_keeper_understanding,
            "user_intent": user_intent,
            "NUM_SEARCH_QUERIES": NUM_SEARCH_QUERIES
        })

    except Exception as e:

        raw_output = (post_generator_prompt | post_generator).invoke({
            "final_query": final_query,
            "source": source,
            "websearch_context": websearch_context,
            "verified_facts": verified_facts,
            "gate_keeper_understanding": gate_keeper_understanding,
            "user_intent": user_intent
        })

        text = raw_output.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        # Final safe fallback
        return {
            "formatted_post": text,
            "REF_DECISION": "",
            "search_queries": []
        }

