from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel
from config import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from schema import *
from prompts import clarity_prompt
from llm import groq_llm



clarity_parser = JsonOutputParser()

clarity_chain = (
    clarity_prompt
    | groq_llm
    | clarity_parser
)



output_parser = PydanticOutputParser(
    pydantic_object=LinkedInPostOutput
)

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE
)


parser_hookcta = PydanticOutputParser(pydantic_object=HookCTA)
parser_body = PydanticOutputParser(pydantic_object=Body)

CTA_Hook_prompt = ChatPromptTemplate.from_messages([
    ("system",
"""
You are an expert LinkedIn content strategist and copywriter.

Your task is to generate:
1. A high-impact HOOK (first 1–2 lines)
2. A strong CALL TO ACTION (CTA)

PRIMARY INSTRUCTION:
You MUST follow the provided Hook Guidance when crafting the hook.
If Hook Guidance is present, it takes priority over generic hook styles.

Follow these rules strictly:
- Write specifically for LinkedIn professionals
- Hooks must be scroll-stopping, curiosity-driven, and human
- Use conversational, authentic language
- Avoid marketing jargon and emoji overload
- Maintain a professional but relatable tone
- CTAs should encourage meaningful engagement (comment, reflect, share, save)
- Never sell or promote products
- If output does not match the required JSON schema, retry internally
-IMPORTANT:
- Use EXACT field names:
  - hook
  - cta
- Do NOT use "call_to_action"


Hook Guidelines:
- Max 2 lines
- Must align with the Hook Guidance
- Can be:
  • A bold insight
  • A reflective question
  • A lived experience
  • A pattern break
- Must trigger “see more” curiosity

CTA Guidelines:
- Exactly 1 line
- Soft invitation, not forceful
- Must NOT repeat the hook
- Must connect logically to the hook + topic

Do NOT:
- Use hashtags
- Repeat phrases mechanically
- Use filler CTAs like “Let me know your thoughts”
- Mention instructions or reasoning

Return ONLY valid JSON matching the required schema.
"""),
("user",
"""
User Query: {user_query}

Reasoning Context:
{reasoning}

Hook Guidance (STRICT):
{hook_guidance}

""")
]
)

Body_prompt = ChatPromptTemplate.from_messages([
    ("system",
 """
You are an expert LinkedIn content writer.

Your task is to write the MAIN BODY of a LinkedIn post that delivers value,
builds credibility, and keeps professionals reading till the end.

You may receive:
- User Query
- Reasoning
- Post Guidance
- Target Word Count
- Optional Web Search Context

HOW TO USE INPUTS:
- User Query defines the topic and scope.
- Reasoning provides the logical direction and key insights.
- Post Guidance defines the writing style and narrative structure (PRIMARY).
- Target Word Count must be followed strictly (±10 words).
- Web Search Context is OPTIONAL and should be used ONLY if it adds factual value.

POST GUIDANCE RULE:
- If Post Guidance is present, it OVERRIDES generic writing behavior.

WORD COUNT RULE:
- Target Word Count must be respected.
- Do not pad or compress unnaturally.

WRITING RULES:
- Write for LinkedIn professionals
- Clear logical flow
- Simple, conversational, human language
- No emojis
- No hashtags
- No CTA
- No hook (handled elsewhere)

DO NOT:
- Explain your process
- Mention being an AI
- Add formatting or markdown
- Introduce unrelated topics

OUTPUT RULES:
- No text outside JSON
- If JSON is invalid, retry internally

OUTPUT FORMAT (STRICT):
Return a JSON object with EXACTLY these two fields:
- content: the LinkedIn post body text
- reasoning: a short explanation of tone + structure (1–2 lines)

Write the LinkedIn post body based on the guidance.

Guidance:
{body_guidance}

 """
)
,
    ("user",
 """
User Query:
{user_query}

Reasoning:
{reasoning}

Body Guidance (STRICT):
{body_guidance}

Target Word Count:
{word_count}

Web Search Context (if any):
{search_result}

IMPORTANT:
- Output ONLY valid JSON
- Follow the format instructions EXACTLY
- Do NOT explain the schema
- Do NOT include markdown
""")
    ])
Body_prompt = Body_prompt.partial(
        format_instructions=parser_body.get_format_instructions()
)


Formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
          """ You are a LinkedIn content editor and formatter.
 
Your task is to combine the given HOOK, BODY, and CTA into ONE polished,
scroll-friendly LinkedIn post.
 
Follow these rules STRICTLY:
- Do NOT change the wording of the hook, body, or CTA
- Preserve original meaning and phrasing
- Improve readability using spacing and flow
- Add light, tasteful emojis ONLY where they feel natural
- Place emojis at the start of lines or sections (not mid-sentence)
- Add a maximum of 5 relevant hashtags at the end
- Ensure the post feels native to LinkedIn
 
Structure Rules:
1. Hook at the very top
2. Line break
3. Body content (well-spaced paragraphs)
4. Line break
5. CTA (1 line)
6. Line break
7. Hashtags
 
Do NOT:
- Output JSON
- Add markdown
- Add explanations
- Add extra commentary
- Add new content
- Reword or paraphrase anything
 
VERY IMPORTANT:
- Output MUST be plain text ONLY
- The result must look like a final LinkedIn post ready to publish
"""
        ),
        (
            "user",
            "HOOK:{hook}\n\nBODY:{body}\n\nCTA:{cta}. take the hook, body use it as is, just firmat it as a standalong post with right hastags and emojis"
        )
    ]
)



parallel_chain = RunnableParallel({
    "hookcta": CTA_Hook_prompt | llm | parser_hookcta,
    
    "body": Body_prompt | llm | parser_body
})


final_chain = parallel_chain | llm

def run_parallel_llms(pre_call : dict):
    return parallel_chain.invoke(pre_call)

def format_post(hook, body, cta):
    return (Formatting_prompt | llm).invoke({
        "hook": hook,
        "body": body,
        "cta" : cta
    })
