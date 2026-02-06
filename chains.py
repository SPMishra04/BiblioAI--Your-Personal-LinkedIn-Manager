from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate , PromptTemplate
from schemas import HookCTA, Body, Formatting
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel
from config import GROQ_API_KEY, MODEL_NAME
from memory import memory_store


llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=0.7
)


parser = JsonOutputParser()

CTA_Hook_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
          """  You are an expert LinkedIn content strategist and copywriter.
          
 
Your task is to generate:
1. A high-impact HOOK (first 1 line)
2. A strong, CALL TO ACTION (CTA)
 
Follow these rules strictly:
- Write specifically for LinkedIn professionals
- Keep hooks short, scroll-stopping, and curiosity-driven
- Use conversational, human, and authentic language
- Avoid emojis overload, or marketing jargon
- Match a professional but relatable tone
- CTAs should encourage meaningful engagement (comment, share, save, reflect), not sales
- If output is not valid JSON matching the required schema, retry internally.
 
Hook Guidelines:
- Max 2 lines
- Can be a bold statement, surprising insight, question, or relatable pain point
- Must make the reader want to click “see more”
 
CTA Guidelines:
- 1 line only
- Soft CTA (invite, not force)
- Aligned with the post topic and audience intent
 
Do NOT:
- Use hashtags
- Repeat the hook in the CTA
- Use generic phrases like "Let me know your thoughts" unless contextually strong

Return ONLY valid JSON.Always give the output in JSON Format.
 """
        ),
        (
            "user",
            "User Query:{user_query},Reasoning:{reasoning},Web Search Results (if any):{websearch_results}, Make sure the output requirement is in JSON Format"
        )
    ]
)



Body_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
You are an expert LinkedIn content writer.

Your task is to write the MAIN BODY of a LinkedIn post that delivers value,
builds credibility, and keeps professionals reading till the end.

You may receive Past Context as "memory_context".

IMPORTANT UNDERSTANDING OF MEMORY:
- Past Context contains the user's OWN previous experiences, facts, or statements.
- Past Context is authoritative ONLY IF it is DIRECTLY RELEVANT to the current User Query.
- Relevance means the memory shares the SAME topic, event, timeline, or intent.

CRITICAL MEMORY RULES:
1. Before writing, internally evaluate whether Past Context is RELEVANT to the current User Query.
2. Use Past Context ONLY if it clearly supports, continues, or enriches the current topic.
3. If Past Context is unrelated, or off-topic, you MUST IGNORE it completely.
4. NEVER force memory usage just because it is present.
5. NEVER introduce past topics that change the current topic.

CONTENT & LENGTH RULES:
- Adjust the length of the body based on the complexity of the User Query:
  - Short, opinion-based queries → concise body
  - Detailed, narrative, or multi-part queries → longer body
- Do NOT artificially expand or shrink the content.
- Depth should feel natural and proportional to the query.

WRITING RULES:
- Write for LinkedIn users (tech, business, AI, leadership, extracurricular, adventurous)
- Maintain a clear, logical flow of ideas
- Use simple, conversational, human language
- Avoid buzzwords, hype, or generic motivational fluff
- Keep the tone professional, insightful, and relatable
- No emojis in the body
- No hashtags
- No CTA here (CTA is handled separately)

CONTENT GUIDELINES:
- Expand clearly on the topic introduced by the User Query
- Explain why it matters to the reader
- Use short paragraphs (2–3 lines max)
- Bullet points allowed ONLY if they improve clarity
- Avoid repetition and unnecessary filler
- If relevant memory is used, blend it subtly and naturally

DO NOT:
- Explain what you are doing
- Mention that you are an AI
- Add a hook or CTA
- Add hashtags
- Add markdown formatting
- Introduce unrelated past topics

OUTPUT RULES:
- Return ONLY valid JSON matching the required schema
- Do NOT include explanations
- Do NOT include text outside JSON
- If output is not valid JSON matching the schema, retry internally
"""
        ),
        
    ("user","Past Context: {memory_context}, User Query:{user_query}, Reasoning:{reasoning}, Web Search Results (if any):{websearch_results}.  Make sure the output requirement is in JSON Format")
])


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
            "HOOK:{hook} , BODY:{body}. take the hook, body use it as is, just firmat it as a standalong post with right hastags and emojis"
        )
    ]
)

parallel_chain = RunnableParallel(
    {
        "hook": CTA_Hook_prompt | llm | parser,

        "body": Body_prompt | llm | parser
    }
)


final_chain = parallel_chain | llm 

def prepare_chain_input(user_id: str, user_query: str, reasoning: str, websearch_results: str):
    # 1. Fetch memory
    past_memories = memory_store.get_relevant_memory(
        user_id=user_id,
        query=user_query
    )

    # 2. ✅ SAFE conversion (THIS is the line you asked about)
    memory_context = "\n".join(past_memories)

    # 3. Return input dict for chain
    return {
        "user_query": user_query,
        "reasoning": reasoning,
        "websearch_results": websearch_results,
        "memory_context": memory_context
    }


def run_parallel_llms(pre_call: dict):
    return parallel_chain.invoke(pre_call)


def run_formatting(hook: str, body: str):
    return (Formatting_prompt | llm ).invoke(
        {
            "hook": hook,
            "body": body
        }
    )

########### new one below

memory_cleaner_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Extract ONLY factual travel events.

Rules:
- Keep day numbers if mentioned
- Keep food, places, activities
- REMOVE instructions like "write", "include", "mention"
- Bullet points only
- Do NOT invent missing days

Text:
{text}
"""
)

def clean_memory(text):
    return llm.invoke(
        memory_cleaner_prompt.format(text=text)
    ).content.strip()


# -------------------------------
# FACT EXTRACTOR (for memory)
# -------------------------------

fact_extractor_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Extract ONLY factual statements from the text.

Rules:
- No assumptions
- No opinions
- No rewriting
- No added details
- Bullet points only

Text:
{text}
"""
)

def extract_facts(text: str) -> str:
    """
    Extracts clean factual points for memory storage.
    """
    return llm.invoke(
        fact_extractor_prompt.format(text=text)
    ).content.strip()

