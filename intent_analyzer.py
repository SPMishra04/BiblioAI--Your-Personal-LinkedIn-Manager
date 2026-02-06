from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from config import GROQ_API_KEY, MODEL_NAME
from schemas import IntentAnalyzer


llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=0
)

parser = PydanticOutputParser(pydantic_object=IntentAnalyzer)

intent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an intent analyzer for a multi-agent LLM system.

Your task is to analyze a user query and decide:
1. Understand the the goal and meaning of what User is asked for, Know the user requirement first.
2. Check whether to generate a LinkedIn Post for the user query web search is required or not.
3. Mostly when the user asked for any universal subject then Web search is required.
4. When the user tells about a story or their personal experience or sentimental experiences, there web search is not required at all.
5. Make sure the web_search decesion should always be correct and alligned with the requirement if required.
4. Whatever you result generate a clear and on-point reasoning behind it.

Always give the output in JSON Format"

{format_instructions}
"""
        ),
        (
            "user",
            "{user_input}.  Make sure the output requirement is in JSON Format"
        )
    ]
)


def analyze_intent(user_input: str) -> IntentAnalyzer:
    prompt = intent_prompt.format(
        user_input=user_input,
        format_instructions=parser.get_format_instructions()
    )

    response = llm.invoke(prompt)
    return parser.parse(response.content)
