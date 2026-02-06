from pydantic import BaseModel, Field

class IntentAnalyzer(BaseModel):
    reasoning: str = Field(
        description=(
            "Explanation of how the user's query was interpreted, "
            "including intent detection and whether external information is required."
        )
    )
    websearch: bool = Field(
        description=(
            "Indicates whether a web search is required to answer the query accurately. "
            "True if the query depends on real-time, recent, factual external data or if the data is not present in memorystore ."
        )
    )

 
class HookCTA(BaseModel):
    hook: str = Field(
        description="An attention-grabbing opening line designed to capture the reader’s interest immediately."
    )
    cta: str = Field(
        description="A clear call-to-action instructing the reader on what to do next."
    )

 
class Body(BaseModel):
    content: str = Field(
        description="The main body of the generated content presented to the user."
    )
    reasoning: str = Field(
        description="Internal explanation of why the content was structured or written in this way."
    )

class Formatting(BaseModel):
    post: str = Field(
        description="The fully formatted final output ready for publishing."
    )


