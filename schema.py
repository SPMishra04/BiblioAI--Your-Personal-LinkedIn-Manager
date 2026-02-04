from pydantic import BaseModel, Field
from typing import Optional


class RestrictedCheck(BaseModel):
    allowed: bool = Field(description="Whether the user query is allowed")
    message: Optional[str] = Field(description="Rejection message if not allowed")


class IntentAnalyzer(BaseModel):
    reasoning: str = Field(description="Why the query is interpreted this way")
    websearch: bool = Field(description="Whether web search is required")
    websearch_key: Optional[str] = Field(
        description="Main keyword/topic for web search if required"
    )


class LinkedInTrainer(BaseModel):
    word_count: int = Field(description="Recommended word count for the post")
    hook_guidance: str = Field(description="Guidance on how the hook should be written")
    body_guidance: str = Field(description="Guidance on how the post body should be written")


# class HookCTA(BaseModel):
#     hook: str
#     cta: str


# class Body(BaseModel):
#     content: str


class LinkedInPostOutput(BaseModel):
    hook: str = Field(description="Opening hook for LinkedIn post")
    cta: str = Field(description="Call to action")
    body: str = Field(description="Main body of the LinkedIn post")



class HookCTA(BaseModel):
    hook: str = Field(
        description="An attention-grabbing opening line designed to capture the reader’s interest immediately."
    )
    cta: str = Field(
        description="A clear call-to-action instructing the reader on what to do next."
    )

 
class Body(BaseModel):
    content: str = Field(
        description="Final LinkedIn post body text."
    )
    reasoning: str = Field(
        description="Why this post was written this way."
    )
