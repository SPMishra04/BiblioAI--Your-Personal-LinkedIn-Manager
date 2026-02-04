from langchain_groq import ChatGroq

groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)
