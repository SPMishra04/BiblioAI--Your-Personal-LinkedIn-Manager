import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ----------- API KEYS -----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ----------- MODEL NAMES -----------
LLM1_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM2_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM3_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# ----------- CONTROL PARAMETERS -----------
MAX_CLARIFICATION_TURNS = int(os.getenv("MAX_CLARIFICATION_TURNS", 3))
NUM_SEARCH_QUERIES = int(os.getenv("NUM_SEARCH_QUERIES", 3))
