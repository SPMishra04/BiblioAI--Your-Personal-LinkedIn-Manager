import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()


AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# ----------- CONTROL PARAMETERS -----------
MAX_CLARIFICATION_TURNS = int(os.getenv("MAX_CLARIFICATION_TURNS", 3))
NUM_SEARCH_QUERIES = int(os.getenv("NUM_SEARCH_QUERIES", 3))

# ----------- API KEYS -----------
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# # ----------- MODEL NAMES -----------
# gate_keeper_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
# fact_checker_MODEL= os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
# post_generator_MODEL= os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# # ----------- CONTROL PARAMETERS -----------
# MAX_CLARIFICATION_TURNS = int(os.getenv("MAX_CLARIFICATION_TURNS", 3))
# NUM_SEARCH_QUERIES = int(os.getenv("NUM_SEARCH_QUERIES", 3))
