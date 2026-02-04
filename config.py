import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =========================
# LLM CONFIGURATION
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Please set it in the .env file.")

if not TAVILY_API_KEY:
    raise ValueError("❌ TAVILY_API_KEY not found. Please set it in the .env file.")

# Recommended fast + cheap Groq model
MODEL_NAME = "llama-3.1-8b-instant"

# Default temperature for deterministic + safe generation
TEMPERATURE = 0.4
