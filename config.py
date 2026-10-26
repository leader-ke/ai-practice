import os

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.environ.get("LLM_MODEL")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MAX_TOKENS = os.environ.get("MAX_TOKENS", 1000)

if not LLM_MODEL:
    raise ValueError("LLM_MODEL is missing from environment configuration.")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is missing from environment configuration.")
