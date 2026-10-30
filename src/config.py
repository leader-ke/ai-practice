import os

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.environ.get("LLM_MODEL")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1000"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))
DATABASE_URL = os.environ.get("DATABASE_URL")

if not LLM_MODEL:
    raise ValueError("LLM_MODEL is missing from environment configuration.")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is missing from environment configuration.")

if MAX_TOKENS <= 0:
    raise ValueError("MAX_TOKENS must be a positive integer.")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing from environment configuration.")
