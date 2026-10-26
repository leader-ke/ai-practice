import os

LLM_MODEL = os.environ.get("LLM_MODEL")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MAX_TOKENS = os.environ.get("MAX_TOKENS", 1000)
