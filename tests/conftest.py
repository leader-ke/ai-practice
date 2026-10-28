"""Test setup: tests must never use a real key or reach the real API."""

import os

os.environ["LLM_MODEL"] = "test-model"
os.environ["ANTHROPIC_API_KEY"] = "test-key"
