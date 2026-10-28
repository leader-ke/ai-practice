# AI Engineering Practice

Hands-on practice building with the Anthropic SDK — conversations, tool use, and agentic patterns.

## Files

| File | What it covers |
|------|---------------|
| `basics.py` | Multi-turn conversation, accessing thinking blocks |
| `compute_usage.py` | Token counting and cost estimation per model |
| `config.py` | Environment variable loading (model, API key, max tokens) |
| `agent.py` | Agentic loop with tool use, custom error hierarchy |
| `main.py` | FastAPI app exposing `/ask` and `/health` endpoints |
| `test_main.py` | Tests for the FastAPI endpoints |
| `transformers_chunking_retrieval.py` | Document chunking and semantic retrieval with sentence-transformers |

## Setup

```bash
python3 -m venv clean_env
source clean_env/bin/activate
pip install -r requirements.txt
pip install ruff pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=your-key-here
export LLM_MODEL="claude-haiku-5-5"
```

### Run
```bash
uvicorn main:app --reload
```

### Test
```bash
python -m pytest -q
```

## Linting

Ruff runs automatically on commit (auto-fix) and push (strict, no fix). CI runs on every push/PR to main.

```bash
ruff check .        # lint
ruff format .       # format
ruff check --fix .  # lint + auto-fix
```
