# AI Engineering Practice

Hands-on practice building with the Anthropic SDK — conversations, tool use, and agentic patterns.

## Layout

| Path | What it holds |
|------|---------------|
| `src/agent/` | The agentic loop with tool use, and the custom error types |
| `src/api/` | The FastAPI app with `/ask` and `/health` |
| `src/retrieval/` | Chunking, embeddings and retrieval evaluation |
| `src/config.py` | Settings from environment variables (model, API key, max tokens) |
| `tests/` | Tests for the API, the agent loop (with a fake client), chunking and evaluation |
| `data/` | The sample corpus and the golden set of questions |
| `experiments/` | Numbered experiments and early scripts |
| `docs/` | Learning log, decisions and notes. Start at `docs/index.md` |

## Setup

```bash
python3 -m venv clean_env
source clean_env/bin/activate
pip install -r requirements-dev.txt
pip install ruff pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

`requirements.txt` is what the app needs, `requirements-dev.txt` adds what the tests need, and `requirements-retrieval.txt` adds the embedding model for the experiments (it installs PyTorch, which is large):

```bash
pip install -r requirements-retrieval.txt
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=your-key-here
export LLM_MODEL="claude-haiku-5-5"
```

Optional settings: `MAX_TOKENS` (default 1000) and `REQUEST_TIMEOUT_SECONDS` (default 30).

### Run
```bash
uvicorn api.main:app --app-dir src --reload
```

Run the chunking experiment:

```bash
PYTHONPATH=src python experiments/001-chunk-size-and-prefix/run.py
```

Run the Postgres vector search experiment. It needs `DATABASE_URL` (a Neon connection string) in `.env`:

```bash
PYTHONPATH=src python experiments/002-pgvector-search/run.py
```

Run the query rewriting experiment (run 002 first so the chunks are stored; it calls the Anthropic API the first time and reuses saved rewrites after that):

```bash
PYTHONPATH=src python experiments/003-query-rewriting/run.py
```

### Test
```bash
python -m pytest -q
```

The tests never use a real key or reach the real API, so they need no setup. CI runs them on every push and pull request.

## Linting

Ruff runs automatically on commit (auto-fix) and push (strict, no fix). CI runs ruff and the tests on every push and pull request to main.

```bash
ruff check .        # lint
ruff format .       # format
ruff check --fix .  # lint + auto-fix
```
