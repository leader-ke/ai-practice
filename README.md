# AI Engineering Practice

Hands-on practice building with the Anthropic SDK — conversations, tool use, and agentic patterns.

## Files

| File | What it covers |
|------|---------------|
| `basics.py` | Multi-turn conversation, accessing thinking blocks |
| `compute_usage.py` | Token counting and cost estimation per model |
| `tools_lookup_ticket.py` | Tool definitions, tool use handling, agentic loop with max rounds |

## Setup

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install anthropic ruff pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=your-key-here
```

## Linting

Ruff runs automatically on commit (auto-fix) and push (strict, no fix). CI runs on every push/PR to main.

```bash
ruff check .        # lint
ruff format .       # format
ruff check --fix .  # lint + auto-fix
```
