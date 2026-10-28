# Learning log

A first draft. Lines marked TODO(ken) are for you to rewrite in your own words, and the mistakes are listed from memory, so correct anything that is wrong.

## 28 Oct 2026: tidying and hardening the project
Moved the code into `src/agent`, `src/api` and `src/retrieval`, with `tests/`, `data/` and `experiments/` beside it. The tests still passed and experiment 001 printed exactly the same results, which is how I knew the move was safe. Python needs to know where `src` is, so pytest has a `pythonpath` setting and uvicorn gets `--app-dir src`.

Then I fixed what the move showed up. `requirements.txt` was missing three libraries, so the project could not be rebuilt from scratch. CI never ran the tests, and the tests only passed because my real `.env` existed. The agent loop itself had no test, because the API tests replaced it with a fake. Now a fake Anthropic client drives the loop through every path, and the tests run in a fresh environment with no key (44 pass, up from 13).

To check the new tests are worth anything, I broke the code four ways on purpose. Three were caught. The fourth, changing the round cap from 5 to 6, slipped through because my test compared against the constant itself. Lesson: a test must pin the value, not repeat the code.

Also: a 30 second client timeout, `AgentExecutionError` when the agent gives up, `MAX_TOKENS` as an integer, and one table instead of six repeated `except` branches.

TODO(ken): which of these would you do first on a new project, and why?

## 27 to 28 Oct: embeddings, chunking and measuring retrieval
I turned sentences into 384 numbers with `all-MiniLM-L6-v2` and compared them with cosine similarity. "How do I reset my password?" and "I forgot my login details" scored 0.58 with almost no words in common. A weather sentence scored about 0. The scores only mean something next to each other, since 0.58 counted as strong for this model.

Embeddings match topic, not truth. "The test passed" and "the test did not pass" scored 0.83, so a search can return the opposite of what you want. The model also ignores everything past 256 tokens, about 1000 characters, which is why chunk size has a ceiling.

I then built a paragraph-aware chunker and a five-question golden set, and measured Hit@1 and Recall@3 at several sizes, with and without a document-name prefix ([experiment 001](../experiments/001-chunk-size-and-prefix/README.md)). The main lessons: Hit@1 and Recall@k answer different questions, and for RAG, where the model reads the top few chunks, Recall@k usually matters more.

Mistakes: my retrieval code sat outside the loop, so it ran once while I thought I was comparing sizes. 800 and 1200 characters made the same three chunks, so they were one experiment. Recall@3 was 100% with only three chunks, which tells you nothing.

TODO(ken): why does chunking matter, in your words?

Open: does the prefix still hurt Hit@1 on a bigger golden set? And what fixes the one question that fails everywhere (keyword plus vector search, re-ranking)?

## 23 to 26 Oct: a tested HTTP backend
I built `/health` and `/ask` with FastAPI ([main.py](../src/api/main.py)), checked input with Pydantic, mapped agent failures to status codes, and wrote 13 pytest tests with a fake agent ([test_api.py](../tests/test_api.py)). I also pinned `requirements.txt`, moved settings into environment variables ([config.py](../src/config.py)), added ruff with pre-commit, and proved the setup in a fresh Python 3.12 environment.

What stuck:
- Validate at the boundary. Pydantic turns empty, missing, wrong-type and non-JSON input into a `422` before my code runs.
- Pick the status code by asking who is at fault. A bad API key is our configuration, so `500`, not `401`.
- Never send `str(e)` to the caller. Log the detail on the server and return a short generic message.
- Translate the SDK's errors into my own types in `agent`, so the API layer knows nothing about the vendor. Use `raise ... from e`, and catch the most specific error first.
- Test with a fake agent: it is free, fast, repeatable and can fail on demand. Patch the name where it is used (`main.run_agent`), use `parametrize` for tables of cases, test the boundaries, and break a test on purpose to prove it can fail.

Mistakes: I started a second server while the first held port 8000 and tested the old one. A string missing its `f` returned the literal text `{str(e)}`, and fixing it would have leaked the error anyway. I imported `run_agent` from two modules, and the second silently replaced the first. I also thought state belonged to a "session". The API has none: my code keeps the history and resends it.

TODO(ken): what does "reproducible" mean for a project, in your words?

Open: can the repeated `except` branches in `main.py` become a table or exception handlers? Where should expected failures be logged at a lower level than `logger.exception`?

## 22 Oct: the first agent
I made my first API call, read the response and token usage ([basics.py](../experiments/000-first-calls/basics.py), [compute_usage.py](../experiments/000-first-calls/compute_usage.py)), then gave Claude a tool and wrote the agent loop by hand ([src/agent](../src/agent/)). The API is stateless, a response is a list of blocks, and `stop_reason` tells the code why Claude stopped. The full picture is in [the agent loop note](notes/agent-loop.md).

Mistakes: I read `content[1].text` and it broke when Claude's first block was a tool request. I wrote the assistant turn by hand instead of sending Claude's real reply. My first version handled one tool request and one round.

TODO(ken): the tool round trip in four lines, in your words.

Open: when is the SDK's Tool Runner enough, and when do I want the loop in my own hands?
