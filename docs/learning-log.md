# Learning log

A first draft. The mistakes are listed from memory, so correct anything that is wrong.

## 30 Oct 2026: query rewriting
I wrote a function that asks Claude to rewrite a question before searching, and tested whether it finds the right chunk more often ([experiment 003](../experiments/003-query-rewriting/README.md)). On 45 casual questions it raised Hit@1 from 0.56 to 0.70 on the small corpus, and from 0.42 to 0.63 once I added 15 unrelated paragraphs. It also made some questions worse (19 of 135 comparisons), costs one model call and about a second per search, and one rewrite in 231 was unusable.

What stuck:
- My first test was flawed: I wrote the "manual" rewrites knowing which paragraph they should find, so they proved nothing. A test question must not be written with the answer in view.
- Five questions say almost nothing. The corpus was also so small that Hit@3 was nearly luck, so a harder corpus gave a fairer test.
- A model's output varies, so save the rewrites and test several samples.
- A rewrite can fail or be cut off, so the search needs a fallback to the original question.

Mistakes: I did not check for a cut-off reply (`stop_reason`), or translate SDK errors, in my first version of the function.

On 20 realistic questions, written after looking at how people really phrase these and never used to tune anything, the gain shrank (Hit@1 0.60 to 0.72 and 0.55 to 0.65) and was within noise, because 20 questions is too few to separate a real gain from chance. It hurt most on short, vague questions.

TODO(ken): collect real tester questions and rerun [extended.py](../experiments/003-query-rewriting/extended.py). Does the gain hold?

Open: is the gain worth one extra model call per search? How does it compare with keyword plus vector search?

## 30 Oct 2026: vector search in Postgres
I moved the chunk vectors from memory into Postgres on Neon, with pgvector, and searched them with SQL ([experiment 002](../experiments/002-pgvector-search/README.md), [note](notes/vector-search.md)). On the demo table I predicted the distances before running the query, and `<=>` gave 0, 0.0061 and 1, because it is a distance (smaller is closer), not a similarity.

The result is the point: Postgres gave Hit@1 4/5 and Recall@3 4/5, exactly like the in-memory search from experiment 001, and returned the same top three chunks for the question that always fails. A database changes where vectors live and how fast they are found, not what is found.

What stuck:
- Turning on pgvector is per database. Running the demo in my old database gave `type "vector" does not exist`, because I had enabled it only in `ai_practice`. Check which database the editor is pointing at.
- A vector column has a fixed size that must match the model (384 here).
- Pass values as parameters, never build SQL by pasting strings, and keep the connection string in `.env` only.

Mistakes: I ran the demo in the wrong database. Reading the code also showed that running the experiment twice would store every chunk twice and duplicate the search results, because storing never clears old rows. I have not checked that `1 - distance` equals the earlier cosine scores.

TODO(ken): why is vector search "just a sort"?

Open: when does an index become worth it, and what does it cost in accuracy? Should the experiment reuse the metric code in `evaluation.py` instead of repeating it?

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
