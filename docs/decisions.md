# Decisions

Each entry says what I chose, why, and what I turned down. Newest first.

## 30 Oct: query rewriting is provisional, and failures fall back to the original question
Rewriting helped on every test I ran, but the questions were written by Claude and the corpus is tiny, and on the most realistic set the gain was small and within noise. I treat it as a likely gain, not a settled one, and will decide after testing real tester questions. The function never blocks a search: if the rewrite fails, `rewrite_or_original` uses the original question. Rewrites are saved per model in `data/rewrites/`, so experiments give the same numbers each time. I dropped my hand-written "manual" rewrites because they leaked the answer. See [experiment 003](../experiments/003-query-rewriting/README.md).

## 30 Oct: one table with a label for each experiment
`chunks` has a `config` column (for example `400-noprefix`), so chunk sizes and prefixes can sit in one table and each search filters on one label. The alternative was a table per experiment, which means copying the schema every time. The cost is that the label must be filtered on every query, and re-running an experiment needs its old rows cleared first.

## 30 Oct: vector search lives in its own database on Neon
I used Neon's free Postgres, which supports pgvector, and created a separate `ai_practice` database inside an existing project that had not been used since June, renaming the project to `ai-practice`. The old database in it was left untouched. A new project was not an option because the free plan already held ten. The connection string is only in `.env`, and I created the tables through the SQL editor and the Neon MCP, not from my code.

## 28 Oct: tests never touch a real key
`tests/conftest.py` sets a dummy model and key before anything imports. A test that slips through to the real API fails with a `401` instead of spending money, and the suite runs the same on any machine, including CI.

## 28 Oct: the app and the experiments have separate requirements
`requirements.txt` holds what the API needs, `requirements-dev.txt` adds the test tools, and `requirements-retrieval.txt` adds the embedding model. PyTorch is large, and CI only needs it for experiments, not for the tests, so the embedding library is imported when a model is first loaded. `pyproject.toml` lists the same libraries without versions, as the loose list, while the requirements files hold the exact versions.

## 28 Oct: the Anthropic client has a 30 second timeout and is created on first use
The default timeout is ten minutes, far too long for a web request that can also retry. Creating the client when it is first needed, and not at import, keeps imports free of side effects and lets tests swap in a fake. The agent now also logs tokens and seconds for every round.

## 28 Oct: project layout
Code lives in `src/` (agent, api, retrieval), with `tests/`, `data/`, `experiments/` and `docs/` beside it. It groups the code by stage, keeps the corpus and golden set out of the code, and lets each experiment run against the same modules. The cost is that Python needs to be told where `src` is. I kept the flat layout until the tests existed, so the move could be checked.

## 28 Oct: chunking setup is provisional
Use paragraph-aware chunks of 250 to 400 characters. The prefix is undecided. The two sizes tie without a prefix (Hit@1 4/5, Recall@3 4/5). A prefix raises Recall@3 to 5/5 but lowers Hit@1 to 3/5, and since RAG reads the top few chunks, Recall@3 probably matters more. Five questions is too few to settle it, so revisit with a bigger golden set. See [experiment 001](../experiments/001-chunk-size-and-prefix/README.md).

## 27 Oct: embeddings from a local model
I use `all-MiniLM-L6-v2` through `sentence-transformers`. It is free, works offline once downloaded and needs no new account while I learn. A hosted embeddings API would need another key.

## 26 Oct: settings come from environment variables
`config.py` reads the model, API key and token limit from the environment and stops at startup if the model or key is missing, so a mistake shows up immediately and not on the first request. The `.env` file stays out of git. The alternative was hard-coding the model name, which meant editing code to change it.

## 23 Oct: agent failures become my own error types
`agent` translates the SDK's errors into `AgentConfigError` (500), `AgentUnavailableError` (503), `AgentTimeoutError` (504) and `AgentExecutionError` (500), and the API maps those to status codes. The API stays independent of the vendor, each failure gets an honest status, and tests can force every case. My first version used one error for everything and mapped a bad key to `503`.

## 23 Oct: a bad Anthropic key returns 500, not 401
The caller did nothing wrong. It is our configuration, so a `401` would blame them and a `503` would tell them to retry something that can never work.

## 22 Oct: Haiku 5.5 for practice
It is the cheapest current model and nothing in the code depends on it. The model is a setting, so it can change when there is something worth the extra cost.
