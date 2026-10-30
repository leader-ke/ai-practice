# Experiment 002: vector search in Postgres

Does searching with SQL in Postgres return the same results as the in-memory search from [experiment 001](../001-chunk-size-and-prefix/README.md)?

Run it from the project root with `PYTHONPATH=src python experiments/002-pgvector-search/run.py`. It needs `DATABASE_URL` in `.env` and the packages in `requirements-retrieval.txt`. Notes: [vector search](../../docs/notes/vector-search.md).

## Setup
- **Database:** Neon Postgres 17 with pgvector 0.8.0, in a separate database called `ai_practice`. It holds two tables: `demo` (three tiny vectors, for learning the operators) and `chunks`.
- **Chunks:** the same corpus and golden set as experiment 001, split at 400 characters without a prefix, which gives 8 chunks. Each row stores the chunk text, its paragraph IDs, a label for how it was made (`400-noprefix`) and its 384-number embedding.
- **Search:** embed the question, then `ORDER BY embedding <=> query LIMIT 3`. The operator `<=>` is cosine distance, so smaller means closer.
- **Code:** [src/retrieval/search.py](../../src/retrieval/search.py) and [run.py](run.py).

## Results
| | Hit@1 | Recall@3 |
|---|---|---|
| Postgres search | 4/5 | 4/5 |
| In-memory search (experiment 001, 400 characters, no prefix) | 4/5 | 4/5 |

The two agree. For the question that always fails ("How should QA testers protect sensitive information in bug reports?"), Postgres returned the same top three chunks in the same order as experiment 001: `environment_1`, `environment_2`, `report_1`.

The demo table behaved as expected: against `[1,0,0]`, row `a` had distance 0, row `b` 0.0061 and row `c` 1.

## What I saw
- The database gives the same answers as the code I wrote by hand. Storing vectors in Postgres changes where they live and how fast they are found, not what is found.
- Nothing here needed an index. With 8 rows, Postgres compares every vector, which is exact and instant. An index only matters when the table is large, and it trades a little accuracy for speed.

## Not checked yet
- `1 - distance` should equal the cosine scores printed in experiment 001 (about 0.510 for the top chunk of the sensitive-information question). The script now prints it as "Similarity", so compare the two outputs.

## Limitations
- Storing now deletes the old rows for a label first, so running the script twice no longer duplicates chunks.
- The script re-implements the Hit@1 and Recall@3 counting that already exists in `retrieval/evaluation.py`.
- It imports `config`, which also requires the Anthropic settings that retrieval does not use.

## Next
Answer questions with the retrieved chunks and cite them (RAG), then measure the answers.
