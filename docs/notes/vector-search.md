# Vector search in Postgres

Code: [src/retrieval/search.py](../../src/retrieval/search.py). Experiment: [002](../../experiments/002-pgvector-search/README.md).

Keeping vectors in Python memory means they vanish when the program stops, have to be recomputed every run, and cannot be filtered. Postgres with the pgvector extension stores them in a column and finds the nearest ones with SQL.

## The pieces
- **Extension, per database.** `CREATE EXTENSION vector;` turns it on for one database only. I ran it in `ai_practice`, then got `type "vector" does not exist` when the SQL editor was pointed at my old database. Always check which database is selected.
- **A vector column.** `embedding vector(384)`. The number must match the model, and a vector of another length is rejected.
- **A distance operator.** `<=>` is cosine distance, which is `1 - cosine similarity`, so smaller means closer. On my demo rows, `[1,0,0]` against `[1,0,0]` gave 0, against `[0.9,0.1,0]` gave 0.0061, and against `[0,1,0]` gave 1.
- **The search is a sort.** `ORDER BY embedding <=> query LIMIT 3` returns the three nearest chunks.

## The table
One table holds every experiment. A `config` column labels how a chunk was made (for example `400-noprefix`), so different chunk sizes and prefixes can live side by side and each search filters on one label. `paragraph_ids` is a text array, which is what the golden set is checked against.

## Habits that matter
- Pass values as parameters (`%s`), never build SQL by pasting strings. It stops injection and handles vectors correctly.
- Commit after writing, and close the connection in `finally`.
- The connection string contains the password. It lives only in `.env`.
- Cosine distance ignores vector length, so normalizing vectors does not change the ranking here. It would matter for a different operator such as inner product.

## When an index matters
With a few rows, Postgres compares every vector, which is exact and fast. Large tables need an approximate index (pgvector offers HNSW and IVFFlat), which is faster but can miss a true neighbor. I have not needed one yet.

## Neon notes
A free Neon branch that has been idle is archived, and the first query wakes it, so it can be slow once. I used a separate database inside the project so my other tables are untouched.

TODO(ken): explain in your own words why the search is "just a sort".
