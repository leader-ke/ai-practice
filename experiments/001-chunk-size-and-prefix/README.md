# Experiment 001: chunk size and document prefix

How do chunk size and a document-name prefix affect finding the right paragraph?

Run it from the project root with `PYTHONPATH=src python experiments/001-chunk-size-and-prefix/run.py`. Notes: [retrieval evaluation](../../docs/notes/retrieval-evaluation.md).

## Setup
- **Corpus:** three short documents of three paragraphs each (bug severity, bug reports, test environment), in `data/corpus/`.
- **Golden set:** five questions, each with the paragraph that answers it, in `data/golden_sets/`.
- **Model:** `all-MiniLM-L6-v2`, cosine similarity.
- **Chunker:** groups whole paragraphs up to a character limit and splits an oversized paragraph on word boundaries.
- **Prefix variant:** the text embedded is `Document: <name>` on one line followed by the chunk. The displayed text is unchanged.

## Results
| Max characters | Chunks | Prefix | Hit@1 | Recall@3 |
|---|---|---|---|---|
| 250 | 9 | no | 4/5 | 4/5 |
| 250 | 9 | yes | 3/5 | 5/5 |
| 400 | 8 | no | 4/5 | 4/5 |
| 400 | 8 | yes | 3/5 | 5/5 |
| 600 | 6 | no | 3/5 | 5/5 |
| 600 | 6 | yes | 3/5 | 5/5 |

800 and 1200 characters both give three chunks, one per document, so they are one run (Hit@1 3/5, and Recall@3 is always 5/5 with three chunks).

## What I saw
250 and 400 tie without a prefix, and from 600 up the paragraphs blur together. The prefix helped find the right document: for the sensitive-information question the correct paragraph moved from rank 4 to rank 3. It hurt inside a document: for "difference between severity and priority", `severity_3` overtook `severity_1`, because every severity chunk now shares words with the question. One question fails at Hit@1 in every setting, because paragraphs from another document use the same words.

## Caveats
Five questions means one question is 20 points, so treat a gap of one as a hint. Recall@3 says nothing when there are only three chunks.

## Conclusion so far
250 to 400 characters work best on this corpus. For RAG the prefix may be worth it for the Recall@3 gain, but it is not decided. See [decisions](../../docs/decisions.md). Next: 20 to 50 questions, including some that need two chunks, and keyword plus vector search for the stubborn question.
