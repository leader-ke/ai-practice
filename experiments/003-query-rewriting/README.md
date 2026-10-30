# Experiment 003: query rewriting

Does asking a model to rewrite the question before searching find the right chunk more often? Nothing else changes: same embedding model, same chunks, same database.

Run it from the project root with `PYTHONPATH=src python experiments/003-query-rewriting/run.py`. Run [experiment 002](../002-pgvector-search/README.md) first, so the chunks are stored. Code: [query_rewriter.py](../../src/rewriting/query_rewriter.py) and [run.py](run.py).

## Setup
- **Rewriter:** `rewrite_query` sends the question to Claude Haiku 5.5 with a short instruction (keep the meaning, do not answer, use clear terms, return one question). It raises `QueryRewriteError` for an empty, cut-off, multi-line or very long reply. `rewrite_or_original` falls back to the original question, so a failed rewrite never stops a search.
- **Questions:** the original 5, plus 27 everyday-wording questions in [qa_retrieval_colloquial.json](../../data/golden_sets/qa_retrieval_colloquial.json), split into 18 to tune on and 9 held out.
- **Saved rewrites:** `data/rewrites/<model>.json`. Later runs reuse them, so the numbers do not change and cost nothing.
- **Metrics:** Hit@1 (right chunk first) and Hit@3 (right chunk in the top three).

## Results from the script
| Questions | Hit@1 original | Hit@1 rewritten | Hit@3 original | Hit@3 rewritten |
|---|---|---|---|---|
| original (5) | 4/5 | 4/5 | 4/5 | 5/5 |
| colloquial tune (18) | 10/18 | 14/18 | 15/18 | 17/18 |
| colloquial held out (9) | 5/9 | 5/9 | 6/9 | 8/9 |

## Tougher check
The script's corpus has 8 chunks, so even a poor search gets Hit@3 by luck. [extended.py](extended.py) repeats the test in memory (no database) with 45 more questions written by a separate Claude agent as casual newcomer wording, 20 realistic questions that were never used to tune anything, and again with 15 unrelated paragraphs added (23 chunks). Each question is rewritten three times, because the model's answer varies, and the samples are saved in `data/rewrites/`. MRR is the average of 1 divided by the rank of the right chunk. Run it with `PYTHONPATH=src python experiments/003-query-rewriting/extended.py`.

| Corpus | Search | Hit@1 | Hit@3 | MRR |
|---|---|---|---|---|
| 8 chunks, 45 questions | original | 0.56 | 0.87 | 0.73 |
| | rewritten (mean of 3) | 0.70 | 0.95 | 0.82 |
| 23 chunks, 45 questions | original | 0.42 | 0.73 | 0.61 |
| | rewritten (mean of 3) | 0.63 | 0.84 | 0.75 |
| 23 chunks, my 27 questions | original | 0.44 | 0.67 | 0.60 |
| | rewritten (mean of 3) | 0.70 | 0.88 | 0.81 |
| 8 chunks, 20 realistic | original | 0.60 | 0.85 | 0.75 |
| | rewritten (mean of 3) | 0.72 | 0.88 | 0.82 |
| 23 chunks, 20 realistic | original | 0.55 | 0.70 | 0.67 |
| | rewritten (mean of 3) | 0.65 | 0.87 | 0.77 |

Searching with both the original and the rewrite and taking the better score for each chunk did about as well as the rewrite alone.

## What I saw
- Rewriting helped in every setting, and more as the corpus got harder (Hit@1 gain of 14 points on 8 chunks, 21 on 23). The three samples agreed closely, so this is not one lucky rewrite.
- It is not free of harm. Across the 45 questions the rewrite put the right chunk higher 42 times and lower 19 times (8 chunks, summed over three samples). Example: "Should I attach a video of the problem?" worked as written, and the rewrite moved the answer down.
- One rewrite out of 231 was unusable (the model added a second line) and fell back to the original question.
- On the 20 realistic questions the gain is smaller and not clearly more than chance: the rank improved 15 times and worsened 10 (8 chunks), or 16 and 9 (23 chunks). With 20 questions I cannot tell a real gain from noise (sign test p=0.42 and 0.23). The three samples also differed more here (Hit@1 0.65 to 0.75 on 8 chunks).
- The rewrite hurts most on questions that are very short or vague, where it adds generic words (for example "typo on the settings page" became a question about reporting a text error, and the right chunk fell from rank 4 to 12).
- A rewrite took about 1.1 seconds (median, 2.6 at the slowest tenth) and one model call per search.

## Limitations
- All questions were written by Claude, who had seen the corpus, so they are probably easier than real user questions. The realistic set is the most honest one, and it shows the smallest gain. Rerun with questions from real users before trusting the size of the gain.
- Quality tests are not part of `pytest`, because they need the embedding model, the database and an API key. `tests/test_golden_sets.py` only checks that the question files and saved rewrites are valid.
- The corpus is tiny. The gain may be smaller or larger on a real knowledge base.
- Only one rewrite prompt and one model were tried. The prompt was not tuned.
- The original 5-question set cannot show much: 4 of 5 already worked.

## Next
Collect real questions from testers, or write 20 yourself without looking at the corpus, and rerun `extended.py`. If the gain holds, compare it with the cost and time. Two cheaper ideas to compare: only rewrite when the original search is unsure (low top score), and keyword plus vector search.
