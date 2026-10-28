# Retrieval evaluation

Code: [src/retrieval](../../src/retrieval/). Experiment: [001](../../experiments/001-chunk-size-and-prefix/README.md).

The question is simple: given a user's question, does search put the right chunk where it will be used? **Hit@1** counts the cases where the right chunk ranked first. **Recall@k** counts the cases where it is somewhere in the top k. If a system uses only the first result, Hit@1 is the number to watch. In RAG the model reads the top few chunks, so the answer only has to be among them, and Recall@k usually matters more.

The test data is a **golden set**: a small file of questions, each with the chunk that should answer it, kept in `data/` and not in the code. Mine has five questions. They are written in everyday words and not copied from the text, so they test meaning.

## What can mislead you
- **k close to the number of chunks.** With three chunks, Recall@3 is always 100%. A metric that cannot fail tells you nothing.
- **Runs that are not different.** 800 and 1200 characters both gave three chunks, so they were one experiment. Print the chunk count.
- **Tiny test sets.** With five questions, one question is 20 points. A gap of one question is a hint, not proof. Aim for 20 to 50 questions before deciding anything.
- **Absolute scores.** A cosine score of 0.58 was strong for this model. Compare scores within one search.
- **Opposites.** "Passed" and "did not pass" scored 0.83, because embeddings capture topic, not truth.

## What moved the numbers
- Chunks from 600 characters up blur their paragraphs together. 250 and 400 behaved the same.
- The document-name prefix helps tell documents apart and hurts telling paragraphs apart inside one document, because every chunk of that document then shares words with the question.
- The model reads only 256 tokens, about 1000 characters.

One question, "How should QA testers protect sensitive information in bug reports?", fails at Hit@1 in every setting. Paragraphs from the test-environment notes share its words. Ideas for later: combine keyword and vector scores, or re-rank the top results with a stronger model.

TODO(ken): which metric would you report for the QA triage agent, and why?
