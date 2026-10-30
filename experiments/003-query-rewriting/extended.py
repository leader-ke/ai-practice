"""Experiment 003, tougher check: does rewriting still help on harder tests?

Runs in memory (no database). It uses several question sets, each question
rewritten three times because the model's answer varies, on two corpora:
the 9 handbook paragraphs alone, and with 15 unrelated paragraphs added.

Samples are saved in data/rewrites/<model>-samples.json and reused, so the
numbers do not change between runs. Delete the file to sample again.
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import numpy as np

from config import ANTHROPIC_API_KEY, LLM_MODEL, MAX_TOKENS
from retrieval.chunking import chunk_documents
from retrieval.embeddings import embed, load_model
from rewriting.query_rewriter import QueryRewriteError, rewrite_query

DATA = Path(__file__).resolve().parents[2] / "data"
SAMPLES = 3
MAX_CHARS = 400


def read(path):
    return json.loads((DATA / path).read_text(encoding="utf-8"))


def load_sets():
    return {
        "original (5)": read("golden_sets/qa_retrieval.json")["questions"],
        "colloquial (27)": read("golden_sets/qa_retrieval_colloquial.json")["questions"],
        "blind (45)": read("golden_sets/qa_retrieval_blind.json")["questions"],
        "realistic (20)": read("golden_sets/qa_retrieval_realistic.json")["questions"],
    }


def collect_samples(questions, path):
    """Returns {"<n>|<question>": rewrite or None}, asking the model for missing ones."""
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def ask(job):
        question, i = job
        key = f"{i}|{question}"
        if key in cache:
            return
        try:
            cache[key] = rewrite_query(question, client, LLM_MODEL, MAX_TOKENS)
        except QueryRewriteError as e:
            print(f"Rewrite failed, using the original ({e}): {question}")
            cache[key] = None

    jobs = [(q, i) for q in questions for i in range(SAMPLES)]
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(ask, jobs))

    path.write_text(json.dumps(cache, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return cache


def rank_of(similarities, chunks, expected_id):
    """Position (1 = best) of the first chunk that holds the expected paragraph."""
    for position, index in enumerate(np.argsort(-similarities), start=1):
        if expected_id in chunks[index]["paragraph_ids"]:
            return position
    return len(chunks) + 1


def metrics(ranks):
    ranks = np.array(ranks)
    return (ranks == 1).mean(), (ranks <= 3).mean(), (1 / ranks).mean()


def sign_test(better, worse):
    """Two-sided sign test. Ignores that the three samples of a question are linked."""
    n = better + worse
    if n == 0:
        return 1.0
    k = max(better, worse)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n)


def main():
    sets = load_sets()
    all_questions = sorted({q["question"] for items in sets.values() for q in items})
    samples = collect_samples(all_questions, DATA / "rewrites" / f"{LLM_MODEL}-samples.json")

    model = load_model()
    handbook = read("corpus/qa_handbook.json")["documents"]
    distractors = read("corpus/qa_distractors.json")["documents"]

    for corpus_name, documents in [
        ("handbook only", handbook),
        ("handbook + 15 unrelated paragraphs", handbook + distractors),
    ]:
        chunks = chunk_documents(documents, max_chars=MAX_CHARS)
        chunk_vectors = embed(model, [c["text"] for c in chunks])
        print(f"\n===== {corpus_name}: {len(chunks)} chunks =====")

        for set_name, items in sets.items():
            ranks = {"original": [], **{f"rewrite {i + 1}": [] for i in range(SAMPLES)}}
            worsened = []

            for item in items:
                question, expected_id = item["question"], item["expected_id"]
                original = rank_of(chunk_vectors @ embed(model, [question])[0], chunks, expected_id)
                ranks["original"].append(original)

                for i in range(SAMPLES):
                    rewrite = samples[f"{i}|{question}"] or question
                    r = rank_of(chunk_vectors @ embed(model, [rewrite])[0], chunks, expected_id)
                    ranks[f"rewrite {i + 1}"].append(r)
                    if i == 0 and r > original:
                        worsened.append((question, rewrite, original, r))

            print(f"\n{set_name}        Hit@1  Hit@3    MRR")
            for name, values in ranks.items():
                hit_1, hit_3, mrr = metrics(values)
                print(f"  {name:<10} {hit_1:>9.2f} {hit_3:>6.2f} {mrr:>6.3f}")

            mean = np.mean([metrics(ranks[f"rewrite {i + 1}"]) for i in range(SAMPLES)], axis=0)
            print(f"  rewrite mean {mean[0]:>7.2f} {mean[1]:>6.2f} {mean[2]:>6.3f}")

            better = worse = 0
            for i in range(SAMPLES):
                for a, b in zip(ranks["original"], ranks[f"rewrite {i + 1}"], strict=True):
                    better += b < a
                    worse += b > a
            print(f"  rank improved {better}, worsened {worse} (sign test p={sign_test(better, worse):.3f})")

            for question, rewrite, before, after in worsened:
                print(f"  worse (rank {before} -> {after}): {question}\n    rewrite: {rewrite}")


if __name__ == "__main__":
    main()
