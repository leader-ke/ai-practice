"""Experiment 003: does rewriting the question help retrieval?

Run experiment 002 first, so the chunks are in the database.

Rewrites are saved in data/rewrites/<model>.json and reused on later runs,
so the results do not change between runs. Delete the file to rewrite again.
"""

import json
from pathlib import Path

import anthropic

from config import ANTHROPIC_API_KEY, DATABASE_URL, LLM_MODEL, MAX_TOKENS
from retrieval.embeddings import embed, load_model
from retrieval.search import connect, search
from rewriting.query_rewriter import QueryRewriteError, rewrite_query

DATA = Path(__file__).resolve().parents[2] / "data"

CONFIG = "400-noprefix"
K = 3


def load_questions():
    """The original five questions, then the colloquial ones split in two."""
    original = json.loads((DATA / "golden_sets" / "qa_retrieval.json").read_text(encoding="utf-8"))["questions"]
    colloquial = json.loads((DATA / "golden_sets" / "qa_retrieval_colloquial.json").read_text(encoding="utf-8"))[
        "questions"
    ]

    return {
        "original (5)": original,
        "colloquial tune": [q for q in colloquial if q["split"] == "tune"],
        "colloquial heldout": [q for q in colloquial if q["split"] == "heldout"],
    }


def load_rewrites(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def get_rewrite(question, rewrites, client):
    """Returns the saved rewrite, or asks the model. None if it fails."""
    if question in rewrites:
        return rewrites[question]

    try:
        rewrite = rewrite_query(
            question=question,
            client=client,
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
        )
    except QueryRewriteError as e:
        print(f"  Rewrite failed ({e}), using the original: {question}")
        return None

    rewrites[question] = rewrite
    return rewrite


def score(conn, model, query, expected_id):
    """Returns (hit_at_1, hit_at_3) for one query."""
    results = search(conn, embed(model, [query])[0], CONFIG, K)

    if not results:
        raise RuntimeError("No chunks found. Run experiment 002 first.")

    hit_1 = expected_id in results[0][1]
    hit_3 = any(expected_id in paragraph_ids for _, paragraph_ids, _, _ in results)

    return hit_1, hit_3


def main():
    questions = load_questions()
    rewrites_path = DATA / "rewrites" / f"{LLM_MODEL}.json"
    rewrites_path.parent.mkdir(exist_ok=True)
    rewrites = load_rewrites(rewrites_path)

    model = load_model()
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    conn = connect(DATABASE_URL)

    try:
        print(f"Rewrite model: {LLM_MODEL}")
        print(f"Saved rewrites: {len(rewrites)} ({rewrites_path.name})")

        summary = []

        for set_name, items in questions.items():
            totals = {"Original": [0, 0], "Rewritten": [0, 0]}
            changed = []

            for item in items:
                question = item["question"]
                expected_id = item["expected_id"]

                rewrite = get_rewrite(question, rewrites, client)
                rewrites_path.write_text(
                    json.dumps(rewrites, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

                queries = {"Original": question, "Rewritten": rewrite or question}

                outcome = {}
                for approach, query in queries.items():
                    hit_1, hit_3 = score(conn, model, query, expected_id)
                    totals[approach][0] += int(hit_1)
                    totals[approach][1] += int(hit_3)
                    outcome[approach] = (hit_1, hit_3)

                if outcome["Original"] != outcome["Rewritten"]:
                    changed.append((question, queries["Rewritten"], outcome))

            summary.append((set_name, len(items), totals))

            print(f"\n{set_name}: questions where rewriting changed the result")
            print("-" * 60)
            if not changed:
                print("none")
            for question, rewrite, outcome in changed:
                print(f"\n{question}")
                print(f"  rewrite: {rewrite}")
                for approach, (hit_1, hit_3) in outcome.items():
                    print(f"  {approach:<10} Hit@1={hit_1}  Hit@3={hit_3}")

        print("\nSummary")
        print("-------")
        print(f"{'Set':<20} {'Approach':<10} {'Hit@1':<8} Hit@3")
        for set_name, total, totals in summary:
            for approach, (hit_1, hit_3) in totals.items():
                print(f"{set_name:<20} {approach:<10} {hit_1}/{total:<6} {hit_3}/{total}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
