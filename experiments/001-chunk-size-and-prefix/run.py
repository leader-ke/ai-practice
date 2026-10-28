"""Experiment 001: chunk size and document prefix."""

import json
from pathlib import Path

from retrieval.chunking import chunk_documents
from retrieval.embeddings import load_model
from retrieval.evaluation import evaluate

DATA = Path(__file__).resolve().parents[2] / "data"

documents = json.loads((DATA / "corpus" / "qa_handbook.json").read_text())["documents"]
golden_set = json.loads((DATA / "golden_sets" / "qa_retrieval.json").read_text())["questions"]

model = load_model()

for max_chars in [250, 400, 600]:
    chunks = chunk_documents(documents, max_chars=max_chars)

    for add_prefix in [False, True]:
        results = evaluate(
            chunks,
            golden_set,
            model,
            add_prefix=add_prefix,
        )

        mode = "with prefixes" if add_prefix else "without prefixes"

        print(f"\nSize: {max_chars} | {mode}")
        print(f"Chunks: {len(chunks)}")
        print(f"Hit@1: {results['hit_at_1']}/{results['total']}")
        print(f"Recall@3: {results['recall_at_3']}/{results['total']}")

        for miss in results["misses"]:
            print(f"\nMissed: {miss['question']}")
            for item in miss["ranked"]:
                print(
                    f"  {item['rank']}. {item['score']:.3f} | "
                    f"{item['document']} | "
                    f"{item['paragraph_ids']} | "
                    f"correct={item['correct']}"
                )
