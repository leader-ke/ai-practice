"""Experiment 002: PostgreSQL pgvector search."""

import json
from pathlib import Path

from config import DATABASE_URL
from retrieval.chunking import chunk_documents
from retrieval.embeddings import embed, load_model
from retrieval.search import connect, search, store_chunks

DATA = Path(__file__).resolve().parents[2] / "data"

CONFIG = "400-noprefix"
MAX_CHARS = 400
K = 3


documents = json.loads((DATA / "corpus" / "qa_handbook.json").read_text())["documents"]

golden_set = json.loads((DATA / "golden_sets" / "qa_retrieval.json").read_text())["questions"]


model = load_model()


chunks = chunk_documents(documents, max_chars=MAX_CHARS)


vectors = embed(
    model,
    [chunk["text"] for chunk in chunks],
)


conn = connect(DATABASE_URL)

try:
    store_chunks(conn, chunks, vectors, CONFIG)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT config, count(*)
            FROM chunks
            GROUP BY config
            """
        )

        print("\nDatabase counts:")
        for row in cur.fetchall():
            print(row)

    hit_at_1 = 0
    recall_at_3 = 0
    misses = []

    for item in golden_set:
        question = item["question"]
        expected_id = item["expected_id"]

        query_vector = embed(model, [question])[0]

        results = search(
            conn,
            query_vector,
            CONFIG,
            K,
        )

        hit_1 = bool(results) and expected_id in results[0][1]

        hit_3 = any(expected_id in result[1] for result in results)

        hit_at_1 += int(hit_1)
        recall_at_3 += int(hit_3)

        if not hit_1:
            misses.append(
                {
                    "question": question,
                    "expected_id": expected_id,
                    "ranked": [
                        {
                            "rank": rank,
                            "score": 1 - distance,
                            "document": document,
                            "paragraph_ids": paragraph_ids,
                            "correct": expected_id in paragraph_ids,
                        }
                        for rank, (
                            document,
                            paragraph_ids,
                            text,
                            distance,
                        ) in enumerate(results, start=1)
                    ],
                }
            )

    total = len(golden_set)

    print("\nPostgreSQL evaluation")
    print("---------------------")
    print(f"Chunks: {len(chunks)}")
    print(f"Hit@1: {hit_at_1}/{total}")
    print(f"Recall@3: {recall_at_3}/{total}")

    for miss in misses:
        print(f"\nMissed: {miss['question']}")

        for result in miss["ranked"]:
            print(
                f"  {result['rank']}. {result['score']:.3f} | "
                f"{result['document']} | "
                f"{result['paragraph_ids']} | "
                f"correct={result['correct']}"
            )

    target_question = "How should QA testers protect sensitive information in bug reports?"

    query_vector = embed(model, [target_question])[0]

    top_results = search(
        conn,
        query_vector,
        CONFIG,
        K,
    )

    print("\nSensitive-information similarity check")
    print("---------------------------------------")

    for document, paragraph_ids, _text, distance in top_results:
        print(f"Similarity: {1 - distance:.3f} | {document} | {paragraph_ids}")

    # Test alternative query wording
    test_queries = [
        "What sensitive information must never be included in bug reports?",
        "Should passwords, access tokens, or personal data appear in screenshots or logs?",
        "What security precautions should testers follow when attaching screenshots and logs to bug reports?",
    ]

    expected_id = "report_3"

    print("\nQuery wording experiment")
    print("------------------------")

    for question in test_queries:
        query_vector = embed(model, [question])[0]
        results = search(conn, query_vector, CONFIG, K)

        wording_hit_1 = bool(results) and expected_id in results[0][1]
        wording_hit_3 = any(expected_id in result[1] for result in results)

        print(f"\nQuestion: {question}")
        print(f"Hit@1: {wording_hit_1} | Hit@3: {wording_hit_3}")

        for rank, (document, paragraph_ids, _text, distance) in enumerate(results, start=1):
            similarity = 1 - distance
            correct = expected_id in paragraph_ids

            print(f"{rank}. Similarity: {similarity:.3f} | {document} | {paragraph_ids} | correct={correct}")
finally:
    conn.close()
