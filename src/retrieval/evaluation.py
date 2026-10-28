import numpy as np

from retrieval.embeddings import embed, text_to_embed


def evaluate(chunks, golden_set, model, add_prefix=False):
    texts = [text_to_embed(chunk, add_prefix) for chunk in chunks]

    chunk_vectors = embed(model, texts)
    question_vectors = embed(model, [item["question"] for item in golden_set])

    scores_matrix = question_vectors @ chunk_vectors.T
    hits_at_1 = 0
    hits_at_3 = 0
    misses = []

    for q_index, item in enumerate(golden_set):
        scores = scores_matrix[q_index]
        ranked_indices = np.argsort(scores)[::-1]

        expected_id = item["expected_id"]

        def is_expected(index, _expected_id=expected_id):
            return _expected_id in chunks[int(index)]["paragraph_ids"]

        hit_1 = is_expected(ranked_indices[0])
        hit_3 = any(is_expected(i) for i in ranked_indices[:3])

        hits_at_1 += int(hit_1)
        hits_at_3 += int(hit_3)

        if not hit_1:
            misses.append(
                {
                    "question": item["question"],
                    "expected_id": item["expected_id"],
                    "ranked": [
                        {
                            "rank": rank,
                            "score": float(scores[index]),
                            "document": chunks[int(index)]["document"],
                            "paragraph_ids": chunks[int(index)]["paragraph_ids"],
                            "correct": is_expected(index),
                        }
                        for rank, index in enumerate(ranked_indices, start=1)
                    ],
                }
            )

    total = len(golden_set)
    return {
        "hit_at_1": hits_at_1,
        "recall_at_3": hits_at_3,
        "total": total,
        "misses": misses,
    }
