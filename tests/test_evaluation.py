import numpy as np

from retrieval.embeddings import text_to_embed
from retrieval.evaluation import evaluate

CHUNKS = [
    {"document": "Doc", "paragraph_ids": ["a"], "text": "alpha"},
    {"document": "Doc", "paragraph_ids": ["b"], "text": "beta"},
    {"document": "Doc", "paragraph_ids": ["c"], "text": "gamma"},
]


class FakeModel:
    """Returns fixed vectors for known texts, so the ranking is predictable. Never loads a real model."""

    def __init__(self, vectors):
        self.vectors = vectors
        self.encoded = []

    def encode(self, texts, normalize_embeddings=True):
        self.encoded.append(list(texts))
        return np.array([self.vectors[text] for text in texts], dtype=float)


VECTORS = {
    "alpha": [1, 0, 0],
    "beta": [0, 1, 0],
    "gamma": [0, 0, 1],
    "about alpha": [1, 0, 0],
    "about beta but closer to gamma": [0, 0.6, 0.8],
}


def test_counts_hits_and_reports_the_misses():
    golden_set = [
        {"question": "about alpha", "expected_id": "a"},
        {"question": "about beta but closer to gamma", "expected_id": "b"},
    ]

    results = evaluate(CHUNKS, golden_set, FakeModel(VECTORS))

    assert results["total"] == 2
    assert results["hit_at_1"] == 1
    assert results["recall_at_3"] == 2

    (miss,) = results["misses"]
    assert miss["question"] == "about beta but closer to gamma"
    assert [item["paragraph_ids"] for item in miss["ranked"]] == [["c"], ["b"], ["a"]]
    assert [item["correct"] for item in miss["ranked"]] == [False, True, False]


def test_recall_at_3_cannot_fail_with_only_three_chunks():
    golden_set = [{"question": "about beta but closer to gamma", "expected_id": "a"}]

    results = evaluate(CHUNKS, golden_set, FakeModel(VECTORS))

    assert results["hit_at_1"] == 0
    assert results["recall_at_3"] == 1


def test_prefix_is_added_only_when_asked():
    chunk = CHUNKS[0]

    assert text_to_embed(chunk) == "alpha"
    assert text_to_embed(chunk, add_prefix=True) == "Document: Doc\nalpha"


def test_the_prefixed_text_is_what_gets_embedded():
    unit_vectors = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    prefixed = {
        text_to_embed(chunk, add_prefix=True): vector for chunk, vector in zip(CHUNKS, unit_vectors, strict=True)
    }
    model = FakeModel({**prefixed, "about alpha": [1, 0, 0]})

    results = evaluate(CHUNKS, [{"question": "about alpha", "expected_id": "a"}], model, add_prefix=True)

    assert model.encoded[0] == list(prefixed)
    assert results["hit_at_1"] == 1
