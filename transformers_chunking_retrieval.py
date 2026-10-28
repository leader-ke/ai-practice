import numpy as np
from sentence_transformers import SentenceTransformer


def chunk_documents(documents, max_chars):
    chunks = []
    for document in documents:
        current_paragraphs = []
        current_length = 0

        for paragraph in document["paragraphs"]:
            text = paragraph["text"]
            paragraph_id = paragraph["id"]

            added_length = len(text) + (2 if current_paragraphs else 0)

            if current_paragraphs and current_length + added_length > max_chars:
                chunks.append(
                    {
                        "document": document["name"],
                        "paragraph_ids": [p["id"] for p in current_paragraphs],
                        "text": "\n\n".join(p["text"] for p in current_paragraphs),
                    }
                )
                current_paragraphs = []
                current_length = 0

            if len(text) > max_chars:
                words = text.split()
                pieces = []
                piece = ""

                for word in words:
                    candidate = f"{piece} {word}".strip()

                    if len(candidate) > max_chars and piece:
                        pieces.append(piece)
                        piece = word
                    else:
                        piece = candidate

                if piece:
                    pieces.append(piece)

                for piece in pieces:
                    chunks.append(
                        {
                            "document": document["name"],
                            "paragraph_ids": [paragraph_id],
                            "text": piece,
                        }
                    )

                continue

            current_paragraphs.append(paragraph)
            current_length += added_length

        if current_paragraphs:
            chunks.append(
                {
                    "document": document["name"],
                    "paragraph_ids": [p["id"] for p in current_paragraphs],
                    "text": "\n\n".join(p["text"] for p in current_paragraphs),
                }
            )

    return chunks


model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    {
        "name": "Bug Severity Guide",
        "paragraphs": [
            {
                "id": "severity_1",
                "text": (
                    "Severity describes the impact of a defect on users, "
                    "business operations, and system functionality. "
                    "Severity is different from priority, which determines "
                    "how urgently a defect should be addressed."
                ),
            },
            {
                "id": "severity_2",
                "text": (
                    "Critical severity means a core service is unavailable, "
                    "important data is lost or corrupted, or a major security "
                    "failure occurs. There is no reasonable workaround, "
                    "and immediate investigation is required."
                ),
            },
            {
                "id": "severity_3",
                "text": (
                    "High severity means an important feature is broken, "
                    "but the rest of the application may still work. "
                    "Medium severity covers limited functionality issues "
                    "with a workaround. Low severity covers minor cosmetic "
                    "issues or small inconveniences."
                ),
            },
        ],
    },
    {
        "name": "Bug Report Template",
        "paragraphs": [
            {
                "id": "report_1",
                "text": (
                    "A useful bug report includes a concise title, "
                    "environment, application version, severity, and "
                    "preconditions. The title should describe the "
                    "observable problem rather than suggest an unverified "
                    "root cause."
                ),
            },
            {
                "id": "report_2",
                "text": (
                    "Steps to reproduce should be numbered and repeatable. "
                    "Expected results describe what should happen, while "
                    "actual results describe what happened. Include "
                    "screenshots, logs, or a recording when they help "
                    "demonstrate the failure."
                ),
            },
            {
                "id": "report_3",
                "text": (
                    "Before submitting a report, check whether the issue "
                    "already exists. Record the frequency, affected users "
                    "or workflows, and any known workaround. Do not include "
                    "passwords, access tokens, personal data, or other "
                    "secrets in screenshots and logs."
                ),
            },
        ],
    },
    {
        "name": "Test Environment Notes",
        "paragraphs": [
            {
                "id": "environment_1",
                "text": (
                    "A test environment should be identifiable by its "
                    "environment name, application build, configuration, "
                    "and deployment version. Record these details when "
                    "reporting a defect so another tester can reproduce "
                    "the same conditions."
                ),
            },
            {
                "id": "environment_2",
                "text": (
                    "Test data should be controlled and repeatable. "
                    "Use synthetic accounts and data where possible. "
                    "Reset data between tests when previous runs could "
                    "affect results, and never copy production secrets "
                    "into a test environment."
                ),
            },
            {
                "id": "environment_3",
                "text": (
                    "When a test fails, check service availability, "
                    "network connectivity, configuration, test data, "
                    "and application logs. Distinguish an application "
                    "defect from an environment problem before assigning "
                    "a final root cause."
                ),
            },
        ],
    },
]
golden_set = [
    {
        "question": "What is the difference between severity and priority?",
        "expected_id": "severity_1",
    },
    {
        "question": "When should a defect be classified as critical?",
        "expected_id": "severity_2",
    },
    {
        "question": "What should steps to reproduce and expected results contain?",
        "expected_id": "report_2",
    },
    {
        "question": "How should QA testers protect sensitive information in bug reports?",
        "expected_id": "report_3",
    },
    {
        "question": "What should a tester check when a test fails because of the environment?",
        "expected_id": "environment_3",
    },
]


def evaluate(chunks, golden_set, model, add_prefix=False):
    texts = [f"Document: {chunk['document']}\n{chunk['text']}" if add_prefix else chunk["text"] for chunk in chunks]

    chunk_vectors = model.encode(texts, normalize_embeddings=True)
    question_vectors = model.encode([item["question"] for item in golden_set], normalize_embeddings=True)

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
