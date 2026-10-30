import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data"

QUESTION_FILES = [
    "qa_retrieval.json",
    "qa_retrieval_colloquial.json",
    "qa_retrieval_blind.json",
    "qa_retrieval_realistic.json",
]


def read(path):
    return json.loads((DATA / path).read_text(encoding="utf-8"))


def paragraph_ids(corpus_file):
    return [p["id"] for doc in read(corpus_file)["documents"] for p in doc["paragraphs"]]


@pytest.mark.parametrize("filename", QUESTION_FILES)
def test_every_expected_id_exists_in_the_handbook(filename):
    known = set(paragraph_ids("corpus/qa_handbook.json"))

    for item in read(f"golden_sets/{filename}")["questions"]:
        assert item["expected_id"] in known, item["question"]


@pytest.mark.parametrize("filename", QUESTION_FILES)
def test_questions_are_not_empty_or_repeated(filename):
    questions = [q["question"] for q in read(f"golden_sets/{filename}")["questions"]]

    assert all(q.strip() for q in questions)
    assert len(questions) == len(set(questions))


@pytest.mark.parametrize("filename", QUESTION_FILES[1:])
def test_every_paragraph_has_at_least_one_question(filename):
    covered = {q["expected_id"] for q in read(f"golden_sets/{filename}")["questions"]}

    assert covered == set(paragraph_ids("corpus/qa_handbook.json"))


def test_colloquial_questions_are_split_into_tune_and_heldout():
    splits = [q["split"] for q in read("golden_sets/qa_retrieval_colloquial.json")["questions"]]

    assert set(splits) == {"tune", "heldout"}
    assert splits.count("tune") == 18
    assert splits.count("heldout") == 9


def test_distractors_do_not_reuse_handbook_ids():
    handbook = set(paragraph_ids("corpus/qa_handbook.json"))
    distractors = paragraph_ids("corpus/qa_distractors.json")

    assert not handbook & set(distractors)
    assert len(distractors) == len(set(distractors))


def test_saved_rewrites_are_single_lines():
    for path in (DATA / "rewrites").glob("*.json"):
        for rewrite in json.loads(path.read_text(encoding="utf-8")).values():
            assert rewrite is None or ("\n" not in rewrite and rewrite.strip())
