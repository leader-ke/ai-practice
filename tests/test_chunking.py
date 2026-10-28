from retrieval.chunking import chunk_documents


def document(*texts, name="Doc"):
    return {"name": name, "paragraphs": [{"id": f"p{i}", "text": text} for i, text in enumerate(texts, start=1)]}


def test_groups_whole_paragraphs_up_to_the_limit():
    chunks = chunk_documents([document("a" * 10, "b" * 10, "c" * 10)], max_chars=25)

    assert [c["paragraph_ids"] for c in chunks] == [["p1", "p2"], ["p3"]]
    assert chunks[0]["text"] == "a" * 10 + "\n\n" + "b" * 10


def test_a_chunk_can_be_exactly_the_limit():
    chunks = chunk_documents([document("a" * 10, "b" * 10)], max_chars=22)

    assert [c["paragraph_ids"] for c in chunks] == [["p1", "p2"]]


def test_chunks_never_mix_documents():
    chunks = chunk_documents([document("one", name="First"), document("two", name="Second")], max_chars=1000)

    assert [(c["document"], c["paragraph_ids"]) for c in chunks] == [("First", ["p1"]), ("Second", ["p1"])]


def test_an_oversized_paragraph_is_split_on_word_boundaries():
    text = "alpha beta gamma delta epsilon zeta"
    chunks = chunk_documents([document(text)], max_chars=12)

    pieces = [c["text"] for c in chunks]
    assert pieces == ["alpha beta", "gamma delta", "epsilon zeta"]
    assert " ".join(pieces) == text
    assert all(c["paragraph_ids"] == ["p1"] for c in chunks)


def test_a_single_word_longer_than_the_limit_is_kept_whole():
    chunks = chunk_documents([document("supercalifragilistic")], max_chars=5)

    assert [c["text"] for c in chunks] == ["supercalifragilistic"]


def test_paragraphs_before_an_oversized_one_are_emitted_first():
    chunks = chunk_documents([document("short", "one two three four five six")], max_chars=12)

    assert [c["paragraph_ids"] for c in chunks] == [["p1"], ["p2"], ["p2"], ["p2"]]
    assert chunks[0]["text"] == "short"


def test_no_documents_gives_no_chunks():
    assert chunk_documents([], max_chars=100) == []
