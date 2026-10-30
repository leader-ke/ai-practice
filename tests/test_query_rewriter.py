from types import SimpleNamespace
from unittest.mock import Mock

import anthropic
import httpx
import pytest

from rewriting.query_rewriter import (
    SYSTEM_PROMPT,
    QueryRewriteError,
    rewrite_or_original,
    rewrite_query,
)


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def make_response(text, stop_reason="end_turn"):
    return SimpleNamespace(
        content=[text_block(text)],
        stop_reason=stop_reason,
    )


def make_client(response):
    client = Mock()
    client.messages.create.return_value = response
    return client


def rewrite(client, question="How do I write a bug report?"):
    return rewrite_query(question=question, client=client, model="test-model")


def test_rewrite_query_returns_rewritten_question():
    client = make_client(make_response("What sensitive information must not appear in bug reports?"))

    result = rewrite_query(
        question="How should QA testers protect sensitive information?",
        client=client,
        model="test-model",
    )

    assert result == "What sensitive information must not appear in bug reports?"
    client.messages.create.assert_called_once()


def test_rewrite_query_strips_whitespace():
    client = make_client(make_response("  What information should be excluded from bug reports?  "))

    assert rewrite(client) == "What information should be excluded from bug reports?"


@pytest.mark.parametrize(
    "raw",
    [
        '"What goes in a bug report?"',
        "'What goes in a bug report?'",
        "“What goes in a bug report?”",
    ],
)
def test_rewrite_query_removes_surrounding_quotes(raw):
    assert rewrite(make_client(make_response(raw))) == "What goes in a bug report?"


def test_rewrite_query_sends_the_expected_request():
    client = make_client(make_response("A rewritten question?"))

    rewrite_query(
        question="  How do I write a bug report?  ",
        client=client,
        model="test-model",
        max_tokens=123,
    )

    client.messages.create.assert_called_once_with(
        model="test-model",
        max_tokens=123,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "How do I write a bug report?"}],
    )


def test_rewrite_query_ignores_thinking_blocks():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="Let me think."),
            text_block("What goes in a bug report?"),
        ],
        stop_reason="end_turn",
    )

    assert rewrite(make_client(response)) == "What goes in a bug report?"


def test_rewrite_query_rejects_empty_question():
    client = Mock()

    with pytest.raises(ValueError):
        rewrite_query(question=" ", client=client, model="test-model")

    client.messages.create.assert_not_called()


def test_rewrite_query_rejects_missing_model():
    client = Mock()

    with pytest.raises(ValueError):
        rewrite_query(
            question="How do I write a bug report?",
            client=client,
            model="",
        )

    client.messages.create.assert_not_called()


def test_rewrite_query_rejects_empty_model_response():
    with pytest.raises(QueryRewriteError):
        rewrite(make_client(make_response(" ")))


def test_rewrite_query_rejects_response_cut_off_by_token_limit():
    client = make_client(make_response("What goes in a", stop_reason="max_tokens"))

    with pytest.raises(QueryRewriteError, match="cut off"):
        rewrite(client)


def test_rewrite_query_rejects_multiple_lines():
    client = make_client(make_response("A question?\nHere is why I changed it."))

    with pytest.raises(QueryRewriteError, match="more than one line"):
        rewrite(client)


def test_rewrite_query_rejects_overlong_output():
    client = make_client(make_response("word " * 100))

    with pytest.raises(QueryRewriteError, match="longer than expected"):
        rewrite(client)


def test_rewrite_query_translates_sdk_errors():
    client = Mock()
    client.messages.create.side_effect = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com")
    )

    with pytest.raises(QueryRewriteError) as error:
        rewrite(client)

    assert isinstance(error.value.__cause__, anthropic.APIConnectionError)


def test_rewrite_or_original_returns_the_rewrite_when_it_works():
    client = make_client(make_response("A rewritten question?"))

    result = rewrite_or_original(
        question="vague question",
        client=client,
        model="test-model",
    )

    assert result == "A rewritten question?"


def test_rewrite_or_original_falls_back_when_rewriting_fails():
    client = make_client(make_response("cut off", stop_reason="max_tokens"))

    result = rewrite_or_original(
        question="  vague question  ",
        client=client,
        model="test-model",
    )

    assert result == "vague question"


def test_rewrite_or_original_still_rejects_invalid_input():
    with pytest.raises(ValueError):
        rewrite_or_original(question=" ", client=Mock(), model="test-model")
