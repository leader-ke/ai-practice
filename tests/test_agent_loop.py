"""Tests for the agent loop itself, using a fake Anthropic client. No network, no key, no cost."""

import logging
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from agent import AgentConfigError, AgentExecutionError, AgentTimeoutError, AgentUnavailableError, loop, run_agent
from agent.tools import TOOL_MAPPING

REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(block_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def response(stop_reason, *blocks, input_tokens=10, output_tokens=5):
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks), usage=usage)


class FakeClient:
    """Plays back scripted responses (or raises scripted errors) and records the messages it was sent."""

    def __init__(self, *scripted):
        self.scripted = list(scripted)
        self.sent = []
        self.messages = self

    def create(self, **kwargs):
        self.sent.append(list(kwargs["messages"]))
        item = self.scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def fake_client(monkeypatch):
    def install(*scripted):
        client = FakeClient(*scripted)
        monkeypatch.setattr(loop, "get_client", lambda: client)
        return client

    return install


def tool_results_sent(client, call_index):
    """The tool_result blocks in the last message of a given call."""
    return client.sent[call_index][-1]["content"]


def test_returns_the_text_when_no_tool_is_needed(fake_client):
    client = fake_client(response("end_turn", text_block("2 + 2 = 4.")))

    assert run_agent("What is 2 + 2?") == "2 + 2 = 4."
    assert len(client.sent) == 1


def test_runs_a_tool_and_sends_the_result_back(fake_client):
    tool_request = response("tool_use", tool_block("t1", "lookup_ticket", {"ticket_id": "1234567890"}))
    client = fake_client(tool_request, response("end_turn", text_block("It is open.")))

    assert run_agent("Status of 1234567890?") == "It is open."

    second_call = client.sent[1]
    assert [m["role"] for m in second_call] == ["user", "assistant", "user"]
    assert second_call[1]["content"] is tool_request.content
    assert tool_results_sent(client, 1) == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "Ticket status is: open", "is_error": False}
    ]


def test_two_tool_requests_get_two_results_in_one_message(fake_client):
    two_requests = response(
        "tool_use",
        tool_block("t1", "lookup_ticket", {"ticket_id": "1234567890"}),
        tool_block("t2", "lookup_ticket", {"ticket_id": "1234567891"}),
    )
    client = fake_client(two_requests, response("end_turn", text_block("open and closed")))

    run_agent("Statuses?")

    results = tool_results_sent(client, 1)
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert [r["content"] for r in results] == ["Ticket status is: open", "Ticket status is: closed"]


def test_a_missing_ticket_is_reported_as_an_error_result(fake_client):
    request = response("tool_use", tool_block("t1", "lookup_ticket", {"ticket_id": "99999"}))
    client = fake_client(request, response("end_turn", text_block("Not found.")))

    run_agent("Status of 99999?")

    (result,) = tool_results_sent(client, 1)
    assert result["is_error"] is True
    assert "not found" in result["content"]


@pytest.mark.parametrize(
    "name, tool_input",
    [
        ("no_such_tool", {"ticket_id": "1"}),
        ("lookup_ticket", "not an object"),
        ("lookup_ticket", {"wrong_field": "1"}),
    ],
)
def test_bad_tool_requests_become_error_results(fake_client, name, tool_input):
    request = response("tool_use", tool_block("t1", name, tool_input))
    client = fake_client(request, response("end_turn", text_block("ok")))

    run_agent("question")

    (result,) = tool_results_sent(client, 1)
    assert result["is_error"] is True


def test_a_tool_that_raises_does_not_crash_the_loop(fake_client, monkeypatch):
    def broken_tool(ticket_id):
        raise RuntimeError("database is down")

    monkeypatch.setitem(TOOL_MAPPING, "lookup_ticket", broken_tool)
    client = fake_client(
        response("tool_use", tool_block("t1", "lookup_ticket", {"ticket_id": "1"})),
        response("end_turn", text_block("Sorry, the lookup failed.")),
    )

    assert run_agent("question") == "Sorry, the lookup failed."
    (result,) = tool_results_sent(client, 1)
    assert result["is_error"] is True


def test_gives_up_after_the_round_cap(fake_client):
    ask_again = response("tool_use", tool_block("t1", "lookup_ticket", {"ticket_id": "1234567890"}))
    endless = [ask_again] * 10
    client = fake_client(*endless)

    with pytest.raises(AgentExecutionError):
        run_agent("question")

    assert loop.MAX_ROUNDS == 5
    assert len(client.sent) == 5


def test_a_cut_off_answer_is_an_error(fake_client):
    fake_client(response("max_tokens", text_block("The answer is cut o")))

    with pytest.raises(AgentExecutionError):
        run_agent("question")


def test_an_unexpected_stop_reason_is_an_error(fake_client):
    fake_client(response("refusal"))

    with pytest.raises(AgentExecutionError):
        run_agent("question")


def status_error(error_class, status_code):
    return error_class("boom", response=httpx.Response(status_code, request=REQUEST), body=None)


@pytest.mark.parametrize(
    "sdk_error, expected",
    [
        (status_error(anthropic.AuthenticationError, 401), AgentConfigError),
        (status_error(anthropic.RateLimitError, 429), AgentUnavailableError),
        (anthropic.APITimeoutError(request=REQUEST), AgentTimeoutError),
        (anthropic.APIConnectionError(request=REQUEST), AgentUnavailableError),
        (status_error(anthropic.APIStatusError, 500), AgentUnavailableError),
        (status_error(anthropic.APIStatusError, 400), AgentConfigError),
        (ValueError("surprise"), AgentExecutionError),
    ],
)
def test_sdk_errors_are_translated_and_keep_their_cause(fake_client, sdk_error, expected):
    fake_client(sdk_error)

    with pytest.raises(expected) as caught:
        run_agent("question")

    assert caught.value.__cause__ is sdk_error


def test_the_client_gets_the_key_and_a_timeout(monkeypatch):
    created = {}

    def fake_anthropic(**kwargs):
        created.update(kwargs)
        return object()

    monkeypatch.setattr(loop.anthropic, "Anthropic", fake_anthropic)
    loop.get_client.cache_clear()
    try:
        loop.get_client()
    finally:
        loop.get_client.cache_clear()

    assert created == {"api_key": "test-key", "timeout": loop.REQUEST_TIMEOUT_SECONDS}


def test_token_usage_is_logged(fake_client, caplog):
    fake_client(response("end_turn", text_block("hi"), input_tokens=18, output_tokens=7))

    with caplog.at_level(logging.INFO, logger="agent.loop"):
        run_agent("question")

    assert "agent finished: rounds=1 input_tokens=18 output_tokens=7" in caplog.text
