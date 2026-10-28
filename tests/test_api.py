import pytest
from fastapi.testclient import TestClient

from agent import AgentConfigError, AgentError, AgentExecutionError, AgentTimeoutError, AgentUnavailableError
from api import main

client = TestClient(main.app)


@pytest.mark.parametrize(
    "raised_error, expected_status, expected_generic_text",
    [
        (AgentConfigError("SECRET_KEY_EXPOSED_123"), 500, "Internal server configuration error."),
        (AgentUnavailableError("OVERLOADED_CLUSTER_XYZ"), 503, "Service temporarily unavailable. Try again later."),
        (AgentTimeoutError("TIMEOUT_GATEWAY_ABC"), 504, "Gateway timeout. Upstream service failed to respond."),
        (AgentExecutionError("INFINITY_LOOP_ALERT_999"), 500, "An error occurred processing your request."),
        (AgentError("BASE_CLASS_INTERCEPTED"), 500, "An internal agent process failed."),
        (ValueError("RAW_PYTHON_CRASH_VALUE"), 500, "An unexpected system error occurred."),
    ],
)
def test_error_mapping_and_leak_prevention(monkeypatch, raised_error, expected_status, expected_generic_text):
    """Test that errors map to correct status codes and NEVER leak internal messages."""

    def fake_raise(_question):
        raise raised_error

    monkeypatch.setattr(main, "run_agent", fake_raise)
    response = client.post("/ask", json={"question": "hi"})

    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_generic_text
    assert "SECRET" not in response.text
    assert "OVERLOADED" not in response.text
    assert "TIMEOUT" not in response.text
    assert "INFINITY" not in response.text
    assert "BASE" not in response.text
    assert "RAW" not in response.text


@pytest.mark.parametrize(
    "payload, expected_status, description",
    [
        ({"question": ""}, 422, "Empty string"),
        ({}, 422, "Missing field"),
        ({"question": 12345}, 422, "Wrong data type"),
        ({"question": "a" * 2001}, 422, "Just past maximum limit (2001 chars)"),
        ({"question": "a"}, 200, "Minimum valid boundary (1 char)"),
        ({"question": "a" * 2000}, 200, "Maximum valid boundary (2000 chars)"),
    ],
)
def test_validation_boundaries_and_execution_guard(monkeypatch, payload, expected_status, description):
    """Test Pydantic bounds and verify the agent loop is never invoked for 422 failures."""
    calls = []

    def fake_recording_agent(question):
        calls.append(question)
        return "fake answer"

    monkeypatch.setattr(main, "run_agent", fake_recording_agent)
    response = client.post("/ask", json=payload)

    assert response.status_code == expected_status

    if expected_status == 422:
        assert calls == [], f"Security failure: Agent loop was executed for bad input type: {description}"
    else:
        assert len(calls) == 1


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
