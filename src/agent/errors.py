class AgentError(Exception):
    """Base exception for all agent-related problems."""

    pass


class AgentConfigError(AgentError):
    """Raised when the agent is misconfigured (e.g., bad API key)."""

    pass


class AgentUnavailableError(AgentError):
    """Raised when the upstream service is overloaded, rate-limited, or unreachable."""

    pass


class AgentTimeoutError(AgentError):
    """Raised when the upstream service fails to respond within the timeout window."""

    pass


class AgentExecutionError(AgentError):
    """Raised when the internal agent execution loop aborts or hits thresholds."""

    pass
