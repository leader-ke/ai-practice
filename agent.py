import anthropic

from config import LLM_MODEL

client = anthropic.Anthropic()


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


tickets = {"1234567890": "open", "1234567891": "closed", "1234567892": "on hold"}


def lookup_ticket(ticket_id):
    if ticket_id in tickets:
        status = tickets.get(ticket_id)
        return f"Ticket status is: {status}", False
    else:
        return f"Error: Ticket id {ticket_id} not found.", True


TOOL_MAPPING = {"lookup_ticket": lookup_ticket}

tools = [
    {
        "name": "lookup_ticket",
        "description": "Look up a ticket by its ID and return its status",
        "input_schema": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}},
            "required": ["ticket_id"],
        },
    }
]


def run_agent(question: str) -> str:
    messages = [
        {"role": "user", "content": question},
    ]

    MAX_ROUNDS = 5

    for round_count in range(1, MAX_ROUNDS + 1):
        print(f"--- Round {round_count} ---")

        try:
            msg = client.messages.create(model=LLM_MODEL, max_tokens=1000, tools=tools, messages=messages)
        except anthropic.AuthenticationError as e:
            raise AgentConfigError("Agent API configuration error.") from e

        except anthropic.RateLimitError as e:
            raise AgentUnavailableError("Upstream service is currently rate limited.") from e

        except anthropic.APITimeoutError as e:
            raise AgentTimeoutError("Upstream service timed out.") from e

        except anthropic.APIConnectionError as e:
            raise AgentUnavailableError("Failed to connect to upstream service.") from e

        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise AgentUnavailableError("Upstream service reports internal failure.") from e
            else:
                raise AgentConfigError(f"Bad API request format (Status {e.status_code}).") from e

        except Exception as e:
            raise AgentExecutionError("Unexpected system anomaly during model invocation.") from e

        if msg.stop_reason == "tool_use":
            tool_use_blocks = [block for block in msg.content if block.type == "tool_use"]

            if tool_use_blocks:
                tool_result_content = []

                for tool_use in tool_use_blocks:
                    tool_name = tool_use.name
                    requested_id = tool_use.input.get("ticket_id") if isinstance(tool_use.input, dict) else None

                    print(f"Claude requested tool {tool_name} for ticket id {requested_id}")

                    try:
                        if not requested_id:
                            raise ValueError("Missing or invalid 'ticket_id' parameter structure")

                        if tool_name in TOOL_MAPPING:
                            output_message, has_error = TOOL_MAPPING[tool_name](requested_id)
                        else:
                            output_message = f"Error: Tool '{tool_name}' not implemented."
                            has_error = True
                    except Exception as e:
                        output_message = f"System Error executing tool '{tool_name}' : {str(e)}"
                        has_error = True

                    print(f"Tool result -> '{output_message}' (is_error: {has_error})")

                    tool_result_content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": output_message,
                            "is_error": has_error,
                        }
                    )

                messages.append({"role": "assistant", "content": msg.content})
                messages.append({"role": "user", "content": tool_result_content})

        elif msg.stop_reason == "end_turn" or msg.stop_reason == "stop_sequence":
            final_text_block = next((block for block in msg.content if block.type == "text"), None)
            if final_text_block:
                return final_text_block.text
            else:
                return str(msg.content)

        elif msg.stop_reason == "max_tokens":
            raise RuntimeError("Agent execution halted: Claude hit token limits midway through production.")

        else:
            raise RuntimeError(f"Agent execution stopped due to unexpected stop condition: {msg.stop_reason}")

    raise RuntimeError(
        f"Agent loop exhausted maximum safety authorization limit of {MAX_ROUNDS} rounds without completing."
    )


if __name__ == "__main__":
    print("Starting Agentic Loop...\n")
    ans1 = run_agent("What is 2 + 2?")
    print(f"Result: {ans1}\n")

    print("==============================================")
    ans2 = run_agent("What are the statuses of tickets 1234567890 and 1234567891?")
    print(f"Result: {ans2}\n")
