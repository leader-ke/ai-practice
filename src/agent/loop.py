import logging
import time
from functools import lru_cache

import anthropic

from agent.errors import (
    AgentConfigError,
    AgentExecutionError,
    AgentTimeoutError,
    AgentUnavailableError,
)
from agent.tools import run_tool, tools
from config import ANTHROPIC_API_KEY, LLM_MODEL, MAX_TOKENS, REQUEST_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

MAX_ROUNDS = 5


@lru_cache(maxsize=1)
def get_client():
    """Create the Anthropic client on first use, not at import."""
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=REQUEST_TIMEOUT_SECONDS)


def call_model(messages):
    """One model call, with the SDK's errors translated into the agent's own types."""
    try:
        return get_client().messages.create(model=LLM_MODEL, max_tokens=MAX_TOKENS, tools=tools, messages=messages)
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


def run_agent(question: str) -> str:
    messages = [
        {"role": "user", "content": question},
    ]
    input_tokens = 0
    output_tokens = 0

    for round_count in range(1, MAX_ROUNDS + 1):
        started = time.perf_counter()
        msg = call_model(messages)
        input_tokens += msg.usage.input_tokens
        output_tokens += msg.usage.output_tokens
        logger.info(
            "round %d: stop_reason=%s input_tokens=%d output_tokens=%d seconds=%.2f",
            round_count,
            msg.stop_reason,
            msg.usage.input_tokens,
            msg.usage.output_tokens,
            time.perf_counter() - started,
        )

        if msg.stop_reason == "tool_use":
            tool_use_blocks = [block for block in msg.content if block.type == "tool_use"]

            if tool_use_blocks:
                tool_result_content = []

                for tool_use in tool_use_blocks:
                    logger.info("tool requested: %s %s", tool_use.name, tool_use.input)
                    output_message, has_error = run_tool(tool_use.name, tool_use.input)
                    logger.info("tool result: is_error=%s", has_error)

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
            logger.info(
                "agent finished: rounds=%d input_tokens=%d output_tokens=%d", round_count, input_tokens, output_tokens
            )
            final_text_block = next((block for block in msg.content if block.type == "text"), None)
            if final_text_block:
                return final_text_block.text
            else:
                return str(msg.content)

        elif msg.stop_reason == "max_tokens":
            raise AgentExecutionError("Agent execution halted: Claude hit token limits midway through production.")

        else:
            raise AgentExecutionError(f"Agent execution stopped due to unexpected stop condition: {msg.stop_reason}")

    raise AgentExecutionError(
        f"Agent loop exhausted maximum safety authorization limit of {MAX_ROUNDS} rounds without completing."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Starting Agentic Loop...\n")
    ans1 = run_agent("What is 2 + 2?")
    print(f"Result: {ans1}\n")

    print("==============================================")
    ans2 = run_agent("What are the statuses of tickets 1234567890 and 1234567891?")
    print(f"Result: {ans2}\n")
