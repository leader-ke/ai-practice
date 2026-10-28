# The agent loop

Code: [src/agent](../../src/agent/). The loop is tested with a fake client in [test_agent_loop.py](../../tests/test_agent_loop.py), and the API in [test_api.py](../../tests/test_api.py).

Claude never runs my code. I describe a tool (a name, a description and a JSON schema for its input). When Claude wants it, the response contains a `tool_use` block. My code runs the function and sends the result back, and the conversation goes on until Claude answers in plain text.

## One round trip
1. Send the question and the tool descriptions.
2. The response has `stop_reason == "tool_use"` and one or more `tool_use` blocks, each with an `id`, a `name` and an `input`.
3. Run each tool.
4. Send the next request with the original question, then Claude's response exactly as received (as the assistant turn), then one user message holding a `tool_result` for every `tool_use`, each carrying the matching `tool_use_id`.
5. Repeat until `stop_reason` is something other than `tool_use`.

## What matters in practice
- The API is stateless. My code keeps the message list and resends it on every call.
- Read blocks by `type`, never by position. A reply can hold text, tool requests and sometimes thinking.
- Every tool request in a turn needs a result, all in one user message.
- A failed tool should return a result with `is_error: true`, and a tool that raises must not crash the loop.
- Cap the rounds (5 here), so a model that keeps asking for tools cannot run forever or run up the bill. Giving up raises `AgentExecutionError`.
- Log tokens and seconds for each round. It is the first step toward watching cost and speed.
- Wording changes between runs, so tests check structure and behavior, never exact text.

| `stop_reason` | What the code does |
|---|---|
| `end_turn`, `stop_sequence` | return the text |
| `tool_use` | run the tools and loop |
| `max_tokens` | raise `AgentExecutionError`, because the answer was cut off |
| anything else | raise `AgentExecutionError` |

## From loop to service
`agent` turns the SDK's exceptions into my own error types, and the API maps those to HTTP status codes. See [decisions](../decisions.md).

## Mistakes
I read `content[1].text` and it broke when the first block was a tool request. I wrote the assistant turn by hand instead of sending Claude's real response. My first version handled only one tool request and one round.

## Still unclear
When is the SDK's Tool Runner enough, and when do I want the loop in my own hands? TODO(ken): add anything else.
