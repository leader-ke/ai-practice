import anthropic

client = anthropic.Anthropic()

RATES = {
    "claude-haiku-5-5": {"under_100k": {"input": 0.10, "output": 0.50}, "over_100k": {"input": 0.50, "output": 2.50}},
    "claude-sonnet-5-5": {"input": 3.00, "output": 15.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
}

model_name = "claude-haiku-5-5"

try:
    print(f"Sending API request using active model: {model_name}...")
    msg = client.messages.create(
        model=model_name,
        max_tokens=300,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
    )

    input_tokens = msg.usage.input_tokens
    output_tokens = msg.usage.output_tokens

    if model_name == "claude-haiku-5-5":
        tier = "under_100k" if input_tokens <= 100_000 else "over_100k"
        model_rates = RATES[model_name][tier]
    else:
        model_rates = RATES.get(model_name, {"input": 3.00, "output": 15.00})

    input_cost = (input_tokens / 1_000_000) * model_rates["input"]
    output_cost = (output_tokens / 1_000_000) * model_rates["output"]
    total_cost = input_cost + output_cost

    print("\n--- RESPONSE ---")
    print("".join(block.text for block in msg.content if block.type == "text"))

    print("\n--- METRICS ---")
    print(f"Tokens Used -> Input: {input_tokens} | Output: {output_tokens}")
    print(f"💳 Total Call Cost: ${total_cost:.6f}")

except Exception as e:
    print(f"\n❌ Error captured: {e}")
