import anthropic

client = anthropic.Anthropic()
conversation_history = [
    {"role": "user", "content": "My name is Kennedy Isiaho"},
    {"role": "assistant", "content": "Hello Kennedy Isiaho! I will remember your name. How can I help you today?"},
    {"role":"user", "content": "What is my name?"}
]
msg = client.messages.create(
    model="claude-haiku-5-5", max_tokens=300,
    messages=conversation_history,
)
for block in msg.content:
    if block.type == "thinking":
        print(f"🧠 Claude is thinking: {block.thinking}")
    elif block.type == "text":
        print(f"💬 Claude's response: {block.text}")
