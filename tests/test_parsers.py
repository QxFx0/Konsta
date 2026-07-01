from src.api_parsers import (
    Message,
    parse_anthropic_messages,
    parse_openai_messages,
    serialize_anthropic_messages,
    serialize_openai_messages,
)

# --- OpenAI Tests ---

def test_parse_openai_messages_success():
    messages = [
        {"role": "system", "content": "You are a helper"},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    parsed = parse_openai_messages(messages)
    assert len(parsed) == 3
    assert parsed[0] == Message(role="system", content="You are a helper")
    assert parsed[1] == Message(role="user", content="Hello!")
    assert parsed[2] == Message(role="assistant", content="Hi there!")

def test_parse_openai_messages_missing_keys():
    messages = [
        {"role": "user"},           # missing content
        {"content": "hi"},          # missing role
        {},                            # missing both
        "not a dict",                 # invalid type
    ]
    parsed = parse_openai_messages(messages)
    assert len(parsed) == 3
    assert parsed[0] == Message(role="user", content="")
    assert parsed[1] == Message(role="unknown", content="hi")
    assert parsed[2] == Message(role="unknown", content="")

def test_serialize_openai_messages():
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi")
    ]
    serialized = serialize_openai_messages(messages)
    assert serialized == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}
    ]

# --- Anthropic Tests ---

def test_parse_anthropic_messages_simple():
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi!"}
    ]
    parsed = parse_anthropic_messages(messages)
    assert len(parsed) == 2
    assert parsed[0] == Message(role="user", content="Hello!")
    assert parsed[1] == Message(role="assistant", content="Hi!")

def test_parse_anthropic_messages_complex_content():
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "World"},
            {"type": "image", "image": {}}
        ]
    }]
    parsed = parse_anthropic_messages(messages)
    assert len(parsed) == 1
    # Should join text blocks and ignore non-text blocks
    assert parsed[0].content == "Hello World"

def test_parse_anthropic_messages_missing_keys():
    messages = [
        {"content": "hello"}, # missing role
        {},                      # missing both
        None,                    # invalid type
    ]
    parsed = parse_anthropic_messages(messages)
    assert len(parsed) == 2
    assert parsed[0] == Message(role="unknown", content="hello")
    assert parsed[1] == Message(role="unknown", content="")

def test_serialize_anthropic_messages_with_system():
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi")
    ]
    system_prompt = "You are a helpful assistant"
    serialized = serialize_anthropic_messages(messages, system_prompt=system_prompt)

    assert serialized["system"] == system_prompt
    assert len(serialized["messages"]) == 2
    assert serialized["messages"][0] == {"role": "user", "content": "Hello"}

def test_serialize_anthropic_messages_no_system():
    messages = [
        Message(role="user", content="Hello")
    ]
    serialized = serialize_anthropic_messages(messages)

    assert "system" not in serialized
    assert len(serialized["messages"]) == 1

def test_serialize_anthropic_filters_system_messages():
    # Anthropic's 'messages' list should not contain 'system' role
    messages = [
        Message(role="system", content="Internal system msg"),
        Message(role="user", content="Hello")
    ]
    serialized = serialize_anthropic_messages(messages)

    # The system message should be filtered out of the messages list
    assert len(serialized["messages"]) == 1
    assert serialized["messages"][0]["role"] == "user"
