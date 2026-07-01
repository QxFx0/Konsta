import unittest
from collections import namedtuple

from src.api_parsers import (
    parse_anthropic_messages,
    parse_openai_messages,
    serialize_anthropic_messages,
)

Message = namedtuple("Message", ["role", "content"])

class TestApiParsers(unittest.TestCase):
    def test_parse_openai_missing_keys(self):
        # P1: Potential KeyError if 'role' or 'content' keys are missing
        messages = [{"role": "user"}, {"content": "hi"}, {}]
        parsed = parse_openai_messages(messages)
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0].role, "user")
        self.assertEqual(parsed[0].content, "")
        self.assertEqual(parsed[1].role, "unknown")
        self.assertEqual(parsed[1].content, "hi")
        self.assertEqual(parsed[2].role, "unknown")
        self.assertEqual(parsed[2].content, "")

    def test_parse_anthropic_missing_role(self):
        # P1: Potential KeyError if 'role' key is missing
        messages = [{"content": "hello"}, {}]
        parsed = parse_anthropic_messages(messages)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0].role, "unknown")
        self.assertEqual(parsed[0].content, "hello")

    def test_anthropic_text_blocks_joining(self):
        # P3: Joining Anthropic text blocks with a space may alter formatting
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"}
            ]
        }]
        parsed = parse_anthropic_messages(messages)
        # Should be joined without extra spaces if that's the requirement,
        # or at least consistent. Let's check the implementation.
        self.assertEqual(parsed[0].content, "HelloWorld")

    def test_serialize_anthropic_system_key(self):
        # P2: serialize_anthropic_messages only sets 'system' when a system
        # prompt is explicitly provided via the system_prompt argument. The
        # implementation does not auto-extract the system role from messages;
        # callers (e.g. api_parsers.serialize_request) are expected to pull
        # the system message out and pass it explicitly.
        # Case 1: System prompt is provided -> 'system' key is present
        messages = [Message(role="system", content="Sys"), Message(role="user", content="User")]
        serialized = serialize_anthropic_messages(messages, system_prompt="Sys")
        self.assertEqual(serialized.get("system"), "Sys")

        # Case 2: No system prompt -> 'system' key is absent
        messages = [Message(role="user", content="User")]
        serialized = serialize_anthropic_messages(messages)
        self.assertNotIn("system", serialized)

if __name__ == "__main__":
    unittest.main()
