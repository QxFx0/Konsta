import unittest
from src.api_parsers import (
    parse_openai_messages, 
    serialize_openai_messages, 
    parse_anthropic_messages, 
    serialize_anthropic_messages
)

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
        # P2: serialize_anthropic_messages always sets 'system' key
        # Case 1: System prompt exists
        messages = [Message(role="system", content="Sys"), Message(role="user", content="User")]
        serialized = serialize_anthropic_messages(messages)
        self.assertEqual(serialized.get("system"), "Sys")
        
        # Case 2: No system prompt
        messages = [Message(role="user", content="User")]
        serialized = serialize_anthropic_messages(messages)
        self.assertNotIn("system", serialized)

from collections import namedtuple
Message = namedtuple("Message", ["role", "content"])

if __name__ == "__main__":
    unittest.main()