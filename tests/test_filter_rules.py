import unittest
from src.filter_rules import truncate_string, deduplicate_strings

class TestFilterRules(unittest.TestCase):
    def test_truncate_string_no_truncation(self):
        self.assertEqual(truncate_string("hello", 10), "hello")

    def test_truncate_string_with_truncation(self):
        # max_len = 10, suffix = "... [truncated]" (length 15)
        # If max_len is smaller than suffix length, it should just return the suffix sliced or the string sliced
        # But typically max_len > suffix_len
        text = "This is a very long string that needs truncation"
        max_len = 20
        result = truncate_string(text, max_len)
        self.assertTrue(len(result) <= max_len)
        self.assertTrue(result.endswith("... [truncated]"))

    def test_truncate_string_very_short_max_len(self):
        # Test case where max_len is even shorter than the suffix
        text = "Long string"
        max_len = 5
        result = truncate_string(text, max_len)
        self.assertTrue(len(result) <= max_len)

    def test_deduplicate_strings(self):
        input_list = ["apple", "banana", "apple", "orange", "banana"]
        expected = ["apple", "banana", "orange"]
        self.assertEqual(deduplicate_strings(input_list), expected)

    def test_deduplicate_empty(self):
        self.assertEqual(deduplicate_strings([]), [])

if __name__ == "__main__":
    unittest.main()