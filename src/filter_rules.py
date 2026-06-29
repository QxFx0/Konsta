from typing import List

def truncate_string(text: str, max_len: int, suffix: str = "... [truncated]") -> str:
    """
    Truncates a string to a maximum length, ensuring the final length
    including the suffix does not exceed max_len.
    """
    if not text:
        return ""
    
    if len(text) <= max_len:
        return text
    
    suffix_len = len(suffix)
    
    # If max_len is smaller than the suffix itself, we just truncate the text
    # and return as much of the suffix as possible, or just the truncated text.
    # But typically max_len will be reasonably larger than the suffix.
    if max_len <= suffix_len:
        return text[:max_len]
    
    return text[:max_len - suffix_len] + suffix

def deduplicate_strings(strings: List[str]) -> List[str]:
    """
    Removes duplicate strings from a list while preserving the original order.
    """
    seen = set()
    result = []
    for s in strings:
        if s not in seen:
            result.append(s)
            seen.add(s)
    return result

if __name__ == "__main__":
    # Simple internal verification
    test_text = "This is a very long string that should be truncated"
    max_l = 20
    truncated = truncate_string(test_text, max_l)
    print(f"Truncated: '{truncated}' (Length: {len(truncated)})")
    assert len(truncated) <= max_l
    
    test_list = ["apple", "banana", "apple", "orange", "banana"]
    deduped = deduplicate_strings(test_list)
    print(f"Deduped: {deduped}")
    assert deduped == ["apple", "banana", "orange"]
    print("All internal tests passed!")
