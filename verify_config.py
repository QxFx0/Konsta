import os

from src.config import Config


def test_config():
    print("Testing default config...")
    cfg = Config()
    print(f"Similarity Threshold: {cfg.similarity_threshold}")
    print(f"Exclusion Patterns: {cfg.exclusion_patterns}")

    assert cfg.similarity_threshold == 0.85
    assert ".git" in cfg.exclusion_patterns

    print("Testing environment variable override...")
    os.environ["PROXY_SIMILARITY_THRESHOLD"] = "0.95"
    cfg_env = Config()
    print(f"Overridden Similarity Threshold: {cfg_env.similarity_threshold}")

    assert cfg_env.similarity_threshold == 0.95
    print("All config tests passed!")

if __name__ == "__main__":
    try:
        test_config()
    except Exception as e:
        print(f"Test failed: {e}")
        exit(1)
