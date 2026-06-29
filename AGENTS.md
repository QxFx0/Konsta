# AGENTS.md - Instructions for AI Agents

This project is designed to be maintained and extended by AI software engineers.

## Core Logic
- **Proxy Core**: Intercepts requests. Logic is in `src/proxy_core.py`.
- **Compression Pipeline**: 
    1. Merge consecutive messages (`_merge_consecutive_messages`).
    2. Semantic Deduplication via `sentence-transformers` (`_deduplicate_semantically`).
    3. LLM Distillation via Cerebras/Custom LLM (`LLMProcessor.process_context`).
- **CA Management**: Handles certificates via `src/ca_manager.py`.

## Contribution Guidelines for AI
1. **Async Safety**: When modifying `proxy_core.py`, remember it's a synchronous mitmproxy addon. Always use the asyncio bridge for `LLMProcessor`.
2. **Performance**: Aim for vectorized operations in `compression_engine.py`.
3. **Security**: Never hardcode API keys. Use `Config` class and environment variables.

## Testing
- Run tests in the `tests/` directory.
- Ensure `mitmdump` is in PATH for integration tests.
