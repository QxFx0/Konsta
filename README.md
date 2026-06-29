# 🌌 Konsta: Intelligent Context Compression Proxy

**Konsta** is a high-performance context compression proxy designed to reduce LLM token usage and improve response quality through multi-stage semantic distillation.

By sitting between your application and the LLM provider, Konsta automatically optimizes prompt history, removing redundancy and distilling complex contexts into a concise, high-density format before they ever hit the expensive API.

---

## 🚀 The Value Proposition

### 💰 Cost Reduction
Most LLM providers charge per token. For applications with long conversation histories, costs scale quadratically. Konsta reduces the "token tax" by:
- **Semantic Deduplication**: removing redundant information using local embeddings.
- **LLM Distillation**: using a fast, specialized model (e.g., Gemma-4-31B) to summarize context without losing critical facts.

### 🎯 Higher Quality (Anti-Noise)
LLMs often suffer from the "Lost in the Middle" phenomenon, where critical information in long prompts is ignored. Konsta cleans the noise, providing the target model with a **concentrated extract** of the context, leading to:
- Higher accuracy.
- Reduced hallucinations.
- Faster time-to-first-token.

---

## 🛠 How It Works

Konsta implements a three-stage compression pipeline:

1. **Semantic Layer (Local)**: Uses `all-MiniLM-L6-v2` to identify and remove semantically identical messages.
2. **Distillation Layer (Remote)**: A specialized "compressor" LLM rewrites the remaining context into a dense, summarized format.
3. **Transparent Proxy**: Acts as a MITM proxy, modifying requests on-the-fly and forwarding them to providers like Mistral, OpenAI, or Anthropic.

---

## 📦 Quick Start (Demo Mode)

To showcase the compression magic without configuring a full proxy, use the Demo CLI:

### 1. Installation
```bash
pip install -r requirements.txt
pip install rich httpx
```

### 2. Run the Demo
```bash
export LLM_API_KEY="your_cerebras_key"
python3 -m demo.demo_cli
```

### 3. Run as a Proxy
```bash
export LLM_API_KEY="your_cerebras_key"
python3 -m src.main
```
Then point your application to `http://127.0.0.1:8080`.

---

## 🏗 Project Structure

- `src/`: The core engine.
  - `proxy_core.py`: The MITM proxy logic (powered by `mitmproxy`).
  - `compression_engine.py`: Local semantic deduplication.
  - `llm_processor.py`: Remote distillation pipeline.
  - `api_parsers.py`: Request/Response serialization for various providers.
- `demo/`: Quickstart tools.
  - `demo_cli.py`: Beautiful TUI to visualize the compression process.
- `docs/`: Architectural decisions and audit reports.

---

## 📜 License
MIT License.
