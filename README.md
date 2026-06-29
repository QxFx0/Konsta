# 🌌 Konsta: Intelligent Context Compression Proxy

**Konsta** is a high-performance context compression proxy designed to reduce LLM token usage and improve response quality through multi-stage semantic distillation.

By sitting between your application and the LLM provider, Konsta automatically optimizes prompt history, removing redundancy and distilling complex contexts into a concise, high-density format before they ever hit the expensive API.

---

## 📈 Efficiency Metrics & Economics

Konsta transforms the economics of long-context LLM applications by replacing expensive "raw" tokens with high-density "distilled" tokens.

| Metric | Raw Request | With Konsta | Improvement |
| :--- | :--- | :--- | :--- |
| **Context Volume** | 100% (Full History) | 30% - 60% | **40-70% Reduction** |
| **Noise Level** | High (Redundant) | Low (Concentrated) | **Significant $\downarrow$** |
| **Cost (Avg)** | $1.00 (Premium Model) | $0.20 - $0.40 | **60-80% Savings** |

### 💰 The "Distillation Arbitrage"
Why use Konsta? Because the cost of compressing context with a fast model is negligible compared to the cost of processing that same context in a flagship model.

**Example Scenario:**
- **Target Model:** Claude 3.5 Sonnet / GPT-4o (Expensive)
- **Compressor:** Gemma-4-31B via Cerebras (Ultra-fast & Cheap)
- **The Win:** You pay a fraction of a cent to compress 10k tokens down to 2k, then pay the premium model only for those 2k tokens. **The ROI is immediate.**

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

## 🛠 How It Works

Konsta implements a three-stage compression pipeline:

1. **Semantic Layer (Local)**: Uses `all-MiniLM-L6-v2` to identify and remove semantically identical messages.
2. **Distillation Layer (Remote)**: A specialized "compressor" LLM rewrites the remaining context into a dense, summarized format.
3. **Transparent Proxy**: Acts as a MITM proxy, modifying requests on-the-fly and forwarding them to providers like Mistral, OpenAI, or Anthropic.

---

## 🚀 Getting Started (Step-by-Step)

### 1. Clone the Repository
```bash
git clone https://github.com/QxFx0/Konsta.git
cd Konsta
```

### 2. Environment Setup
It is recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
Konsta uses `sentence-transformers` and `torch` for local semantic analysis.

**For CPU only (Standard):**
```bash
pip install -r requirements.txt
```

**For GPU acceleration (Recommended for high load):**
Install the CUDA-enabled version of PyTorch first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Configuration & API Keys
Konsta is configured via environment variables. You can set them in your terminal or create a `.env` file in the root directory.

**Required Keys:**
- `LLM_API_KEY`: Your **Cerebras** API key (used for the Gemma-4-31B compressor).
- `TARGET_API_KEY`: (Optional) Your key for the target provider (OpenAI, Mistral, Anthropic), though most users keep these in their application side.

**Example setup:**
```bash
export LLM_API_KEY="your_cerebras_api_key_here"
# Add other provider keys if needed by your app
```

---

## 📦 Using Konsta

### Option A: The Demo CLI (Fastest way to test)
Perfect for seeing how a prompt is compressed without setting up a proxy.
```bash
python3 -m demo.demo_cli
```

### Option B: Full Transparent Proxy
Run Konsta as a middleware. It will intercept traffic and compress it on the fly.
```bash
python3 -m src.main
```
- **Proxy Address:** `http://127.0.0.1:8080`
- **Integration:** Point your LLM client/application to this address instead of the provider's direct URL.

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
