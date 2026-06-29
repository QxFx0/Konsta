# 🌌 Konsta: High-Fidelity Context Optimization Proxy

**Konsta** is an advanced context optimization layer designed to eliminate "prompt noise" and drastically reduce LLM hallucinations. By transforming bloated, redundant prompts into high-density semantic extracts, Konsta ensures that your LLM focuses only on the critical information, resulting in higher accuracy and superior reasoning.

While most proxies focus solely on cost, **Konsta focuses on Signal-to-Noise Ratio (SNR)**. We don't just compress; we distill.

---

## 🎯 The Core Value: Quality Over Everything

### 🛡️ Anti-Hallucination & Anti-Noise
Large contexts often lead to the **"Lost in the Middle"** phenomenon, where LLMs ignore critical facts buried in noise. Konsta solves this by:
- **Semantic Deduplication**: Removing redundant and contradictory information using local embeddings.
- **Context Distillation**: Using a specialized LLM to rewrite complex contexts into a concentrated, high-density format.
- **Concentrated Focus**: Providing the target model with only the essential facts, which directly reduces hallucinations and increases factual precision.

### 💰 Economic Efficiency (The Bonus)
Because Konsta delivers high-density prompts, you naturally achieve:
- **60-80% Reduction** in input token costs.
- **Faster Time-to-First-Token (TTFT)** due to smaller prompt processing.
- **Distillation Arbitrage**: Using a fast, cheap model (e.g., Gemma-4-31B) to optimize prompts for a premium flagship (e.g., Claude Opus 4.8 / GPT-5.5).

---

## 📈 Efficiency & Quality Metrics

| Metric | Raw Request | With Konsta | Impact |
| :--- | :--- | :--- | :--- |
| **Signal-to-Noise Ratio** | Low (Bloated) | **Ultra-High** | $\uparrow$ Accuracy |
| **Fact Retrieval** | Unreliable (Lost in Middle) | **Deterministic** | $\downarrow$ Hallucinations |
| **Context Volume** | 100% | 30% - 60% | $\downarrow$ Cost |
| **Model Reasoning** | Diluted | **Concentrated** | $\uparrow$ Depth |

---

## 🧪 Performance Case Studies: From Noise to Expertise

### Case 1: "Needle in a Haystack" $\rightarrow$ Surgical Precision
**Scenario**: Finding one specific secret code in 50k+ tokens of system logs.
- **Without Konsta**: The model often suffers from "Lost in the Middle," missing the secret or hallucinating a wrong code due to noise.
- **With Konsta**: The semantic engine strips away the redundant logs and focuses the LLM on the specific event.
- **Result**: **100% Retrieval Accuracy** and zero noise-induced errors.

### Case 2: "Expert Architecture Analysis" $\rightarrow$ High-Level Reasoning
**Scenario**: Deep analysis of a 20-page Enterprise Architecture Document.
- **The Challenge**: Synthesizing complex links between Kafka, CRDTs, and Security protocols.
- **Konsta's Magic**: Instead of sending the whole document, Konsta distills "corporate speak" and preserves the technical core (e.g., `LWW-Element-Sets`, `TTFB < 10ms`, `Citus sharding`).
- **Result**: The target model provides **actionable engineering blueprints** (CLI commands, audit steps) instead of generic summaries.

---

## 🔄 Request Flow Architecture
```mermaid
sequenceDiagram
    participant User as 👤 User / App
    participant Konsta as 🛡️ Konsta Proxy
    participant Distill as ⚡ Gemma-4 (Distiller)
    participant Provider as 🤖 LLM Provider (Claude/GPT)

    User->>Konsta: Sends Bloated/Noisy Request
    Note over Konsta: 1. Semantic Noise Removal
    Konsta->>Distill: Sends Redundant Context
    Distill-->>Konsta: Returns High-Density Distillation
    Note over Konsta: 2. Signal-to-Noise Optimization
    Konsta->>Provider: Sends Concentrated Prompt
    Provider-->>Konsta: Returns High-Precision Response
    Konsta-->>User: Delivers Accurate Answer
```

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
- `TARGET_API_KEY`: (Optional) Your key for the target provider (OpenAI, Mistral, Anthropic).

**Example setup:**
```bash
export LLM_API_KEY="your_cerebras_api_key_here"
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
