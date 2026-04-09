# 📊 Assignment 3 — Financial RAG Agent (LangGraph vs LangChain)

A Retrieval-Augmented Generation (RAG) system that answers financial questions from Apple and Tesla's 2024 annual reports (10-K), built with **LangGraph** and compared against a **LangChain** baseline.

> **Best Result: 14/14** using `intfloat/multilingual-e5-base` with `chunk_size=1000`

---

## 🏗️ Architecture Overview

This project implements a **LangGraph** agent with four core nodes:

```
retrieve_node → grade_documents_node → generate_node
                        ↓ (if irrelevant)
                  rewrite_node → retrieve_node (retry)
```

### LangGraph vs LangChain

| Feature | LangChain (Legacy) | LangGraph |
|---|---|---|
| Control Flow | Implicit, black-box ReAct loop | Explicit nodes and edges |
| State Management | Stateless, pass-through only | Global `AgentState` (tracks `search_count`, etc.) |
| Error Recovery | LLM self-corrects (unreliable) | Forced `rewrite_node` on failed retrieval |
| Routing | All tools given to LLM at once | Router classifies `apple` / `tesla` / `both` first |
| Loop Prevention | Depends on `max_iterations` | Programmable retry cap via `search_count` |

---

## 🧪 Experiment Results

### Embedding Model Comparison (`chunk_size=1000`)

| Model | Score | Notes |
|---|---|---|
| `intfloat/multilingual-e5-base` | **14/14** ✅ | Designed for asymmetric retrieval; 768-dim vectors |
| `paraphrase-multilingual-MiniLM-L12-v2` | **13/14** | General-purpose similarity; 384-dim vectors; failed Tesla CapEx |

### Chunk Size Comparison (`intfloat/multilingual-e5-base`)

| Chunk Size | Score | Key Observation |
|---|---|---|
| 500 | 10/14 | Table rows split across chunks; Q4 vs annual data confusion |
| **1000** | **14/14** ✅ | Aligns with financial statement block size — sweet spot |
| 1500 | 13/14 | Multiple tables merged into one chunk; embedding semantics diluted |

---

## 🛠️ Prerequisites

- Python **3.11** (strict requirement)
- Google Cloud API Key or other LLM provider key

---

## ⚙️ Environment Setup

### 1. Create Virtual Environment

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Rename `.env_example` to `.env` and add your API key:
```
API_KEY=your_key_here
```

---

## 🚀 Execution Order

```bash
# Step 1: Build vector database from PDFs
python build_rag.py

# Step 2: Run evaluation benchmark
python evaluator.py
```

> In `evaluator.py`, set `test_mode = "GRAPH"` for LangGraph or `"LEGACY"` for LangChain baseline.

---

## 📂 File Structure

```
Assignment-3/
├── data/                    # Raw PDF financial reports (Apple & Tesla 10-K)
├── langgraph_agent.py       # ⭐ Main workspace — all node logic lives here
├── evaluator.py             # Benchmark script with LLM-as-a-Judge scoring
├── build_rag.py             # PDF ingestion → Chroma vector DB
├── config.py                # API key loading, LLM & embedding model init
├── .env_example             # Template for environment variables
└── requirements.txt
```

### Key File Details

- **`langgraph_agent.py`** — Contains `retrieve_node`, `grade_documents_node`, `generate_node`, `rewrite_node`, and `run_legacy_agent`
- **`evaluator.py`** — Runs test suite (Apple Revenue, Tesla R&D, Comparisons, Trap questions) and scores Pass/Fail
- **`config.py`** — Handles model initialization; swap embedding models or LLM providers here

---

## 📝 Task Summary

| Task | Node | Description |
|---|---|---|
| Task 1 | `run_legacy_agent` | LangChain ReAct baseline with prompt template |
| Task 2 | `retrieve_node` | Route query to `apple`, `tesla`, or `both` |
| Task 3 | `grade_documents_node` | Filter irrelevant retrieved chunks |
| Task 4 | `generate_node` | Generate English answer with citations |
| Task 5 | `rewrite_node` | Refine failed queries (e.g. translate Chinese financial terms) |

---

## 💡 Key Implementation Notes

- The `rewrite_node` translates Chinese financial terms to English (e.g. `研發費用` → `research and development expenses`) to improve vector search recall
- For `both` routing, separate queries are generated for Apple and Tesla using their respective financial terminology (`Total net sales` vs `Total revenues`)
- `search_count` in `AgentState` prevents infinite retry loops — max 2 rewrites before forcing generation
- Out-of-scope questions (e.g. future product pricing) are handled by scope-checking before rewrite, returning `"This information is not available in the financial reports provided."`
