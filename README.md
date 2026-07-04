# ⚖️ Truth Engine — Tier 2

> **Hybrid Search · Cross-Encoder Re-Ranking · Confidence Scoring · Conflict Detection**

Tier 2 builds directly on top of Tier 1 with four new features that significantly
improve retrieval quality and answer reliability.

---

## What's New in Tier 2

### 1. Hybrid Search (BM25 + FAISS → RRF)

Every source now has **two indexes** built during ingestion:
- `FAISSStore` — semantic vector search (same as Tier 1)
- `BM25Store` — keyword search using `rank-bm25`

At query time, both are searched in parallel. Results are fused using
**Reciprocal Rank Fusion (RRF)**: `score = 1 / (60 + rank)`.

Chunks appearing in both result sets get a higher combined score.
This handles both semantic queries ("how does the timeout work?") and
exact technical queries ("ota.update_timeout_seconds").

### 2. Cross-Encoder Re-Ranking

After hybrid retrieval, the top candidates are re-scored by
`cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB, CPU-friendly).

Unlike embedding similarity (query vs chunk separately), a cross-encoder
reads **query + chunk together**, giving much more accurate relevance scores.
The re-ranker runs on all candidates and returns only the top-k most relevant.

### 3. Confidence Scoring

Every answer gets a **High / Medium / Low** confidence label:
- 🟢 **High** — cross-encoder score ≥ 65% after sigmoid normalisation
- 🟡 **Medium** — 35–65%
- 🔴 **Low** — below 35%

Shown per-source in the UI and per-chunk in the retrieved chunks panel.

### 4. Conflict Detection

When both Source A and Source C are loaded, the system automatically checks for contradictions:

- **Semantic**: Compares cosine similarity of top chunks
- **Numeric**: Detects differing numbers, durations, version strings
- **Negation**: Flags deprecation or negation language in Source C

When a conflict is found, the UI shows **both versions side by side** so you can decide which to trust. Source A (Golden Truth) is recommended but Source C is fully visible.

---

## Architecture

```
POST /ingest/{A|B|C}
  file_parser.py → text_splitter.py → embeddings.py
                                     ↓              ↓
                              FAISSStore        BM25Store   ← NEW Tier 2
                              (vector)         (keyword)

POST /chat
  embed query (once, shared)
        ↓
  HybridRetriever per source   ← NEW Tier 2
    FAISS search + BM25 search → RRF fusion
        ↓
  CrossEncoderReranker         ← NEW Tier 2
    re-scores top candidates
        ↓
  SourceConfidence computed    ← NEW Tier 2
        ↓
  ConflictDetector (A vs C)    ← NEW Tier 2
        ↓
  LLMRouter → structured answer
```

---

## Setup (same as Tier 1, one extra package)

```cmd
cd truth-engine-t2

copy .env.example .env
# Add GEMINI_API_KEY and GROQ_API_KEY

py -m venv venv
venv\Scripts\activate
py -m pip install -r requirements.txt
```

The only new package vs Tier 1: `rank-bm25==0.2.2` (tiny, no model download needed).

The cross-encoder model (`ms-marco-MiniLM-L-6-v2`, ~80MB) is downloaded automatically
by `sentence-transformers` on first startup — same library already installed.

---

## Run

**Terminal 1 — Backend:**
```cmd
venv\Scripts\activate
py main.py
```

Wait for:
```
Embedding engine ready.
Cross-encoder reranker ready.
=== Startup complete. Listening... ===
```

**Terminal 2 — Frontend:**
```cmd
venv\Scripts\activate
py -m streamlit run streamlit_app.py
```

---

## API Changes from Tier 1

### POST /chat — Response now includes:

```json
{
  "source_a": {
    "confidence": {
      "label": "High",
      "score": 0.82,
      "top_chunk_score": 0.91
    },
    "chunks": [
      {
        "confidence": 0.91,
        "confidence_label": "High",
        ...
      }
    ]
  },
  "conflict": {
    "conflict_detected": true,
    "severity": "high",
    "conflict_type": "numeric",
    "source_a_excerpt": "OTA timeout is 300 seconds...",
    "source_c_excerpt": "OTA timeout is 60 seconds...",
    "explanation": "Numerical values differ: Source A references 300 seconds while Source C references 60 seconds.",
    "numeric_conflicts": ["300 seconds", "60 seconds"]
  }
}
```

---

## File Changes from Tier 1

| File | Status | Change |
|------|--------|--------|
| `app/db/bm25_store.py` | 🆕 New | BM25 keyword index |
| `app/core/hybrid_retriever.py` | 🆕 New | RRF fusion of FAISS + BM25 |
| `app/core/reranker.py` | 🆕 New | Cross-encoder + confidence scoring |
| `app/core/conflict_detector.py` | 🆕 New | A vs C contradiction detection |
| `app/core/truth_engine.py` | 🔄 Updated | Full Tier 2 pipeline |
| `app/api/session_store.py` | 🔄 Updated | Holds BM25Store per source |
| `app/api/routes/ingest.py` | 🔄 Updated | Builds BM25 alongside FAISS |
| `app/api/routes/chat.py` | 🔄 Updated | Confidence + conflict in response |
| `app/ui/chat_panel.py` | 🔄 Updated | Confidence labels + conflict panel |
| `main.py` | 🔄 Updated | Warms reranker at startup |
| `requirements.txt` | 🔄 Updated | Added `rank-bm25` |

All other files (config, embeddings, LLM providers, upload panel, etc.) are **unchanged**.
