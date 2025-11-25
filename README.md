# FinVERIFY - Multi-Aspect Retrieval-Augmented Financial Fact-Checking

**Author:** Shashank Dugad (sd5957@nyu.edu)  
**Course:** NYU DS-GA 1011 Natural Language Processing  
**Project:** Final Project - First 25% Deliverable

---

## Project Overview

FinVERIFY is a retrieval-augmented generation (RAG) system for financial fact-checking. This deliverable implements the core retrieval infrastructure using both lexical (BM25) and semantic (FAISS + BGE) approaches.

### Goal
Enable accurate financial question answering by retrieving relevant context from SEC filings and financial QA datasets.

---

## What's Included (25% Deliverable)

### ✅ Completed Components

1. **Data Acquisition**
   - SEC EDGAR: 73 10-K filings (1.6GB) from major companies
   - FinanceBench: 150 financial QA pairs
   - TATQA: 2,757 table+text QA examples
   - Glaive RAG-v1: 10,000 training examples

2. **Document Processing**
   - Chunking: 899,226 chunks (2048 chars with 200 char overlap)
   - Stored in: `data/processed/chunks.json` (1.9GB)

3. **BM25 Index (Lexical Retrieval)**
   - Index: 899,226 documents
   - Storage: `data/indexes/bm25/` (3.3GB)
   - Performance: 1.5-3.3s latency, keyword-based matching

4. **BGE Embeddings + FAISS Index (Semantic Retrieval)**
   - Embeddings: 899,226 × 768 dimensions (2.6GB)
   - Model: BAAI/bge-base-en-v1.5
   - FAISS Index: `data/indexes/faiss/faiss_index.bin` (2.6GB)
   - Performance: 1.0-1.6s latency, semantic similarity

5. **Demo Application**
   - Gradio UI for testing retrieval
   - Supports both BM25 and FAISS methods

---

## Directory Structure
```
finverify/
├── data/
│   ├── raw/                        # Original datasets
│   │   ├── sec_edgar/              # 73 SEC filings (1.6GB)
│   │   ├── financebench/           # 150 examples
│   │   ├── tatqa/                  # 2,757 examples
│   │   └── glaive/                 # 10,000 examples
│   ├── processed/                  # Processed data
│   │   ├── chunks.json             # 899K chunks (1.9GB)
│   │   └── embeddings.npy          # BGE embeddings (2.6GB)
│   └── indexes/                    # Search indexes
│       ├── bm25/                   # BM25 index (3.3GB)
│       └── faiss/                  # FAISS index (2.6GB)
├── src/
│   ├── ingestion/                  # Data download scripts
│   ├── chunking/                   # Document chunking
│   ├── embeddings/                 # BGE embedding generation
│   └── bm25/                       # BM25 index building
├── demo/
│   └── gradio_app.py              # Gradio demo UI
└── README.md                       # This file
```

---

## Setup Instructions

### Prerequisites
- NYU HPC Greene cluster access
- JupyterHub with GPU (g2-standard-12 partition)

### Environment Setup
```bash
# Navigate to project directory
cd /scratch/$USER/finverify

# Install dependencies
pip install torch transformers sentence-transformers faiss-gpu rank-bm25 gradio datasets --break-system-packages
```

### Quick Start

#### 1. Test BM25 Retrieval
```bash
python3 src/bm25/test_bm25.py
```

#### 2. Test FAISS Retrieval
```bash
python3 src/embeddings/test_faiss.py
```

#### 3. Launch Demo (Optional)
```bash
python3 demo/gradio_app.py
```

---

## Data Statistics

| Dataset | Count | Size |
|---------|-------|------|
| SEC Filings | 73 | 1.6GB |
| FinanceBench | 150 | 52KB |
| TATQA | 2,757 | 14MB |
| Glaive RAG-v1 | 10,000 | 616KB |
| **Total Chunks** | **899,226** | **1.9GB** |
| **BM25 Index** | 899,226 docs | 3.3GB |
| **FAISS Index** | 899,226 vectors | 2.6GB |

---

## Performance Metrics

### BM25 (Lexical)
- **Latency:** 1.5-3.3 seconds
- **Method:** TF-IDF based keyword matching
- **Best for:** Exact term matching (e.g., company names, specific metrics)

### FAISS (Semantic)
- **Latency:** 1.0-1.6 seconds
- **Method:** Cosine similarity on BGE embeddings
- **Best for:** Conceptual queries (e.g., "cash position" matches "liquidity")

---

## Next Steps (Remaining 75%)

The team will continue with:

1. **Baseline Models** (Week 3-4)
   - BM25 + T5 baseline
   - DPR + T5 baseline

2. **MAINRAG Implementation** (Week 3-4)
   - 4-aspect retrieval (semantic, lexical, entity, temporal)
   - Reciprocal Rank Fusion (RRF)

3. **Reranking** (Week 5-6)
   - Cross-encoder reranking

4. **Generation** (Week 5-6)
   - Flan-T5-XL fine-tuning on Glaive RAG-v1

5. **Evaluation** (Week 7-9)
   - FinanceBench + TATQA test sets
   - Metrics: EM, F1, Recall@K, Latency

6. **Final Deliverables** (Week 10-11)
   - Poster presentation (Dec 5)
   - Final report (8 pages ACL format)

---

## Technical Details

### Chunking Strategy
- **Size:** 2048 characters
- **Overlap:** 200 characters
- **Method:** Character-based sliding window
- **Why:** Balance between context preservation and retrieval granularity

### Embedding Model
- **Model:** BAAI/bge-base-en-v1.5
- **Dimensions:** 768
- **Device:** NVIDIA L4 GPU
- **Time:** ~3.5 hours for 899K chunks

### Index Types
- **BM25:** Okapi BM25 implementation via rank-bm25
- **FAISS:** IndexFlatIP (exact inner product search)
- **Similarity:** Cosine similarity (L2 normalized vectors)

---

## Files Not in Git

Due to size, the following are excluded via `.gitignore`:

- `data/raw/*` (datasets)
- `data/processed/*` (chunks, embeddings)
- `data/indexes/*` (BM25, FAISS indexes)

**Reproduction:** Run download + processing scripts in `src/ingestion/` and `src/chunking/`

---

## Contact

**Shashank Dugad**  
Email: sd5957@nyu.edu  
GitHub: https://github.com/ShashankDugad/finverify

---

## Acknowledgments

- NYU HPC Greene cluster for compute resources
- Course: DS-GA 1011 Natural Language Processing
- Datasets: SEC EDGAR, PatronusAI/FinanceBench, NExTplusplus/TATQA, Glaive RAG-v1
