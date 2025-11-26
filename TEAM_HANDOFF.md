# FinVERIFY - Team Handoff Instructions

**Prepared by:** Shashank Dugad (sd5957)  
**Date:** November 25, 2025  
**Status:** 25% Complete → Team continues with remaining 75%

---

## What's Already Done (Shashank - 25%)

### ✅ Completed
- Downloaded 73 SEC filings, FinanceBench (150), TATQA (2,757), Glaive (10K)
- Created 899K chunks from all documents
- Built BM25 index (keyword search)
- Generated BGE embeddings (semantic search)
- Built FAISS index (fast similarity search)
- Both retrieval methods tested and working

### 📁 Data Location (NYU HPC)
```
/scratch/sd5957/finverify/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/
│   │   ├── chunks.json         # 899K chunks (1.9GB)
│   │   └── embeddings.npy      # BGE vectors (2.6GB)
│   └── indexes/
│       ├── bm25/               # BM25 index (3.3GB)
│       └── faiss/              # FAISS index (2.6GB)
└── src/                        # All code
```

**Access:** Copy data from `/scratch/sd5957/finverify/` to your scratch space

---

## Team Work Division (75% Remaining)

### **Person 1: Utkarsh (Weeks 3-4, ~25%)**
**Tasks:**
1. Implement BM25+T5 baseline
   - Use existing BM25 retrieval
   - Add Flan-T5-base for answer generation
   - Test on 50 FinanceBench questions
2. Implement DPR+T5 baseline
   - Use existing FAISS retrieval
   - Add Flan-T5-base
   - Compare with BM25+T5
3. Document baseline results (accuracy, latency)

**Deliverables:**
- `src/baselines/bm25_t5.py`
- `src/baselines/dpr_t5.py`
- Baseline results spreadsheet

---

### **Person 2: Shivam (Weeks 5-6, ~25%)**
**Tasks:**
1. Implement MAINRAG 4-aspect retrieval
   - Combine: semantic (FAISS), lexical (BM25), entity, temporal
   - Implement Reciprocal Rank Fusion (RRF)
2. Add cross-encoder reranking
   - Use `ms-marco-MiniLM-L-12-v2`
   - Rerank top-20 results → top-5
3. Test hybrid retrieval on 100 questions

**Deliverables:**
- `src/rag/mainrag.py`
- `src/reranker/cross_encoder.py`
- Retrieval improvement analysis

---

### **Person 3: Surbhi (Weeks 7-9, ~25%)**
**Tasks:**
1. Fine-tune Flan-T5-XL on Glaive RAG-v1
   - Use 10K training examples from `/scratch/sd5957/finverify/data/raw/glaive/`
   - Train on HPC with GPU
2. Run full evaluation
   - Test on FinanceBench (150 questions)
   - Test on TATQA (2,757 questions)
   - Metrics: EM, F1, Recall@10, Latency
3. Create final demo + poster

**Deliverables:**
- `src/generator/finetune_t5.py`
- Evaluation results
- Working Gradio demo
- Poster draft

---

## Quick Start Guide

### Step 1: Copy Data (One-time)
```bash
# From your HPC account
cd /scratch/$USER
cp -r /scratch/sd5957/finverify/data ./finverify/
cp -r /scratch/sd5957/finverify/src ./finverify/
cd finverify
```

### Step 2: Setup Environment
```bash
# Request GPU node (non-Tandon students use srun)
srun --gres=gpu:1 --time=8:00:00 --pty bash

# Install packages
pip install torch transformers sentence-transformers faiss-gpu rank-bm25 gradio datasets
```

### Step 3: Test Existing Retrieval
```bash
# Test BM25
python3 src/bm25/test_bm25.py

# Test FAISS
python3 src/embeddings/test_faiss.py
```

If both work → You're ready to start!

---

## Claude Prompts for Each Person

### 🤖 Prompt for Utkarsh (Baseline Models)
```
I'm Utkarsh working on the FinVERIFY NLP project. My teammate Shashank completed the first 25%:
- Downloaded SEC filings, FinanceBench, TATQA datasets
- Created 899K chunks
- Built BM25 index (keyword retrieval)
- Built FAISS index (semantic retrieval with BGE embeddings)

Data location: /scratch/sd5957/finverify/

MY TASKS (25%):
1. Implement BM25+T5 baseline:
   - Use existing BM25 retrieval (src/bm25/)
   - Add Flan-T5-base for generation
   - Test on 50 FinanceBench questions
2. Implement DPR+T5 baseline:
   - Use existing FAISS retrieval (src/embeddings/)
   - Add Flan-T5-base
   - Compare performance

I have access to NYU HPC Greene cluster with srun (not JupyterHub).
Help me implement these baselines step-by-step.
```

---

### 🤖 Prompt for Shivam (MAINRAG + Reranking)
```
I'm Shivam working on FinVERIFY NLP project. Previous work done:
- Shashank (25%): Data + BM25 + FAISS indexes built
- Utkarsh (25%): BM25+T5 and DPR+T5 baselines working

Data location: /scratch/sd5957/finverify/

MY TASKS (25%):
1. Implement MAINRAG 4-aspect retrieval:
   - Semantic (FAISS - already built)
   - Lexical (BM25 - already built)
   - Entity extraction + matching
   - Temporal filtering
   - Combine with Reciprocal Rank Fusion (RRF)
2. Add cross-encoder reranking:
   - Model: ms-marco-MiniLM-L-12-v2
   - Rerank top-20 → top-5
3. Test hybrid retrieval vs baselines

I have NYU HPC access with srun.
Help me implement MAINRAG step-by-step.
```

---

### 🤖 Prompt for Surbhi (Fine-tuning + Evaluation + Demo)
```
I'm Surbhi working on FinVERIFY NLP project. Previous work:
- Shashank (25%): Data + retrieval indexes
- Utkarsh (25%): Baseline models
- Shivam (25%): MAINRAG + reranking

Data location: /scratch/sd5957/finverify/

MY TASKS (25%):
1. Fine-tune Flan-T5-XL on Glaive RAG-v1:
   - Training data: /scratch/sd5957/finverify/data/raw/glaive/ (10K examples)
   - Use HPC GPU
2. Full evaluation:
   - FinanceBench (150 questions)
   - TATQA (2,757 questions)
   - Metrics: Exact Match, F1, Recall@10, Latency
3. Create demo + poster:
   - Working Gradio interface
   - Poster for Dec 5 presentation

I have NYU HPC access with srun.
Help me complete the project step-by-step.
```

---

## Important Notes

### File Sizes
- Chunks: 1.9GB
- Embeddings: 2.6GB
- BM25 index: 3.3GB
- FAISS index: 2.6GB
- **Total:** ~10GB (make sure you have space)

### GPU Requirements
- Baseline models: 1 GPU, 4 hours
- MAINRAG: 1 GPU, 6 hours
- T5-XL fine-tuning: 1 GPU, 12 hours

### Code Structure
```
src/
├── ingestion/       # Done - data download
├── chunking/        # Done - document processing
├── bm25/            # Done - BM25 index
├── embeddings/      # Done - BGE + FAISS
├── baselines/       # TODO - Utkarsh
├── rag/             # TODO - Shivam
├── reranker/        # TODO - Shivam
├── generator/       # TODO - Surbhi
└── evaluation/      # TODO - Surbhi
```

---

## Contact

**Questions about existing code:**  
Shashank Dugad - sd5957@nyu.edu

**Team coordination:**  
Use project Slack/Discord channel

---

## Timeline

| Week | Person | Milestone |
|------|--------|-----------|
| 3-4 | Utkarsh | Baselines complete |
| 5-6 | Shivam | MAINRAG complete |
| 7-9 | Surbhi | Evaluation complete |
| 10 | All | Poster ready |
| 11 | All | Final report (Dec 10) |

---

## GitHub

Repository: https://github.com/ShashankDugad/finverify  
Latest commit: `8a16924`

All code is pushed. Data files are NOT in git (too large) - copy from HPC.

---

**Good luck team! 🚀**

---

## HOW TO COPY DATA (For Teammates)

### Step 1: Create Your Directory
```bash
cd /scratch/$USER
mkdir -p finverify
cd finverify
```

### Step 2: Copy Data
```bash
# Copy all data (takes 5-10 minutes, ~10GB)
cp -r /scratch/sd5957/finverify/data ./

# Copy source code
cp -r /scratch/sd5957/finverify/src ./
```

### Step 3: Verify
```bash
# Check data copied successfully
ls -lh data/processed/
ls -lh data/indexes/bm25/
ls -lh data/indexes/faiss/

# Should see:
# chunks.json (1.9GB)
# embeddings.npy (2.6GB)
# bm25 index (3.3GB)
# faiss index (2.6GB)
```

### Quick Copy Command (All-in-One)
```bash
cd /scratch/$USER && \
mkdir -p finverify && \
cd finverify && \
cp -r /scratch/sd5957/finverify/data . && \
cp -r /scratch/sd5957/finverify/src . && \
echo "✓ Copy complete!"
```

**Estimated time:** 5-10 minutes  
**Space needed:** ~10GB free in your /scratch/

---
