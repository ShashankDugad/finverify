# Step-by-Step Guide to Run FinanceBench Multi-Aspect Retrieval System

## Prerequisites

### 1. Install Dependencies

```bash
cd "/Users/sbalikondwar/nyu work/NLP/finverify"
pip install -r requirements.txt

# Additional dependencies for PDF processing
pip install unstructured[pdf] faiss-gpu  # Use faiss-cpu if no GPU

# Install spaCy model for NER
python -m spacy download en_core_web_sm
```

### 2. Verify Directory Structure

Ensure you have the following structure:
```
finverify/
├── config.yaml
├── data/
│   ├── raw/
│   │   └── financebench/
│   │       ├── pdfs/          # PDF files go here
│   │       ├── financebench_open_source.jsonl
│   │       └── financebench_document_information.jsonl
│   ├── processed/              # Will contain chunks.jsonl
│   ├── embeddings/             # Will contain embeddings.npy
│   └── indexes/               # Will contain FAISS index
│       └── faiss/
└── src/
```

---

## Step 1: Download FinanceBench Data (if not already done)

If you don't have FinanceBench PDFs yet:

```bash
cd "/Users/sbalikondwar/nyu work/NLP/finverify"

# Option 1: Use the existing download script
python financebench-graph-rag/download_financebench.py

# Option 2: Manual download using datasets library
python -c "
from datasets import load_dataset
import requests
from pathlib import Path
from tqdm import tqdm

ds = load_dataset('PatronusAI/financebench', split='train')
pdfs_dir = Path('data/raw/financebench/pdfs')
pdfs_dir.mkdir(parents=True, exist_ok=True)

for x in tqdm(ds):
    outp = pdfs_dir / f\"{x['doc_name']}.pdf\"
    if not outp.exists():
        try:
            r = requests.get(x['doc_link'])
            with open(outp, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            print(f'Error downloading {x[\"doc_name\"]}: {e}')
"
```

**Expected output:** PDFs should be in `data/raw/financebench/pdfs/`

---

## Step 2: Chunk PDFs

This step processes all PDFs and creates chunks with overlap:

```bash
cd "/Users/sbalikondwar/nyu work/NLP/finverify"
python src/chunking/pdf_chunker.py
```

**What it does:**
- Reads PDFs from `data/raw/financebench/pdfs/`
- Chunks them with size=512, overlap=128 (from config.yaml)
- Saves chunks to `data/processed/chunks.jsonl`

**Expected output:**
```
Processing X PDF files...
Chunking PDFs: 100%|████████| X/X [XX:XX<00:00, X.XXit/s]
✓ Saved XXXX chunks to data/processed/chunks.jsonl
```

---

## Step 3: Generate Embeddings

Generate BGE embeddings for all chunks:

```bash
cd "/Users/sbalikondwar/nyu work/NLP/finverify"
python src/embeddings/generate_embeddings.py
```

**What it does:**
- Loads chunks from `data/processed/chunks.jsonl`
- Generates embeddings using BGE model (from config.yaml)
- Saves embeddings to `data/embeddings/embeddings.npy`

**Expected output:**
```
============================================================
BGE Embedding Generator
============================================================

Device: cuda
GPU: NVIDIA GeForce RTX 3090

Loading embedding model: BAAI/bge-large-en-v1.5...
Loading chunks from data/processed/chunks.jsonl...
✓ Loaded XXXX chunks

Generating embeddings (batch_size=256)...
Batches: 100%|████████| XX/XX [XX:XX<00:00, X.XXit/s]

Concatenating embeddings...

Saving embeddings to data/embeddings/embeddings.npy...

✓ Embeddings shape: (XXXX, 1024)
✓ Saved to: data/embeddings/embeddings.npy
```

**Note:** This step can take 30-60 minutes depending on GPU and number of chunks.

---

## Step 4: Build FAISS Index

Build the FAISS index from embeddings:

```bash
cd "/Users/sbalikondwar/nyu work/NLP/finverify"
python src/embeddings/build_faiss_index.py
```

**What it does:**
- Loads embeddings from `data/embeddings/embeddings.npy`
- Normalizes embeddings for cosine similarity
- Builds FAISS IndexFlatIP index
- Saves index to `data/indexes/faiss/faiss_index.bin`

**Expected output:**
```
============================================================
FAISS Index Builder
============================================================

Loading embeddings from data/embeddings/embeddings.npy...
✓ Loaded embeddings: (XXXX, 1024)

Normalizing embeddings for cosine similarity...

Building FAISS index (IndexFlatIP)...
Adding embeddings to index...
✓ Index built: XXXX vectors

Saving FAISS index to data/indexes/faiss/faiss_index.bin...

✓ FAISS index saved to data/indexes/faiss/faiss_index.bin
✓ Index size: XXXX vectors
✓ Index dimension: 1024
```

---

## Step 5: Test Individual Retrievers

### Test FAISS Retriever (Baseline)

Create a test script `test_retrieval.py`:

```python
from src.retrieval import FaissRetriever

# Initialize retriever
retriever = FaissRetriever()

# Test query
query = "What is Apple's total revenue?"
results = retriever.retrieve(query, top_k=5)

# Print results
print(f"\nQuery: {query}\n")
print("=" * 60)
for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Score: {result['score']:.4f}")
    print(f"Document: {result['metadata'].get('doc_name', 'N/A')}")
    print(f"Content: {result['content'][:200]}...")
```

Run it:
```bash
python test_retrieval.py
```

### Test Entity Retriever

```python
from src.retrieval import EntityRetriever

retriever = EntityRetriever()
query = "What is Microsoft's cash position?"
results = retriever.retrieve(query, top_k=5)
# ... same as above
```

### Test Temporal Retriever

```python
from src.retrieval import TemporalRetriever

retriever = TemporalRetriever()
query = "What was Apple's revenue in 2023?"
results = retriever.retrieve(query, top_k=5)
# ... same as above
```

---

## Step 6: Use Unified Retriever (Multi-Aspect)

The unified retriever combines all methods with RRF:

```python
from src.retrieval import UnifiedRetriever

# Initialize unified retriever (uses all methods by default)
retriever = UnifiedRetriever(
    methods=['faiss', 'entity', 'temporal'],
    use_reranker=True
)

# Query
query = "What is Tesla's gross margin in 2023?"
results = retriever.retrieve(query, top_k=10)

# Print results
print(f"\nQuery: {query}\n")
print("=" * 60)
for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Score: {result['score']:.4f}")
    print(f"Document: {result['metadata'].get('doc_name', 'N/A')}")
    print(f"Content: {result['content'][:200]}...")
```

---

## Step 7: Evaluate on FinanceBench Questions

Create an evaluation script `evaluate_retrieval.py`:

```python
from src.retrieval import UnifiedRetriever
from src.utils import load_financebench_questions
from tqdm import tqdm

# Load questions
questions = load_financebench_questions()

# Initialize retriever
retriever = UnifiedRetriever()

# Evaluate on first 10 questions
results = []
for q in tqdm(questions[:10]):
    query = q['question']
    retrieved = retriever.retrieve(query, top_k=10)
    
    results.append({
        'question': query,
        'answer': q.get('answer', ''),
        'retrieved': retrieved
    })

# Print summary
print(f"\nEvaluated {len(results)} questions")
for r in results:
    print(f"\nQ: {r['question']}")
    print(f"A: {r['answer']}")
    print(f"Retrieved {len(r['retrieved'])} chunks")
```

---

## Troubleshooting

### Issue: "Chunks file not found"
**Solution:** Make sure Step 2 completed successfully. Check `data/processed/chunks.jsonl` exists.

### Issue: "FAISS index not found"
**Solution:** Run Step 4 to build the FAISS index.

### Issue: "spaCy model not found"
**Solution:** Run `python -m spacy download en_core_web_sm`

### Issue: "CUDA out of memory"
**Solution:** 
- Reduce batch_size in config.yaml
- Use CPU: Set `device: "cpu"` in config.yaml for embedder/reranker

### Issue: "FinanceBench PDFs not found"
**Solution:** Run Step 1 to download PDFs, or manually place PDFs in `data/raw/financebench/pdfs/`

---

## Quick Start (All Steps)

If you want to run everything in sequence:

```bash
cd "/Users/sbalikondwar/nyu work/NLP/finverify"

# 1. Download data (if needed)
python financebench-graph-rag/download_financebench.py

# 2. Chunk PDFs
python src/chunking/pdf_chunker.py

# 3. Generate embeddings
python src/embeddings/generate_embeddings.py

# 4. Build FAISS index
python src/embeddings/build_faiss_index.py

# 5. Test retrieval
python -c "
from src.retrieval import UnifiedRetriever
retriever = UnifiedRetriever()
results = retriever.retrieve('What is Apple revenue?', top_k=5)
for r in results:
    print(f\"Score: {r['score']:.4f} - {r['content'][:100]}...\")
"
```

---

## Configuration

All settings are in `config.yaml`. Key parameters:

- **Chunking:** `data.chunk_size`, `data.chunk_overlap`
- **Embeddings:** `models.embedder.name`, `models.embedder.batch_size`
- **Retrieval:** `retrieval.mainrag.*` (top_k for each method)
- **RRF:** `retrieval.rrf.k_value`, `retrieval.rrf.merge_top_k`
- **Reranking:** `retrieval.reranking.enabled`, `retrieval.reranking.final_top_k`

Modify these as needed and re-run the relevant steps.

