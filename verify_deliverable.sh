#!/bin/bash

echo "========================================"
echo "FinVERIFY 25% Deliverable Verification"
echo "========================================"

echo ""
echo "1. DATA FILES"
echo "----------------------------------------"
echo "SEC EDGAR:"
find data/raw/sec_edgar/ -name "*.txt" 2>/dev/null | wc -l
du -sh data/raw/sec_edgar/ 2>/dev/null

echo ""
echo "FinanceBench:"
ls -lh data/raw/financebench/*.json 2>/dev/null | tail -1

echo ""
echo "TATQA:"
ls -lh data/raw/tatqa/*.json 2>/dev/null

echo ""
echo "Glaive:"
ls -lh data/raw/glaive/*.json 2>/dev/null

echo ""
echo "2. PROCESSED DATA"
echo "----------------------------------------"
echo "Chunks:"
ls -lh data/processed/chunks.json 2>/dev/null

echo ""
echo "Embeddings:"
ls -lh data/processed/embeddings.npy 2>/dev/null

echo ""
echo "3. INDEXES"
echo "----------------------------------------"
echo "BM25:"
du -sh data/indexes/bm25/ 2>/dev/null

echo ""
echo "FAISS:"
ls -lh data/indexes/faiss/faiss_index.bin 2>/dev/null

echo ""
echo "4. SOURCE CODE"
echo "----------------------------------------"
find src/ -name "*.py" 2>/dev/null | wc -l
echo "Python files in src/"

echo ""
echo "5. DEMO"
echo "----------------------------------------"
ls -lh demo/gradio_app.py 2>/dev/null

echo ""
echo "6. DOCUMENTATION"
echo "----------------------------------------"
ls -lh README.md 2>/dev/null

echo ""
echo "========================================"
echo "✓ Verification Complete"
echo "========================================"
