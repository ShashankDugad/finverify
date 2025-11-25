"""
Test FAISS semantic retrieval
"""

import os
import json
import pickle
import numpy as np
import faiss
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer

def test_faiss():
    """Test FAISS retrieval"""
    
    print("=" * 60)
    print("FAISS Semantic Retrieval Test")
    print("=" * 60)
    
    base_dir = Path("/scratch") / os.environ['USER'] / "finverify"
    faiss_dir = base_dir / "data" / "indexes" / "faiss"
    bm25_dir = base_dir / "data" / "indexes" / "bm25"
    
    # Load model
    print("\nLoading BGE model...")
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    
    # Load FAISS index
    print("Loading FAISS index...")
    index = faiss.read_index(str(faiss_dir / "faiss_index.bin"))
    
    # Load chunks
    print("Loading chunks...")
    with open(bm25_dir / "chunks.pkl", 'rb') as f:
        chunks = pickle.load(f)
    
    print(f"✓ Loaded: {index.ntotal} vectors, {len(chunks)} chunks")
    
    # Test queries
    test_queries = [
        "What is Apple's total revenue for fiscal year 2024?",
        "How much cash does Microsoft have?",
        "Tesla gross margin analysis",
        "Amazon operating expenses breakdown",
        "Google advertising revenue growth"
    ]
    
    print("\n" + "=" * 60)
    print("Testing Semantic Retrieval")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Encode query
        start_time = time.time()
        query_embedding = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        k = 10
        scores, indices = index.search(query_embedding, k)
        latency = (time.time() - start_time) * 1000
        
        print(f"Latency: {latency:.1f}ms")
        print("Top 3 results:")
        
        for i in range(min(3, len(indices[0]))):
            idx = indices[0][i]
            score = scores[0][i]
            chunk = chunks[idx]
            text_preview = chunk['text'][:150].replace('\n', ' ')
            print(f"  {i+1}. [{chunk['source']}] Score: {score:.3f}")
            print(f"     {text_preview}...")
    
    print("\n" + "=" * 60)
    print("✓ FAISS semantic retrieval working")
    print("=" * 60)

if __name__ == "__main__":
    test_faiss()
