"""
Test BM25 retrieval
"""

import os
import pickle
import time
from pathlib import Path

def test_bm25():
    """Test BM25 retrieval"""
    
    print("=" * 60)
    print("BM25 Retrieval Test")
    print("=" * 60)
    
    base_dir = Path("/scratch") / os.environ['USER'] / "finverify"
    index_dir = base_dir / "data" / "indexes" / "bm25"
    
    # Load index
    print("\nLoading BM25 index...")
    with open(index_dir / "bm25_index.pkl", 'rb') as f:
        bm25 = pickle.load(f)
    
    with open(index_dir / "chunk_ids.pkl", 'rb') as f:
        chunk_ids = pickle.load(f)
    
    with open(index_dir / "chunks.pkl", 'rb') as f:
        chunks = pickle.load(f)
    
    print(f"✓ Loaded index: {len(chunks)} documents")
    
    # Test queries
    test_queries = [
        "What is Apple's total revenue for fiscal year 2024?",
        "How much cash does Microsoft have?",
        "Tesla gross margin analysis",
        "Amazon operating expenses breakdown",
        "Google advertising revenue growth"
    ]
    
    print("\n" + "=" * 60)
    print("Testing Retrieval")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Search
        start_time = time.time()
        scores = bm25.get_scores(query_tokens)
        
        # Get top 10
        top_indices = scores.argsort()[-10:][::-1]
        latency = (time.time() - start_time) * 1000
        
        print(f"Latency: {latency:.1f}ms")
        print("Top 3 results:")
        
        for i, idx in enumerate(top_indices[:3], 1):
            chunk = chunks[idx]
            text_preview = chunk['text'][:150].replace('\n', ' ')
            print(f"  {i}. [{chunk['source']}] Score: {scores[idx]:.2f}")
            print(f"     {text_preview}...")
    
    print("\n" + "=" * 60)
    print("✓ BM25 retrieval working")
    print("=" * 60)

if __name__ == "__main__":
    test_bm25()
