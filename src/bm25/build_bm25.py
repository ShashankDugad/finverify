"""
Build BM25 index from chunks
"""

import os
import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm

def build_bm25_index():
    """Build BM25 index"""
    
    print("=" * 60)
    print("BM25 Index Builder")
    print("=" * 60)
    
    base_dir = Path("/scratch") / os.environ['USER'] / "finverify"
    chunks_file = base_dir / "data" / "processed" / "chunks.json"
    output_dir = base_dir / "data" / "indexes" / "bm25"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load chunks
    print("\nLoading chunks...")
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    print(f"✓ Loaded {len(chunks)} chunks")
    
    # Tokenize for BM25
    print("\nTokenizing chunks...")
    tokenized_corpus = []
    chunk_ids = []
    
    for chunk in tqdm(chunks, desc="Tokenizing"):
        tokens = chunk['text'].lower().split()
        tokenized_corpus.append(tokens)
        chunk_ids.append(chunk['chunk_id'])
    
    # Build BM25
    print("\nBuilding BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save index
    print("\nSaving BM25 index...")
    
    with open(output_dir / "bm25_index.pkl", 'wb') as f:
        pickle.dump(bm25, f)
    
    with open(output_dir / "chunk_ids.pkl", 'wb') as f:
        pickle.dump(chunk_ids, f)
    
    with open(output_dir / "chunks.pkl", 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"\n✓ BM25 index saved to {output_dir}")
    print(f"✓ Index size: {len(chunks)} documents")
    
    return bm25, chunk_ids, chunks

if __name__ == "__main__":
    build_bm25_index()
