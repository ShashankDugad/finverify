"""
Build BM25 index from chunks
Supports both local workspace and HPC scratch paths
"""

import os
import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def get_base_dir():
    """Determine base directory (local workspace or HPC scratch)"""
    workspace_path = Path(__file__).parent.parent.parent
    # Check if we're in a scratch directory
    if '/scratch/' in str(workspace_path) or 'USER' in str(workspace_path):
        import os
        return Path("/scratch") / os.environ.get('USER', 'user') / "finverify"
    return workspace_path


def build_bm25_index():
    """Build BM25 index"""
    
    print("=" * 60)
    print("BM25 Index Builder")
    print("=" * 60)
    
    base_dir = get_base_dir()
    
    # Try multiple possible chunk file locations
    possible_chunk_files = [
        base_dir / "data" / "processed" / "chunks.jsonl",
        base_dir / "data" / "processed" / "chunks.json",
        base_dir / "data" / "files" / "processed" / "chunks.jsonl",
    ]
    
    chunks_file = None
    for path in possible_chunk_files:
        if path.exists():
            chunks_file = path
            break
    
    if chunks_file is None:
        raise FileNotFoundError(
            f"Chunks file not found. Tried: {possible_chunk_files}"
        )
    
    output_dir = base_dir / "data" / "indexes" / "bm25"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load chunks
    print(f"\nLoading chunks from {chunks_file}...")
    
    chunks = []
    if chunks_file.suffix == ".jsonl":
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
    else:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    
    print(f"✓ Loaded {len(chunks)} chunks")
    
    # Tokenize for BM25
    print("\nTokenizing chunks...")
    tokenized_corpus = []
    chunk_ids = []
    
    for chunk in tqdm(chunks, desc="Tokenizing"):
        # Handle both dict and string chunks
        if isinstance(chunk, dict):
            text = chunk.get('text', '')
            chunk_id = chunk.get('chunk_id', '')
        else:
            text = str(chunk)
            chunk_id = ''
        
        # Simple tokenization: lowercase and split on whitespace
        tokens = text.lower().split()
        tokenized_corpus.append(tokens)
        chunk_ids.append(chunk_id)
    
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
    print(f"✓ Files saved:")
    print(f"  - bm25_index.pkl")
    print(f"  - chunk_ids.pkl")
    print(f"  - chunks.pkl")
    
    return bm25, chunk_ids, chunks


if __name__ == "__main__":
    build_bm25_index()

