"""
Build FAISS index from embeddings
"""

import os
import numpy as np
import faiss
from pathlib import Path

def build_faiss_index():
    """Build FAISS index"""
    
    print("=" * 60)
    print("FAISS Index Builder")
    print("=" * 60)
    
    base_dir = Path("/scratch") / os.environ['USER'] / "finverify"
    embeddings_file = base_dir / "data" / "processed" / "embeddings.npy"
    output_dir = base_dir / "data" / "indexes" / "faiss"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    print("\nLoading embeddings...")
    embeddings = np.load(embeddings_file)
    
    print(f"✓ Loaded embeddings: {embeddings.shape}")
    
    # Normalize for cosine similarity
    print("\nNormalizing embeddings...")
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index
    print("\nBuilding FAISS index...")
    dimension = embeddings.shape[1]
    
    # Use IndexFlatIP for exact cosine similarity search
    index = faiss.IndexFlatIP(dimension)
    
    # Add embeddings
    index.add(embeddings)
    
    print(f"✓ Index built: {index.ntotal} vectors")
    
    # Save index
    print("\nSaving FAISS index...")
    faiss.write_index(index, str(output_dir / "faiss_index.bin"))
    
    print(f"\n✓ FAISS index saved to {output_dir / 'faiss_index.bin'}")
    print(f"✓ Index size: {index.ntotal} vectors")
    
    return index

if __name__ == "__main__":
    build_faiss_index()
