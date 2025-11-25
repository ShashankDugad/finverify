"""
Generate BGE embeddings for all chunks
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def generate_embeddings():
    """Generate embeddings for all chunks"""
    
    print("=" * 60)
    print("BGE Embedding Generator")
    print("=" * 60)
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    base_dir = Path("/scratch") / os.environ['USER'] / "finverify"
    chunks_file = base_dir / "data" / "processed" / "chunks.json"
    output_dir = base_dir / "data" / "processed"
    
    # Load model
    print("\nLoading BGE model...")
    model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)
    
    # Load chunks
    print("Loading chunks...")
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    print(f"✓ Loaded {len(chunks)} chunks")
    
    # Generate embeddings in batches
    print("\nGenerating embeddings...")
    batch_size = 128
    all_embeddings = []
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Batches"):
        batch = chunks[i:i+batch_size]
        texts = [c['text'] for c in batch]
        
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        all_embeddings.append(embeddings)
    
    # Concatenate
    print("\nConcatenating embeddings...")
    all_embeddings = np.vstack(all_embeddings)
    
    # Save
    print(f"\nSaving embeddings...")
    np.save(output_dir / "embeddings.npy", all_embeddings)
    
    print(f"\n✓ Embeddings shape: {all_embeddings.shape}")
    print(f"✓ Saved to: {output_dir / 'embeddings.npy'}")
    
    return all_embeddings

if __name__ == "__main__":
    generate_embeddings()
