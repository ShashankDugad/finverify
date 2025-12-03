"""
Generate BGE embeddings for all chunks using config.yaml paths
"""

import json
import numpy as np
import torch
import yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def get_base_dir():
    """Determine base directory (local workspace or HPC scratch)"""
    workspace_path = Path(__file__).parent.parent.parent
    # Check if we're in a scratch directory
    if '/scratch/' in str(workspace_path) or 'USER' in str(workspace_path):
        import os
        return Path("/scratch") / os.environ.get('USER', 'user') / "finverify"
    return workspace_path


def load_config():
    """Load config.yaml"""
    base_dir = get_base_dir()
    config_path = base_dir / "config.yaml"
    if not config_path.exists():
        # Try relative to workspace
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_embeddings(
    chunks_path: Path = None,
    output_path: Path = None,
    model_name: str = None,
    batch_size: int = None,
    device: str = None
):
    """
    Generate embeddings for all chunks.
    
    Args:
        chunks_path: Path to chunks JSON/JSONL file (if None, uses config)
        output_path: Path to save embeddings.npy (if None, uses config)
        model_name: Embedding model name (if None, uses config)
        batch_size: Batch size for encoding (if None, uses config)
        device: Device to use ('cuda' or 'cpu', if None, auto-detects)
    """
    print("=" * 60)
    print("BGE Embedding Generator")
    print("=" * 60)
    
    # Load config
    config = load_config()
    base_dir = get_base_dir()
    
    # Get paths from config or use provided
    if chunks_path is None:
        chunks_path = base_dir / config['paths']['data_processed'] / "chunks.jsonl"
        # Fallback to chunks.json if JSONL doesn't exist
        if not chunks_path.exists():
            chunks_path = base_dir / config['paths']['data_processed'] / "chunks.json"
    
    if output_path is None:
        output_path = base_dir / config['paths']['embeddings'] / "embeddings.npy"
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get model config
    if model_name is None:
        model_name = config['models']['embedder']['name']
    if batch_size is None:
        batch_size = config['models']['embedder'].get('batch_size', 128)
    if device is None:
        device = config['models']['embedder'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check GPU
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, using CPU")
        device = "cpu"
    
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print(f"\nLoading embedding model: {model_name}...")
    model = SentenceTransformer(model_name, device=device)
    
    # Load chunks
    print(f"Loading chunks from {chunks_path}...")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    
    chunks = []
    if chunks_path.suffix == '.jsonl':
        # Load JSONL format
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
    else:
        # Load JSON format
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    
    print(f"✓ Loaded {len(chunks)} chunks")
    
    # Extract texts
    texts = [chunk['text'] if isinstance(chunk, dict) else chunk for chunk in chunks]
    
    # Generate embeddings in batches
    print(f"\nGenerating embeddings (batch_size={batch_size})...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
        batch = texts[i:i+batch_size]
        
        embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        all_embeddings.append(embeddings)
    
    # Concatenate
    print("\nConcatenating embeddings...")
    all_embeddings = np.vstack(all_embeddings)
    
    # Save
    print(f"\nSaving embeddings to {output_path}...")
    np.save(output_path, all_embeddings)
    
    print(f"\n✓ Embeddings shape: {all_embeddings.shape}")
    print(f"✓ Saved to: {output_path}")
    
    return all_embeddings


if __name__ == "__main__":
    generate_embeddings()

