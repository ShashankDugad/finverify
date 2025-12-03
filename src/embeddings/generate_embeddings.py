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
    device: str = None,
    resume: bool = True,
    save_every_n_batches: int = 10
):
    """
    Generate embeddings for all chunks with resume support.
    
    Args:
        chunks_path: Path to chunks JSON/JSONL file (if None, uses config)
        output_path: Path to save embeddings.npy (if None, uses config)
        model_name: Embedding model name (if None, uses config)
        batch_size: Batch size for encoding (if None, uses config)
        device: Device to use ('cuda', 'mps', or 'cpu', if None, auto-detects)
        resume: If True, resume from existing embeddings (default: True)
        save_every_n_batches: Save embeddings every N batches (default: 10)
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
    
    # Auto-detect best device if not specified
    if device is None:
        device = config['models']['embedder'].get('device', None)
        if device is None:
            # Auto-detect: prefer CUDA, then MPS, then CPU
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
    
    # Handle device fallback
    if device == "cuda" and not torch.cuda.is_available():
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("CUDA not available, using MPS (Apple Silicon GPU) instead")
            device = "mps"
        else:
            print("CUDA not available, using CPU")
            device = "cpu"
    elif device == "mps":
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print("MPS not available, using CPU")
            device = "cpu"
    
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        print("GPU: Apple Silicon (MPS)")
    
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
    total_chunks = len(texts)
    
    # Load existing embeddings if resuming
    existing_embeddings = None
    processed_count = 0
    
    if resume and output_path.exists():
        try:
            print(f"\nLoading existing embeddings from {output_path}...")
            existing_embeddings = np.load(output_path)
            processed_count = existing_embeddings.shape[0]
            print(f"✓ Found {processed_count:,} existing embeddings")
            
            if processed_count >= total_chunks:
                print("✓ All chunks already have embeddings!")
                return existing_embeddings
            else:
                print(f"⏭ Resuming from chunk {processed_count:,} / {total_chunks:,}")
        except Exception as e:
            print(f"⚠ Could not load existing embeddings: {e}")
            print("Starting from scratch...")
            existing_embeddings = None
            processed_count = 0
    
    # Calculate how many batches to process
    start_idx = processed_count
    remaining_texts = texts[start_idx:]
    remaining_count = len(remaining_texts)
    
    if remaining_count == 0:
        print("✓ All embeddings already generated!")
        return existing_embeddings
    
    # Generate embeddings in batches
    print(f"\nGenerating embeddings for {remaining_count:,} remaining chunks (batch_size={batch_size})...")
    new_embeddings = []
    
    num_batches = (remaining_count + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(0, remaining_count, batch_size), desc="Batches", total=num_batches):
        batch = remaining_texts[batch_idx:batch_idx+batch_size]
        
        embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        new_embeddings.append(embeddings)
        
        # Save incrementally every N batches
        if len(new_embeddings) >= save_every_n_batches:
            # Concatenate new embeddings so far
            temp_embeddings = np.vstack(new_embeddings)
            
            # Combine with existing if any
            if existing_embeddings is not None:
                combined = np.vstack([existing_embeddings, temp_embeddings])
            else:
                combined = temp_embeddings
            
            # Save checkpoint
            np.save(output_path, combined)
            
            # Update existing_embeddings and reset new_embeddings
            existing_embeddings = combined
            new_embeddings = []
    
    # Concatenate any remaining new embeddings
    if new_embeddings:
        print("\nConcatenating final embeddings...")
        final_new = np.vstack(new_embeddings)
        
        # Combine with existing
        if existing_embeddings is not None:
            all_embeddings = np.vstack([existing_embeddings, final_new])
        else:
            all_embeddings = final_new
    else:
        all_embeddings = existing_embeddings
    
    # Final save
    print(f"\nSaving final embeddings to {output_path}...")
    np.save(output_path, all_embeddings)
    
    print(f"\n✓ Embeddings shape: {all_embeddings.shape}")
    print(f"✓ Total processed: {all_embeddings.shape[0]:,} / {total_chunks:,} chunks")
    print(f"✓ Saved to: {output_path}")
    
    return all_embeddings


if __name__ == "__main__":
    generate_embeddings()

