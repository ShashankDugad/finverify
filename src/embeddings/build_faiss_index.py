"""
Build FAISS index from embeddings using config.yaml paths
"""

import numpy as np
import faiss
import yaml
from pathlib import Path


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


def build_faiss_index(
    embeddings_path: Path = None,
    output_path: Path = None,
    index_type: str = "IndexFlatIP"
):
    """
    Build FAISS index from embeddings.
    
    Args:
        embeddings_path: Path to embeddings.npy file (if None, uses config)
        output_path: Path to save FAISS index (if None, uses config)
        index_type: Type of FAISS index ('IndexFlatIP' for exact cosine similarity)
    
    Returns:
        FAISS index object
    """
    print("=" * 60)
    print("FAISS Index Builder")
    print("=" * 60)
    
    # Load config
    config = load_config()
    base_dir = get_base_dir()
    
    # Get paths from config or use provided
    if embeddings_path is None:
        embeddings_path = base_dir / config['paths']['embeddings'] / "embeddings.npy"
    
    if output_path is None:
        output_path = base_dir / config['paths']['indexes'] / "faiss" / "faiss_index.bin"
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    print(f"\nLoading embeddings from {embeddings_path}...")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    embeddings = np.load(embeddings_path)
    
    print(f"✓ Loaded embeddings: {embeddings.shape}")
    
    # Check if embeddings are already normalized
    # If not, normalize for cosine similarity
    print("\nNormalizing embeddings for cosine similarity...")
    # Normalize in-place
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index
    print(f"\nBuilding FAISS index ({index_type})...")
    dimension = embeddings.shape[1]
    
    if index_type == "IndexFlatIP":
        # Use IndexFlatIP for exact inner product search (cosine similarity after normalization)
        index = faiss.IndexFlatIP(dimension)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    
    # Add embeddings
    print("Adding embeddings to index...")
    index.add(embeddings)
    
    print(f"✓ Index built: {index.ntotal} vectors")
    
    # Save index
    print(f"\nSaving FAISS index to {output_path}...")
    faiss.write_index(index, str(output_path))
    
    print(f"\n✓ FAISS index saved to {output_path}")
    print(f"✓ Index size: {index.ntotal} vectors")
    print(f"✓ Index dimension: {dimension}")
    
    return index


if __name__ == "__main__":
    build_faiss_index()

