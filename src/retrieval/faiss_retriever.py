"""
FAISS semantic retriever (Baseline Method 1)
"""

import json
import numpy as np
import faiss
import yaml
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

from .base_retriever import BaseRetriever


def get_base_dir():
    """Determine base directory (local workspace or HPC scratch)"""
    workspace_path = Path(__file__).parent.parent.parent.parent
    if '/scratch/' in str(workspace_path) or 'USER' in str(workspace_path):
        import os
        return Path("/scratch") / os.environ.get('USER', 'user') / "finverify"
    return workspace_path


def load_config():
    """Load config.yaml"""
    base_dir = get_base_dir()
    config_path = base_dir / "config.yaml"
    if not config_path.exists():
        config_path = Path(__file__).parent.parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class FaissRetriever(BaseRetriever):
    """
    FAISS-based semantic retriever using BGE embeddings.
    """
    
    def __init__(
        self,
        index_path: Optional[Path] = None,
        chunks_path: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize FAISS retriever.
        
        Args:
            index_path: Path to FAISS index file
            chunks_path: Path to chunks JSON/JSONL file
            embedding_model: Name of embedding model
            device: Device to use ('cuda' or 'cpu')
        """
        config = load_config()
        base_dir = get_base_dir()
        
        # Set paths
        if index_path is None:
            index_path = base_dir / config['paths']['indexes'] / "faiss" / "faiss_index.bin"
        if chunks_path is None:
            chunks_path = base_dir / config['paths']['data_processed'] / "chunks.jsonl"
            if not chunks_path.exists():
                chunks_path = base_dir / config['paths']['data_processed'] / "chunks.json"
        
        if embedding_model is None:
            embedding_model = config['models']['embedder']['name']
        
        if device is None:
            device = config['models']['embedder'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load FAISS index
        print(f"Loading FAISS index from {index_path}...")
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        print(f"✓ Loaded FAISS index: {self.index.ntotal:,} vectors")
        
        # Load chunks
        print(f"Loading chunks from {chunks_path}...")
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
        
        self.chunks = self._load_chunks(chunks_path)
        print(f"✓ Loaded {len(self.chunks)} chunks")
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model}...")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        self.device = device
        print(f"✓ Model loaded on {device}")
    
    def _load_chunks(self, chunks_path: Path) -> List[Dict]:
        """Load chunks from JSON or JSONL file"""
        chunks = []
        if chunks_path.suffix == '.jsonl':
            with open(chunks_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        chunks.append(json.loads(line))
        else:
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
        return chunks
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        doc_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using FAISS semantic search.
        
        Args:
            query: Query string
            top_k: Number of results to return
            doc_filter: Optional document name to filter results
        
        Returns:
            List of result dicts with 'content', 'metadata', 'score'
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k * 2)  # Get more for filtering
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks):  # Invalid index
                continue
            
            chunk = self.chunks[idx]
            score = float(scores[0][i])
            
            result = self._format_result(chunk, score)
            results.append(result)
        
        # Filter by document if requested
        if doc_filter:
            results = self._filter_by_doc(results, doc_filter)
        
        # Return top_k
        return results[:top_k]

