"""
Cross-encoder reranker for improving retrieval relevance
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder


def get_base_dir():
    """Determine base directory"""
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


class CrossEncoderReranker:
    """
    Cross-encoder reranker for reranking retrieved chunks.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Name of cross-encoder model
            device: Device to use ('cuda' or 'cpu')
        """
        config = load_config()
        
        if model_name is None:
            model_name = config['models']['reranker']['name']
        
        if device is None:
            import torch
            device = config['models']['reranker'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading cross-encoder reranker: {model_name}...")
        self.model = CrossEncoder(model_name, device=device)
        self.device = device
        print(f"✓ Reranker loaded on {device}")
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks using cross-encoder.
        
        Args:
            query: Query string
            chunks: List of chunk dicts with 'content' key
            top_k: Number of top results to return
        
        Returns:
            Reranked list of chunk dicts with updated scores
        """
        if not chunks:
            return []
        
        # Prepare query-chunk pairs
        pairs = []
        for chunk in chunks:
            content = chunk.get('content', '')
            pairs.append([query, content])
        
        # Score pairs
        scores = self.model.predict(pairs)
        
        # Update scores and sort
        for i, chunk in enumerate(chunks):
            chunk['score'] = float(scores[i])
        
        # Sort by score (descending)
        reranked = sorted(chunks, key=lambda x: x['score'], reverse=True)
        
        return reranked[:top_k]

