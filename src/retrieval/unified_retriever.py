"""
Unified retriever combining multiple retrieval methods with RRF
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict

from .base_retriever import BaseRetriever
from .faiss_retriever import FaissRetriever
from .entity_retriever import EntityRetriever
from .temporal_retriever import TemporalRetriever
from .reranker import CrossEncoderReranker


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


def reciprocal_rank_fusion(
    results_list: List[List[Dict[str, Any]]],
    k: int = 60,
    top_k: int = 100
) -> List[Dict[str, Any]]:
    """
    Combine multiple retrieval results using Reciprocal Rank Fusion (RRF).
    
    Args:
        results_list: List of result lists from different retrievers
        k: RRF constant (default 60)
        top_k: Maximum number of results to return
    
    Returns:
        Merged and reranked results
    """
    # Score each result by its rank in each method
    doc_scores = defaultdict(float)
    doc_results = {}  # Store first occurrence of each result
    
    for results in results_list:
        for rank, result in enumerate(results, start=1):
            # Use content as key to identify duplicates
            content = result.get('content', '')
            if not content:
                continue
            
            # RRF score: 1 / (k + rank)
            score = 1.0 / (k + rank)
            doc_scores[content] += score
            
            # Store result if first time seeing it
            if content not in doc_results:
                doc_results[content] = result
    
    # Sort by combined score
    sorted_docs = sorted(
        doc_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Return top_k results
    merged_results = []
    for content, score in sorted_docs[:top_k]:
        result = doc_results[content].copy()
        result['score'] = score  # Update with RRF score
        merged_results.append(result)
    
    return merged_results


class UnifiedRetriever:
    """
    Unified retriever combining multiple retrieval methods.
    """
    
    def __init__(
        self,
        methods: Optional[List[str]] = None,
        use_reranker: bool = True
    ):
        """
        Initialize unified retriever.
        
        Args:
            methods: List of methods to use ['faiss', 'entity', 'temporal'] or None for all
            use_reranker: Whether to apply reranker after retrieval
        """
        config = load_config()
        
        if methods is None:
            methods = ['faiss', 'entity', 'temporal']
        
        self.methods = methods
        self.use_reranker = use_reranker and config['retrieval']['reranking'].get('enabled', True)
        self.rrf_k = config['retrieval']['rrf'].get('k_value', 60)
        self.rrf_top_k = config['retrieval']['rrf'].get('merge_top_k', 100)
        self.final_top_k = config['retrieval']['reranking'].get('final_top_k', 10)
        
        # Initialize retrievers
        self.retrievers = {}
        
        if 'faiss' in methods:
            print("Initializing FAISS retriever...")
            self.retrievers['faiss'] = FaissRetriever()
        
        if 'entity' in methods:
            print("Initializing Entity retriever...")
            self.retrievers['entity'] = EntityRetriever()
        
        if 'temporal' in methods:
            print("Initializing Temporal retriever...")
            self.retrievers['temporal'] = TemporalRetriever()
        
        # Initialize reranker if enabled
        if self.use_reranker:
            print("Initializing Reranker...")
            self.reranker = CrossEncoderReranker()
        else:
            self.reranker = None
        
        print(f"✓ Unified retriever initialized with methods: {methods}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        doc_filter: Optional[str] = None,
        methods: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using multiple methods and combine with RRF.
        
        Args:
            query: Query string
            top_k: Number of results to return (uses config if None)
            doc_filter: Optional document name to filter results
            methods: Override methods to use (if None, uses initialized methods)
        
        Returns:
            List of result dicts with 'content', 'metadata', 'score'
        """
        if top_k is None:
            top_k = self.final_top_k
        
        if methods is None:
            methods = self.methods
        
        # Retrieve from each method
        all_results = []
        for method_name in methods:
            if method_name not in self.retrievers:
                continue
            
            retriever = self.retrievers[method_name]
            
            # Get more results for RRF (will be merged)
            method_top_k = self.rrf_top_k if len(methods) > 1 else top_k
            
            try:
                results = retriever.retrieve(query, top_k=method_top_k, doc_filter=doc_filter)
                all_results.append(results)
            except Exception as e:
                print(f"Error in {method_name} retriever: {e}")
                continue
        
        if not all_results:
            return []
        
        # Combine results using RRF if multiple methods
        if len(all_results) > 1:
            merged_results = reciprocal_rank_fusion(
                all_results,
                k=self.rrf_k,
                top_k=self.rrf_top_k
            )
        else:
            merged_results = all_results[0]
        
        # Apply reranker if enabled
        if self.reranker and len(merged_results) > 0:
            merged_results = self.reranker.rerank(
                query,
                merged_results,
                top_k=top_k
            )
        
        return merged_results[:top_k]

