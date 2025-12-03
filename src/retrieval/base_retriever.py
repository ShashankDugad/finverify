"""
Base retriever interface for all retrieval methods
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseRetriever(ABC):
    """Base class for all retrievers"""
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        doc_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            doc_filter: Optional document name to filter results
        
        Returns:
            List of dicts with keys: 'content', 'metadata', 'score'
        """
        pass
    
    def _format_result(
        self,
        chunk: Dict[str, Any],
        score: float
    ) -> Dict[str, Any]:
        """
        Format a chunk result with consistent structure.
        
        Args:
            chunk: Chunk dict with 'text' and 'metadata'
            score: Relevance score
        
        Returns:
            Formatted result dict
        """
        return {
            'content': chunk.get('text', ''),
            'metadata': chunk.get('metadata', {}),
            'score': float(score)
        }
    
    def _filter_by_doc(
        self,
        results: List[Dict[str, Any]],
        doc_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter results by document name.
        
        Args:
            results: List of result dicts
            doc_filter: Document name to filter by
        
        Returns:
            Filtered results
        """
        if doc_filter is None:
            return results
        
        filtered = []
        for result in results:
            metadata = result.get('metadata', {})
            doc_name = metadata.get('doc_name', '')
            if doc_filter.lower() in doc_name.lower() or doc_name.lower() in doc_filter.lower():
                filtered.append(result)
        
        return filtered

