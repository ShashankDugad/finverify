"""
Temporal retriever based on document periods and extracted dates
"""

import json
import yaml
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import spacy

from .base_retriever import BaseRetriever


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


class TemporalRetriever(BaseRetriever):
    """
    Temporal retriever based on document periods and date extraction.
    """
    
    # Year pattern (4 digits)
    YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')
    
    def __init__(
        self,
        chunks_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        ner_model: str = "en_core_web_sm"
    ):
        """
        Initialize temporal retriever.
        
        Args:
            chunks_path: Path to chunks JSON/JSONL file
            metadata_path: Path to FinanceBench document metadata JSONL
            ner_model: spaCy NER model name
        """
        config = load_config()
        base_dir = get_base_dir()
        
        # Set paths
        if chunks_path is None:
            chunks_path = base_dir / config['paths']['data_processed'] / "chunks.jsonl"
            if not chunks_path.exists():
                chunks_path = base_dir / config['paths']['data_processed'] / "chunks.json"
        
        if metadata_path is None:
            # Try FinanceBench metadata
            metadata_path = base_dir / config['paths']['data_raw'] / "financebench" / "financebench_document_information.jsonl"
            if not metadata_path.exists():
                # Try alternative location
                metadata_path = base_dir.parent / "financebench" / "data" / "financebench_document_information.jsonl"
        
        # Load spaCy model
        print(f"Loading spaCy NER model: {ner_model}...")
        try:
            self.nlp = spacy.load(ner_model)
        except OSError:
            print(f"Model {ner_model} not found. Please install: python -m spacy download {ner_model}")
            raise
        
        # Load chunks
        print(f"Loading chunks from {chunks_path}...")
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
        
        self.chunks = self._load_chunks(chunks_path)
        print(f"✓ Loaded {len(self.chunks)} chunks")
        
        # Load document metadata
        self.doc_periods = self._load_metadata(metadata_path)
        print(f"✓ Loaded metadata for {len(self.doc_periods)} documents")
        
        # Extract dates from chunks
        self.chunk_dates = self._extract_chunk_dates()
        print(f"✓ Extracted dates from chunks")
    
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
    
    def _load_metadata(self, metadata_path: Path) -> Dict[str, str]:
        """
        Load document period metadata.
        
        Returns:
            Dict mapping doc_name to doc_period
        """
        doc_periods = {}
        
        if metadata_path and metadata_path.exists():
            print(f"Loading metadata from {metadata_path}...")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        doc_name = item.get('doc_name', '')
                        doc_period = item.get('doc_period', '')
                        if doc_name and doc_period:
                            doc_periods[doc_name] = str(doc_period)
        else:
            print(f"Metadata file not found: {metadata_path}, continuing without metadata")
        
        return doc_periods
    
    def _extract_chunk_dates(self) -> Dict[int, List[str]]:
        """
        Extract dates from chunks using NER.
        
        Returns:
            Dict mapping chunk_idx to list of extracted years
        """
        chunk_dates = defaultdict(list)
        
        for chunk_idx, chunk in enumerate(self.chunks):
            text = chunk.get('text', '')
            doc = self.nlp(text)
            
            # Extract DATE entities
            for ent in doc.ents:
                if ent.label_ == 'DATE':
                    # Extract years from date strings
                    years = self.YEAR_PATTERN.findall(ent.text)
                    for year_match in years:
                        year = ''.join(year_match)
                        if year not in chunk_dates[chunk_idx]:
                            chunk_dates[chunk_idx].append(year)
            
            # Also extract standalone years
            years = self.YEAR_PATTERN.findall(text)
            for year in years:
                year_str = ''.join(year)
                if year_str not in chunk_dates[chunk_idx]:
                    chunk_dates[chunk_idx].append(year_str)
        
        return dict(chunk_dates)
    
    def _extract_query_years(self, query: str) -> List[str]:
        """Extract years from query"""
        years = self.YEAR_PATTERN.findall(query)
        return [''.join(year) for year in years]
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        doc_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks based on temporal matching.
        
        Args:
            query: Query string
            top_k: Number of results to return
            doc_filter: Optional document name to filter results
        
        Returns:
            List of result dicts with 'content', 'metadata', 'score'
        """
        # Extract years from query
        query_years = self._extract_query_years(query)
        
        if not query_years:
            return []  # No temporal information in query
        
        # Score chunks based on temporal matches
        chunk_scores = defaultdict(float)
        
        for chunk_idx, chunk in enumerate(self.chunks):
            score = 0.0
            
            # Check document period
            metadata = chunk.get('metadata', {})
            doc_name = metadata.get('doc_name', '')
            doc_period = self.doc_periods.get(doc_name, '')
            
            if doc_period:
                for year in query_years:
                    if year in doc_period:
                        score += 2.0  # High weight for doc_period match
            
            # Check extracted dates from chunk
            chunk_years = self.chunk_dates.get(chunk_idx, [])
            for year in query_years:
                if year in chunk_years:
                    score += 1.0  # Lower weight for date mention
            
            if score > 0:
                chunk_scores[chunk_idx] = score
        
        # Sort by score
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format results
        results = []
        for chunk_idx, score in sorted_chunks[:top_k * 2]:  # Get more for filtering
            if chunk_idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[chunk_idx]
            result = self._format_result(chunk, score)
            results.append(result)
        
        # Filter by document if requested
        if doc_filter:
            results = self._filter_by_doc(results, doc_filter)
        
        # Return top_k
        return results[:top_k]

