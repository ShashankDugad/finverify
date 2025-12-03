"""
Entity-based NER retriever for matching named entities
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
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


class EntityRetriever(BaseRetriever):
    """
    Entity-based retriever using Named Entity Recognition (NER).
    """
    
    # Entity types to prioritize (financial entities get higher weight)
    ENTITY_WEIGHTS = {
        'ORG': 2.0,      # Organizations (companies)
        'PERSON': 1.5,   # People
        'MONEY': 2.0,    # Monetary amounts
        'DATE': 1.0,     # Dates
        'GPE': 1.5,      # Countries, cities
        'PRODUCT': 1.5,  # Products
    }
    
    def __init__(
        self,
        chunks_path: Optional[Path] = None,
        ner_model: str = "en_core_web_sm",
        entity_index_path: Optional[Path] = None
    ):
        """
        Initialize entity retriever.
        
        Args:
            chunks_path: Path to chunks JSON/JSONL file
            ner_model: spaCy NER model name
            entity_index_path: Path to save/load entity index cache
        """
        config = load_config()
        base_dir = get_base_dir()
        
        # Set paths
        if chunks_path is None:
            chunks_path = base_dir / config['paths']['data_processed'] / "chunks.jsonl"
            if not chunks_path.exists():
                chunks_path = base_dir / config['paths']['data_processed'] / "chunks.json"
        
        if ner_model is None:
            ner_model = config['retrieval']['mainrag']['entity'].get('ner_model', 'en_core_web_sm')
        
        if entity_index_path is None:
            entity_index_path = base_dir / config['paths']['indexes'] / "entity_index.json"
        
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
        
        # Build or load entity index
        self.entity_index = self._build_entity_index(entity_index_path)
        print(f"✓ Entity index ready: {len(self.entity_index)} unique entities")
    
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
    
    def _build_entity_index(self, cache_path: Path) -> Dict[str, List[int]]:
        """
        Build entity-to-chunk mapping index.
        
        Args:
            cache_path: Path to save/load cached index
        
        Returns:
            Dict mapping entity text to list of chunk indices
        """
        # Try to load cached index
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                    # Verify it matches current chunks
                    if len(cached.get('chunk_count', 0)) == len(self.chunks):
                        print(f"✓ Loaded cached entity index from {cache_path}")
                        return {k: v for k, v in cached['index'].items()}
            except Exception as e:
                print(f"Could not load cached index: {e}, rebuilding...")
        
        # Build index
        print("Building entity index...")
        entity_index = defaultdict(list)
        
        for chunk_idx, chunk in enumerate(self.chunks):
            text = chunk.get('text', '')
            doc = self.nlp(text)
            
            # Extract entities
            for ent in doc.ents:
                entity_text = ent.text.lower().strip()
                if entity_text and len(entity_text) > 1:  # Filter very short entities
                    entity_index[entity_text].append(chunk_idx)
        
        # Convert to regular dict
        entity_index = dict(entity_index)
        
        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({
                'index': entity_index,
                'chunk_count': len(self.chunks)
            }, f)
        
        return entity_index
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
        
        Returns:
            List of entity dicts with 'text', 'label', 'weight'
        """
        doc = self.nlp(text)
        entities = []
        seen = set()
        
        for ent in doc.ents:
            entity_text = ent.text.lower().strip()
            if entity_text and entity_text not in seen and len(entity_text) > 1:
                seen.add(entity_text)
                weight = self.ENTITY_WEIGHTS.get(ent.label_, 1.0)
                entities.append({
                    'text': entity_text,
                    'label': ent.label_,
                    'weight': weight
                })
        
        return entities
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        doc_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks based on entity matching.
        
        Args:
            query: Query string
            top_k: Number of results to return
            doc_filter: Optional document name to filter results
        
        Returns:
            List of result dicts with 'content', 'metadata', 'score'
        """
        # Extract entities from query
        query_entities = self.extract_entities(query)
        
        if not query_entities:
            return []  # No entities found
        
        # Score chunks based on entity matches
        chunk_scores = defaultdict(float)
        
        for entity in query_entities:
            entity_text = entity['text']
            weight = entity['weight']
            
            # Find chunks containing this entity
            matching_chunks = self.entity_index.get(entity_text, [])
            
            for chunk_idx in matching_chunks:
                chunk_scores[chunk_idx] += weight
        
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

