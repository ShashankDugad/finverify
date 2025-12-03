"""
Data loading utilities for FinanceBench
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional


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


def load_financebench_questions(data_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load FinanceBench questions from JSONL file.
    
    Args:
        data_path: Path to financebench_open_source.jsonl (if None, uses config)
    
    Returns:
        List of question dicts
    """
    config = load_config()
    base_dir = get_base_dir()
    
    if data_path is None:
        # Try multiple possible locations
        possible_paths = [
            base_dir / config['paths']['data_raw'] / "financebench" / "financebench_open_source.jsonl",
            base_dir.parent / "financebench" / "data" / "financebench_open_source.jsonl",
            base_dir / "financebench" / "data" / "financebench_open_source.jsonl"
        ]
        
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None or not data_path.exists():
            raise FileNotFoundError(f"FinanceBench questions file not found. Tried: {possible_paths}")
    
    questions = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    return questions


def load_financebench_metadata(data_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load FinanceBench document metadata.
    
    Args:
        data_path: Path to financebench_document_information.jsonl (if None, uses config)
    
    Returns:
        Dict mapping doc_name to metadata dict
    """
    config = load_config()
    base_dir = get_base_dir()
    
    if data_path is None:
        # Try multiple possible locations
        possible_paths = [
            base_dir / config['paths']['data_raw'] / "financebench" / "financebench_document_information.jsonl",
            base_dir.parent / "financebench" / "data" / "financebench_document_information.jsonl",
            base_dir / "financebench" / "data" / "financebench_document_information.jsonl"
        ]
        
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None or not data_path.exists():
            print(f"Warning: FinanceBench metadata file not found. Tried: {possible_paths}")
            return {}
    
    metadata = {}
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                doc_name = item.get('doc_name', '')
                if doc_name:
                    metadata[doc_name] = item
    
    return metadata


def get_pdf_path(doc_name: str, pdfs_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Get PDF path from document name.
    
    Args:
        doc_name: Document name (without .pdf extension)
        pdfs_dir: Directory containing PDFs (if None, uses config)
    
    Returns:
        Path to PDF file or None if not found
    """
    config = load_config()
    base_dir = get_base_dir()
    
    if pdfs_dir is None:
        # Try multiple possible locations
        possible_dirs = [
            base_dir / config['paths']['data_raw'] / "financebench" / "pdfs",
            base_dir.parent / "financebench" / "pdfs",
            base_dir / "financebench" / "pdfs"
        ]
        
        for dir_path in possible_dirs:
            if dir_path.exists():
                pdfs_dir = dir_path
                break
        
        if pdfs_dir is None:
            return None
    
    # Try with and without .pdf extension
    pdf_path = pdfs_dir / f"{doc_name}.pdf"
    if pdf_path.exists():
        return pdf_path
    
    # Try case-insensitive search
    for pdf_file in pdfs_dir.glob("*.pdf"):
        if pdf_file.stem.lower() == doc_name.lower():
            return pdf_file
    
    return None


def load_chunks(chunks_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load processed chunks from JSON or JSONL file.
    
    Args:
        chunks_path: Path to chunks file (if None, uses config)
    
    Returns:
        List of chunk dicts
    """
    config = load_config()
    base_dir = get_base_dir()
    
    if chunks_path is None:
        chunks_path = base_dir / config['paths']['data_processed'] / "chunks.jsonl"
        if not chunks_path.exists():
            chunks_path = base_dir / config['paths']['data_processed'] / "chunks.json"
    
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    
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

