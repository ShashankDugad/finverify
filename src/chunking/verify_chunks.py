"""
Verify chunk integrity and completeness
"""

from pathlib import Path
import json
from collections import defaultdict, Counter
from typing import Dict, List, Set
import yaml


def verify_chunks(
    chunks_path: Path,
    pdfs_dir: Path,
    verbose: bool = True
) -> Dict:
    """
    Verify chunk integrity and completeness.
    
    Returns:
        Dict with verification results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # 1. Load all chunks
    chunks = []
    try:
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        chunks.append(chunk)
                    except json.JSONDecodeError as e:
                        results['errors'].append(f"Line {line_num}: Invalid JSON - {e}")
                        results['valid'] = False
    except Exception as e:
        results['errors'].append(f"Could not read chunks file: {e}")
        results['valid'] = False
        return results
    
    results['stats']['total_chunks'] = len(chunks)
    
    # 2. Validate chunk structure
    required_fields = ['text', 'metadata']
    required_metadata = ['doc_name', 'chunk_id', 'page_num', 'chunk_size']
    
    for idx, chunk in enumerate(chunks):
        # Check required fields
        for field in required_fields:
            if field not in chunk:
                results['errors'].append(f"Chunk {idx}: Missing field '{field}'")
                results['valid'] = False
        
        # Check metadata
        if 'metadata' in chunk:
            for meta_field in required_metadata:
                if meta_field not in chunk['metadata']:
                    results['warnings'].append(
                        f"Chunk {idx}: Missing metadata '{meta_field}'"
                    )
        
        # Check text is not empty
        if 'text' in chunk and not chunk['text'].strip():
            results['warnings'].append(f"Chunk {idx}: Empty text content")
    
    # 3. Group chunks by document
    doc_chunks = defaultdict(list)
    for chunk in chunks:
        doc_name = chunk.get('metadata', {}).get('doc_name', '')
        if doc_name:
            doc_chunks[doc_name].append(chunk)
    
    results['stats']['processed_documents'] = len(doc_chunks)
    results['stats']['chunks_per_doc'] = {
        doc: len(chunks) for doc, chunks in doc_chunks.items()
    }
    
    # 4. Check for missing chunk_ids within documents
    for doc_name, doc_chunk_list in doc_chunks.items():
        chunk_ids = sorted([
            c.get('metadata', {}).get('chunk_id', -1)
            for c in doc_chunk_list
        ])
        
        # Check for gaps in chunk_ids
        if chunk_ids:
            expected_ids = set(range(len(chunk_ids)))
            actual_ids = set(chunk_ids)
            missing_ids = expected_ids - actual_ids
            
            if missing_ids:
                results['warnings'].append(
                    f"Document '{doc_name}': Missing chunk_ids {sorted(list(missing_ids))[:10]}"
                )
            
            # Check if chunk_ids are sequential (should start from 0)
            if chunk_ids[0] != 0:
                results['warnings'].append(
                    f"Document '{doc_name}': chunk_ids don't start from 0 (starts at {chunk_ids[0]})"
                )
    
    # 5. Compare with PDF files
    pdf_files = list(pdfs_dir.rglob("*.pdf"))
    pdf_stems = {pdf.stem for pdf in pdf_files}
    processed_stems = set(doc_chunks.keys())
    
    results['stats']['total_pdfs'] = len(pdf_files)
    results['stats']['processed_pdfs'] = len(processed_stems)
    results['stats']['unprocessed_pdfs'] = list(pdf_stems - processed_stems)
    results['stats']['missing_pdfs'] = list(processed_stems - pdf_stems)  # Chunks without PDFs
    
    if pdf_stems - processed_stems:
        unprocessed = sorted(list(pdf_stems - processed_stems))
        results['warnings'].append(
            f"Found {len(unprocessed)} PDFs without chunks: {unprocessed[:5]}..."
        )
    
    if processed_stems - pdf_stems:
        missing = sorted(list(processed_stems - pdf_stems))
        results['warnings'].append(
            f"Found {len(missing)} chunks without matching PDFs: {missing[:5]}..."
        )
    
    # 6. Check for duplicate chunks (same doc_name + chunk_id)
    duplicates = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        doc_name = chunk.get('metadata', {}).get('doc_name', '')
        chunk_id = chunk.get('metadata', {}).get('chunk_id', -1)
        key = (doc_name, chunk_id)
        duplicates[key].append(idx)
    
    duplicate_keys = {k: v for k, v in duplicates.items() if len(v) > 1}
    if duplicate_keys:
        results['errors'].append(
            f"Found {len(duplicate_keys)} duplicate chunk keys: {list(duplicate_keys.keys())[:5]}..."
        )
        results['valid'] = False
    
    # 7. Check file size consistency
    total_text_size = sum(len(c.get('text', '')) for c in chunks)
    results['stats']['total_text_size'] = total_text_size
    results['stats']['avg_chunk_size'] = total_text_size / len(chunks) if chunks else 0
    
    return results


def print_verification_report(results: Dict):
    """Print verification results in a readable format."""
    print("=" * 60)
    print("Chunk Verification Report")
    print("=" * 60)
    
    print(f"\nStatus: {'✓ VALID' if results['valid'] else '✗ INVALID'}")
    
    print(f"\nStatistics:")
    stats = results['stats']
    print(f"  Total chunks: {stats.get('total_chunks', 0):,}")
    print(f"  Processed documents: {stats.get('processed_documents', 0)}")
    print(f"  Total PDFs: {stats.get('total_pdfs', 0)}")
    print(f"  Unprocessed PDFs: {len(stats.get('unprocessed_pdfs', []))}")
    print(f"  Average chunk size: {stats.get('avg_chunk_size', 0):.0f} chars")
    print(f"  Total text size: {stats.get('total_text_size', 0):,} chars")
    
    if results['errors']:
        print(f"\n✗ Errors ({len(results['errors'])}):")
        for error in results['errors'][:10]:
            print(f"  - {error}")
        if len(results['errors']) > 10:
            print(f"  ... and {len(results['errors']) - 10} more errors")
    
    if results['warnings']:
        print(f"\n⚠ Warnings ({len(results['warnings'])}):")
        for warning in results['warnings'][:10]:
            print(f"  - {warning}")
        if len(results['warnings']) > 10:
            print(f"  ... and {len(results['warnings']) - 10} more warnings")
    
    if not results['errors'] and not results['warnings']:
        print("\n✓ All checks passed! Chunks are valid and complete.")
    
    print("=" * 60)


if __name__ == "__main__":
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_dir = Path(__file__).parent.parent.parent
    chunks_path = base_dir / config['paths']['data_processed'] / "chunks.jsonl"
    pdfs_dir = base_dir / "data" / "pdfs"
    
    if not chunks_path.exists():
        print(f"Error: Chunks file not found: {chunks_path}")
    else:
        print(f"Verifying chunks from: {chunks_path}")
        print(f"Comparing with PDFs in: {pdfs_dir}\n")
        results = verify_chunks(chunks_path, pdfs_dir)
        print_verification_report(results)

