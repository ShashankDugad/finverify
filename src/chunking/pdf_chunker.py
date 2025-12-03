"""
PDF chunking module with overlap support for FinanceBench documents
"""

from pathlib import Path
from typing import List, Dict, Any
import json
import yaml
from unstructured.partition.pdf import partition_pdf
from tqdm import tqdm


def chunk_pdf(
    pdf_path: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    doc_name: str = None
) -> List[Dict[str, Any]]:
    """
    Extract and chunk a PDF with configurable overlap.
    
    Args:
        pdf_path: Path to PDF file
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        doc_name: Document name (if None, uses pdf_path.stem)
    
    Returns:
        List of dicts with 'text' and 'metadata' keys
    """
    if doc_name is None:
        doc_name = pdf_path.stem
    
    # Extract elements from PDF
    elements = partition_pdf(pdf_path, url=None, infer_table_structure=False)
    
    chunks = []
    current_chunk = []
    current_len = 0
    chunk_id = 0
    page_num = 1
    
    for element in elements:
        element_str = str(element)
        element_len = len(element_str)
        
        # Try to extract page number from element metadata if available
        if hasattr(element, 'metadata') and element.metadata:
            if hasattr(element.metadata, 'page_number'):
                page_num = element.metadata.page_number
        
        # If adding this element would exceed chunk_size, finalize current chunk
        if current_len + element_len > chunk_size and current_chunk:
            # Create chunk with overlap
            chunk_text = "\n".join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'doc_name': doc_name,
                    'chunk_id': chunk_id,
                    'page_num': page_num,
                    'chunk_size': len(chunk_text)
                }
            })
            chunk_id += 1
            
            # Start new chunk with overlap
            # Keep last elements that fit within overlap size
            overlap_text = ""
            overlap_elements = []
            for elem in reversed(current_chunk):
                if len(overlap_text) + len(elem) <= chunk_overlap:
                    overlap_elements.insert(0, elem)
                    overlap_text = "\n".join(overlap_elements)
                else:
                    break
            
            current_chunk = overlap_elements
            current_len = len(overlap_text)
        
        # Add element to current chunk
        current_chunk.append(element_str)
        current_len += element_len
    
    # Add final chunk if any remaining
    if current_chunk:
        chunk_text = "\n".join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'metadata': {
                'doc_name': doc_name,
                'chunk_id': chunk_id,
                'page_num': page_num,
                'chunk_size': len(chunk_text)
            }
        })
    
    return chunks


def chunk_pdfs_directory(
    pdfs_dir: Path,
    output_path: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 128
) -> List[Dict[str, Any]]:
    """
    Chunk all PDFs in a directory and save to JSONL file.
    
    Args:
        pdfs_dir: Directory containing PDF files
        output_path: Path to save chunks JSONL file
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks in characters
    
    Returns:
        List of all chunks
    """
    pdf_files = list(pdfs_dir.rglob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdfs_dir}")
        return []
    
    all_chunks = []
    
    print(f"Processing {len(pdf_files)} PDF files...")
    for pdf_path in tqdm(pdf_files, desc="Chunking PDFs"):
        try:
            chunks = chunk_pdf(pdf_path, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            continue
    
    # Save to JSONL format
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(all_chunks)} chunks to {output_path}")
    return all_chunks


if __name__ == "__main__":
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get paths from config
    data_raw = Path(config['paths']['data_raw'])
    data_processed = Path(config['paths']['data_processed'])
    chunk_size = config['data']['chunk_size']
    chunk_overlap = config['data']['chunk_overlap']
    
    # FinanceBench PDFs directory
    financebench_pdfs = data_raw / "financebench" / "pdfs"
    
    if not financebench_pdfs.exists():
        print(f"FinanceBench PDFs directory not found: {financebench_pdfs}")
        print("Please download FinanceBench PDFs first")
    else:
        output_file = data_processed / "chunks.jsonl"
        chunk_pdfs_directory(financebench_pdfs, output_file, chunk_size, chunk_overlap)

