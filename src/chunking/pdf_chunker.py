"""
PDF chunking module with overlap support for FinanceBench documents
"""

from pathlib import Path
from typing import List, Dict, Any
import json
import yaml
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Try to import PDF parsing libraries with fallbacks
try:
    from unstructured.partition.pdf import partition_pdf
    HAS_UNSTRUCTURED = True
except ImportError:
    HAS_UNSTRUCTURED = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


def _extract_pdf_elements(pdf_path: Path) -> List[str]:
    """
    Extract text elements from PDF using multiple fallback methods.
    
    Tries in order:
    1. unstructured (poppler-based)
    2. PyMuPDF (fitz) - more robust for complex PDFs
    3. pdfplumber - good for text extraction
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        List of text elements (strings)
    """
    elements = []
    
    # Method 1: Try unstructured (poppler-based)
    if HAS_UNSTRUCTURED:
        try:
            unstructured_elements = partition_pdf(
                pdf_path, 
                url=None, 
                infer_table_structure=False,
                strategy="hi_res"  # Use high-res strategy for better parsing
            )
            elements = [str(elem) for elem in unstructured_elements if str(elem).strip()]
            if elements:
                return elements
        except Exception as e:
            # If unstructured fails, try fallback methods
            pass
    
    # Method 2: Fallback to PyMuPDF (more robust)
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(pdf_path)
            elements = []
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                if text.strip():
                    elements.append(text)
            doc.close()
            if elements:
                return elements
        except Exception as e:
            pass
    
    # Method 3: Fallback to pdfplumber
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                elements = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text and text.strip():
                        elements.append(text)
            if elements:
                return elements
        except Exception as e:
            pass
    
    # If all methods fail, return empty list
    return []


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
    
    # Extract elements from PDF with fallback mechanisms
    elements = _extract_pdf_elements(pdf_path)
    
    if not elements:
        raise ValueError(f"Could not extract any content from PDF: {pdf_path}")
    
    chunks = []
    current_chunk = []
    current_len = 0
    chunk_id = 0
    page_num = 1
    
    for element_idx, element in enumerate(elements):
        # Elements are now always strings from _extract_pdf_elements
        element_str = str(element) if not isinstance(element, str) else element
        # Use element index as page number estimate (fallback methods return page-level text)
        page_num = element_idx + 1
        
        element_len = len(element_str)
        
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
    chunk_overlap: int = 128,
    num_workers: int = None,
    resume: bool = True
) -> List[Dict[str, Any]]:
    """
    Chunk all PDFs in a directory and save to JSONL file (parallelized).
    Supports resuming from previous runs.
    
    Args:
        pdfs_dir: Directory containing PDF files
        output_path: Path to save chunks JSONL file
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        num_workers: Number of parallel workers (default: CPU count)
        resume: If True, skip already processed PDFs (default: True)
    
    Returns:
        List of all chunks (including existing ones if resume=True)
    """
    pdf_files = list(pdfs_dir.rglob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdfs_dir}")
        return []
    
    # Load existing chunks if resuming
    existing_chunks = []
    processed_docs = set()
    
    if resume and output_path.exists():
        print(f"Loading existing chunks from {output_path}...")
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        chunk = json.loads(line)
                        existing_chunks.append(chunk)
                        # Track which documents have been processed
                        doc_name = chunk.get('metadata', {}).get('doc_name', '')
                        if doc_name:
                            processed_docs.add(doc_name)
            
            print(f"✓ Found {len(existing_chunks)} existing chunks from {len(processed_docs)} documents")
        except Exception as e:
            print(f"⚠ Could not load existing chunks: {e}")
            existing_chunks = []
            processed_docs = set()
    
    # Filter out already processed PDFs
    if resume and processed_docs:
        original_count = len(pdf_files)
        pdf_files = [
            pdf for pdf in pdf_files 
            if pdf.stem not in processed_docs
        ]
        skipped_count = original_count - len(pdf_files)
        if skipped_count > 0:
            print(f"⏭ Skipping {skipped_count} already processed PDFs")
    
    if not pdf_files:
        print("✓ All PDFs have already been processed!")
        return existing_chunks
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    print(f"Processing {len(pdf_files)} new PDF files with {num_workers} workers...")
    
    # Worker function for parallel processing
    def process_pdf(pdf_path):
        try:
            chunks = chunk_pdf(pdf_path, chunk_size, chunk_overlap)
            return pdf_path, chunks, None
        except Exception as e:
            return pdf_path, [], str(e)
    
    new_chunks = []
    failed_files = []
    
    # Process PDFs in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_pdf = {
            executor.submit(process_pdf, pdf_path): pdf_path 
            for pdf_path in pdf_files
        }
        
        # Collect results as they complete
        for future in tqdm(as_completed(future_to_pdf), total=len(pdf_files), desc="Chunking PDFs"):
            pdf_path, chunks, error = future.result()
            
            if error:
                print(f"Error processing {pdf_path}: {error}")
                failed_files.append(pdf_path)
            else:
                new_chunks.extend(chunks)
    
    # Combine existing and new chunks
    all_chunks = existing_chunks + new_chunks
    
    # Save all chunks to JSONL format (overwrite to include new chunks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(all_chunks)} total chunks ({len(new_chunks)} new) to {output_path}")
    if failed_files:
        print(f"⚠ Failed to process {len(failed_files)} files")
    
    return all_chunks


if __name__ == "__main__":
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get paths from config
    base_dir = Path(__file__).parent.parent.parent
    data_processed = Path(config['paths']['data_processed'])
    chunk_size = config['data']['chunk_size']
    chunk_overlap = config['data']['chunk_overlap']
    
    # PDFs directory - directly from data/pdfs
    financebench_pdfs = base_dir / "data" / "pdfs"
    
    if not financebench_pdfs.exists():
        print(f"PDFs directory not found: {financebench_pdfs}")
        print("Please add PDF files to data/pdfs folder first")
    else:
        output_file = data_processed / "chunks.jsonl"
        chunk_pdfs_directory(financebench_pdfs, output_file, chunk_size, chunk_overlap)

