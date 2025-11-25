"""
Chunk documents into 512-token segments with 50-token overlap
"""

import os
import json
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

def chunk_text(text, tokenizer, chunk_size=512, overlap=50):
    """Chunk text into overlapping segments"""
    
    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        
        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        chunks.append({
            'text': chunk_text,
            'token_count': len(chunk_tokens),
            'start_pos': start,
            'end_pos': end
        })
        
        # Move window with overlap
        start = end - overlap
        
        # Stop if we're at the end
        if end >= len(tokens):
            break
    
    return chunks

def process_sec_filings(tokenizer, input_dir, chunk_size=512, overlap=50):
    """Process SEC EDGAR filings"""
    
    print("\n" + "=" * 60)
    print("Processing SEC EDGAR Filings")
    print("=" * 60)
    
    sec_dir = Path(input_dir) / "sec_edgar" / "sec-edgar-filings"
    
    all_chunks = []
    file_count = 0
    
    # Find all .txt files
    txt_files = list(sec_dir.rglob("*.txt"))
    
    for filepath in tqdm(txt_files, desc="SEC Files"):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Skip if too short
            if len(text) < 100:
                continue
            
            chunks = chunk_text(text, tokenizer, chunk_size, overlap)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'chunk_id': f"sec_{file_count}_{i}",
                    'source': 'sec_edgar',
                    'source_file': str(filepath.name),
                    'text': chunk['text'],
                    'token_count': chunk['token_count']
                })
            
            file_count += 1
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    print(f"✓ Processed {file_count} SEC filings → {len(all_chunks)} chunks")
    return all_chunks

def process_financebench(tokenizer, input_dir, chunk_size=512, overlap=50):
    """Process FinanceBench dataset"""
    
    print("\n" + "=" * 60)
    print("Processing FinanceBench")
    print("=" * 60)
    
    filepath = Path(input_dir) / "financebench" / "financebench.json"
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    all_chunks = []
    
    for i, item in enumerate(tqdm(data, desc="FinanceBench")):
        text = item.get('context', '') or item.get('answer', '')
        
        if len(text) < 50:
            continue
        
        chunks = chunk_text(text, tokenizer, chunk_size, overlap)
        
        for j, chunk in enumerate(chunks):
            all_chunks.append({
                'chunk_id': f"fb_{i}_{j}",
                'source': 'financebench',
                'question': item.get('question', ''),
                'text': chunk['text'],
                'token_count': chunk['token_count']
            })
    
    print(f"✓ Processed {len(data)} examples → {len(all_chunks)} chunks")
    return all_chunks

def process_tatqa(tokenizer, input_dir, chunk_size=512, overlap=50):
    """Process TATQA dataset"""
    
    print("\n" + "=" * 60)
    print("Processing TATQA")
    print("=" * 60)
    
    tatqa_dir = Path(input_dir) / "tatqa"
    
    all_chunks = []
    
    for filename in ["tatqa_dataset_train.json", "tatqa_dataset_dev.json"]:
        filepath = tatqa_dir / filename
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for i, item in enumerate(tqdm(data, desc=filename)):
            # Combine paragraphs
            paragraphs = item.get('paragraphs', [])
            text = ' '.join(paragraphs) if isinstance(paragraphs, list) else str(paragraphs)
            
            if len(text) < 50:
                continue
            
            chunks = chunk_text(text, tokenizer, chunk_size, overlap)
            
            for j, chunk in enumerate(chunks):
                all_chunks.append({
                    'chunk_id': f"tatqa_{filename}_{i}_{j}",
                    'source': 'tatqa',
                    'question': item.get('question', ''),
                    'text': chunk['text'],
                    'token_count': chunk['token_count']
                })
    
    print(f"✓ Processed TATQA → {len(all_chunks)} chunks")
    return all_chunks

def main():
    """Main chunking pipeline"""
    
    print("=" * 60)
    print("Document Chunking Pipeline")
    print("=" * 60)
    
    # Initialize tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    
    input_dir = Path("/scratch") / os.environ['USER'] / "finverify" / "data" / "raw"
    output_dir = Path("/scratch") / os.environ['USER'] / "finverify" / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all datasets
    all_chunks = []
    
    all_chunks.extend(process_sec_filings(tokenizer, input_dir))
    all_chunks.extend(process_financebench(tokenizer, input_dir))
    all_chunks.extend(process_tatqa(tokenizer, input_dir))
    
    # Save chunks
    output_file = output_dir / "chunks.json"
    print(f"\n" + "=" * 60)
    print(f"Saving {len(all_chunks)} chunks to {output_file}")
    print("=" * 60)
    
    with open(output_file, 'w') as f:
        json.dump(all_chunks, f, indent=2)
    
    # Statistics
    avg_tokens = sum(c['token_count'] for c in all_chunks) / len(all_chunks)
    
    print(f"\n✓ Total chunks: {len(all_chunks)}")
    print(f"✓ Average tokens per chunk: {avg_tokens:.1f}")
    print(f"✓ Saved to: {output_file}")
    
    # Save metadata
    metadata = {
        'total_chunks': len(all_chunks),
        'avg_tokens': avg_tokens,
        'chunk_size': 512,
        'overlap': 50
    }
    
    with open(output_dir / 'chunks_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()
