"""
Fast chunking - process text directly without slow tokenization
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

def quick_chunk_text(text, chunk_size=2048, overlap=200):
    """Quick character-based chunking"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk_text = text[start:end]
        
        if len(chunk_text.strip()) > 50:
            chunks.append(chunk_text)
        
        start = end - overlap
        
        if end >= text_len:
            break
    
    return chunks

def process_sec_fast(input_dir):
    """Fast SEC processing"""
    
    print("\n" + "=" * 60)
    print("Processing SEC EDGAR Filings (Fast Mode)")
    print("=" * 60)
    
    sec_dir = Path(input_dir) / "sec_edgar" / "sec-edgar-filings"
    txt_files = list(sec_dir.rglob("*.txt"))
    
    all_chunks = []
    
    for i, filepath in enumerate(tqdm(txt_files, desc="SEC Files")):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            if len(text) < 100:
                continue
            
            chunks = quick_chunk_text(text)
            
            for j, chunk in enumerate(chunks):
                all_chunks.append({
                    'chunk_id': f"sec_{i}_{j}",
                    'source': 'sec_edgar',
                    'source_file': filepath.name,
                    'text': chunk
                })
        except Exception as e:
            continue
    
    print(f"✓ {len(txt_files)} files → {len(all_chunks)} chunks")
    return all_chunks

def process_json_dataset(filepath, source_name):
    """Process JSON datasets"""
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    all_chunks = []
    
    for i, item in enumerate(data):
        if source_name == 'financebench':
            text = item.get('context', '') or item.get('answer', '')
        elif source_name == 'tatqa':
            # Handle TATQA paragraphs (can be list of strings or dicts)
            paragraphs = item.get('paragraphs', [])
            if isinstance(paragraphs, list):
                text_parts = []
                for p in paragraphs:
                    if isinstance(p, str):
                        text_parts.append(p)
                    elif isinstance(p, dict):
                        text_parts.append(p.get('text', ''))
                text = ' '.join(text_parts)
            else:
                text = str(paragraphs)
        else:
            text = item.get('assistant', '')
        
        if len(text) < 50:
            continue
        
        chunks = quick_chunk_text(text)
        
        for j, chunk in enumerate(chunks):
            all_chunks.append({
                'chunk_id': f"{source_name}_{i}_{j}",
                'source': source_name,
                'text': chunk
            })
    
    return all_chunks

def main():
    """Fast chunking pipeline"""
    
    print("=" * 60)
    print("Fast Document Chunking Pipeline")
    print("=" * 60)
    
    base_dir = Path("/scratch") / os.environ['USER'] / "finverify"
    input_dir = base_dir / "data" / "raw"
    output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_chunks = []
    
    # SEC
    all_chunks.extend(process_sec_fast(input_dir))
    
    # FinanceBench
    print("\n" + "=" * 60)
    print("Processing FinanceBench")
    print("=" * 60)
    fb_chunks = process_json_dataset(input_dir / "financebench" / "financebench_full.json", "financebench")
    print(f"✓ {len(fb_chunks)} chunks")
    all_chunks.extend(fb_chunks)
    
    # TATQA
    print("\n" + "=" * 60)
    print("Processing TATQA")
    print("=" * 60)
    for filename in ["tatqa_dataset_train.json", "tatqa_dataset_dev.json", "tatqa_dataset_test.json"]:
        tatqa_chunks = process_json_dataset(input_dir / "tatqa" / filename, "tatqa")
        print(f"✓ {filename}: {len(tatqa_chunks)} chunks")
        all_chunks.extend(tatqa_chunks)
    
    # Save
    output_file = output_dir / "chunks.json"
    print(f"\n" + "=" * 60)
    print(f"Saving {len(all_chunks)} chunks")
    print("=" * 60)
    
    with open(output_file, 'w') as f:
        json.dump(all_chunks, f)
    
    print(f"\n✓ Total chunks: {len(all_chunks)}")
    print(f"✓ Saved to: {output_file}")
    
    # Metadata
    metadata = {
        'total_chunks': len(all_chunks),
        'chunk_size': 2048,
        'overlap': 200
    }
    
    with open(output_dir / 'chunks_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()
