"""
Chunk documents from 2018-2019 SEC HTML files
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
import re

def clean_html_text(html_content):
    """Clean HTML and extract readable text"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove non-content elements
        for element in soup(['script', 'style', 'meta', 'link', 'head', 'noscript']):
            element.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    except Exception as e:
        return ""

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

def process_2018_html(input_dir):
    """Process 2018-2019 SEC HTML files"""
    
    print("\n" + "=" * 70)
    print("PROCESSING 2018-2019 SEC HTML FILES")
    print("=" * 70)
    
    html_dir = Path(input_dir) / "sec_edgar_2018_html"
    
    if not html_dir.exists():
        print(f"❌ HTML directory not found: {html_dir}")
        print("\nPlease run extract_html_from_2018.py first!")
        return []
    
    html_files = list(html_dir.rglob("*.html")) + list(html_dir.rglob("*.htm"))
    
    if not html_files:
        print(f"❌ No HTML files found in {html_dir}")
        return []
    
    print(f"✓ Found {len(html_files)} HTML files")
    
    all_chunks = []
    processed = 0
    skipped = 0
    total_text = 0
    
    print("\nProcessing HTML files...")
    
    for i, filepath in enumerate(tqdm(html_files, desc="Processing")):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            # Show details for first file
            if i == 0:
                print(f"\n\n--- FIRST FILE DETAILS ---")
                print(f"File: {filepath.name}")
                print(f"HTML size: {len(html_content):,} bytes")
            
            # Clean HTML
            text = clean_html_text(html_content)
            
            if i == 0:
                print(f"Text size: {len(text):,} bytes")
                print(f"Preview (first 300 chars):")
                print(text[:300])
                print("---\n")
            
            # Skip if too short
            if len(text) < 100:
                skipped += 1
                continue
            
            total_text += len(text)
            
            # Chunk
            chunks = quick_chunk_text(text, chunk_size=2048, overlap=200)
            
            if i == 0:
                print(f"Chunks created: {len(chunks)}")
                print(f"First chunk preview:")
                print(chunks[0][:200])
                print("---\n")
            
            for j, chunk in enumerate(chunks):
                all_chunks.append({
                    'chunk_id': f"sec2018_{i}_{j}",
                    'source': 'sec_edgar',
                    'source_file': filepath.name,
                    'text': chunk
                })
            
            processed += 1
            
        except Exception as e:
            skipped += 1
            if i < 5:
                print(f"\n⚠️  Error: {filepath.name}: {e}")
            continue
    
    print(f"\n" + "=" * 70)
    print("SEC 2018-2019 PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Files found:       {len(html_files)}")
    print(f"Files processed:   {processed}")
    print(f"Files skipped:     {skipped}")
    print(f"Chunks created:    {len(all_chunks):,}")
    print(f"Text extracted:    {total_text:,} bytes")
    
    if all_chunks:
        avg_chunk_size = sum(len(c['text']) for c in all_chunks) / len(all_chunks)
        print(f"Avg chunk size:    {avg_chunk_size:.0f} chars")
    
    return all_chunks

def process_json_dataset(filepath, source_name):
    """Process JSON datasets"""
    
    if not filepath.exists():
        return []
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        all_chunks = []
        
        for i, item in enumerate(data):
            if source_name == 'financebench':
                text = item.get('context', '') or item.get('answer', '')
            elif source_name == 'tatqa':
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
            
            chunks = quick_chunk_text(text, chunk_size=2048, overlap=200)
            
            for j, chunk in enumerate(chunks):
                all_chunks.append({
                    'chunk_id': f"{source_name}_{i}_{j}",
                    'source': source_name,
                    'text': chunk
                })
        
        return all_chunks
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []

def main():
    """Main chunking pipeline for 2018 data"""
    
    print("=" * 70)
    print("CHUNKING 2018-2019 SEC DATA")
    print("=" * 70)
    
    base_dir = Path("/scratch") / os.environ['USER'] / "finverify"
    input_dir = base_dir / "data" / "raw"
    output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_chunks = []
    chunk_stats = {}
    
    # Process 2018 SEC HTML
    print("\n📄 STEP 1: Processing 2018-2019 SEC HTML...")
    sec_chunks = process_2018_html(input_dir)
    if sec_chunks:
        all_chunks.extend(sec_chunks)
        chunk_stats['sec_edgar'] = len(sec_chunks)
        print(f"✓ SEC 2018 chunks: {len(sec_chunks):,}")
    else:
        chunk_stats['sec_edgar'] = 0
        print("⚠️  No SEC 2018 chunks")
    
    # FinanceBench
    print("\n📊 STEP 2: Processing FinanceBench...")
    fb_paths = [
        input_dir / "financebench" / "financebench_full.json",
        input_dir / "financebench" / "financebench.json",
    ]
    fb_chunks = []
    for fb_path in fb_paths:
        if fb_path.exists():
            fb_chunks = process_json_dataset(fb_path, "financebench")
            break
    
    if fb_chunks:
        all_chunks.extend(fb_chunks)
        chunk_stats['financebench'] = len(fb_chunks)
        print(f"✓ FinanceBench: {len(fb_chunks):,} chunks")
    else:
        chunk_stats['financebench'] = 0
        print("⚠️  No FinanceBench data")
    
    # TATQA
    print("\n📈 STEP 3: Processing TATQA...")
    tatqa_dir = input_dir / "tatqa"
    tatqa_chunks = []
    if tatqa_dir.exists():
        for filename in ["tatqa_dataset_train.json", "tatqa_dataset_dev.json", "tatqa_dataset_test.json"]:
            filepath = tatqa_dir / filename
            if filepath.exists():
                chunks = process_json_dataset(filepath, "tatqa")
                tatqa_chunks.extend(chunks)
                print(f"✓ {filename}: {len(chunks):,} chunks")
    
    if tatqa_chunks:
        all_chunks.extend(tatqa_chunks)
        chunk_stats['tatqa'] = len(tatqa_chunks)
    else:
        chunk_stats['tatqa'] = 0
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nChunks by source:")
    for source, count in chunk_stats.items():
        pct = 100 * count / len(all_chunks) if all_chunks else 0
        print(f"  {source:20s}: {count:8,} chunks ({pct:5.1f}%)")
    print(f"\n  {'TOTAL':20s}: {len(all_chunks):8,} chunks")
    
    # Save
    if all_chunks:
        output_file = output_dir / "chunks.json"
        print(f"\n💾 Saving to: {output_file}")
        
        with open(output_file, 'w') as f:
            json.dump(all_chunks, f)
        
        print(f"✓ Saved {len(all_chunks):,} chunks")
        
        # Metadata
        metadata = {
            'total_chunks': len(all_chunks),
            'by_source': chunk_stats,
            'chunk_size': 2048,
            'overlap': 200,
            'html_cleaned': True,
            'sec_year': '2018-2019'
        }
        
        with open(output_dir / 'chunks_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "=" * 70)
        print("✓ CHUNKING COMPLETE!")
        print("=" * 70)
        print(f"\nYou now have {len(all_chunks):,} high-quality chunks!")
        print("\nNext steps:")
        print("1. python3 build_bm25.py")
        print("2. python3 generate_embeddings.py")
        print("3. python3 build_faiss.py")
        print("4. python3 bm25_t5.py --test-mode")
    else:
        print("\n❌ No chunks created!")
        print("Check that HTML extraction succeeded.")

if __name__ == "__main__":
    main()
