"""
Aggressive inline XBRL text extraction
Extract ALL visible text, accepting some noise
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
import re

def aggressive_extract_text(html_content):
    """
    Aggressively extract text from inline XBRL
    Accept some noise to get more content
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Step 1: Remove non-content sections
        for tag in soup.find_all(['script', 'style', 'head']):
            tag.decompose()
        
        # Step 2: Remove XBRL resource definitions (but keep hidden sections)
        for tag in soup.find_all('ix:resources'):
            tag.decompose()
        
        # Step 3: Get all text
        text = soup.get_text(separator=' ', strip=True)
        
        # Step 4: Clean up
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove pure XBRL namespace lines
        lines = text.split('.')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that are pure metadata
            if len(line) < 20:
                continue
            if line.startswith('http://'):
                continue
            if line.startswith('xmlns'):
                continue
            if re.match(r'^[a-z\-]+:[A-Za-z]+$', line):  # us-gaap:Revenue
                continue
            if line in ['true', 'false', 'TRUE', 'FALSE']:
                continue
            
            cleaned_lines.append(line)
        
        text = '. '.join(cleaned_lines)
        
        # Remove context IDs (c-1, c-520, etc)
        text = re.sub(r'\bc-\d+\b', '', text)
        
        # Remove pure number sequences without context
        # But keep numbers that are part of sentences
        words = text.split()
        filtered_words = []
        for i, word in enumerate(words):
            # Skip standalone CIK numbers
            if re.match(r'^\d{10}$', word):
                continue
            # Skip pure duration codes
            if re.match(r'^P\d+[YMD]$', word):
                continue
            filtered_words.append(word)
        
        text = ' '.join(filtered_words)
        
        return text.strip()
        
    except Exception as e:
        print(f"Error: {e}")
        return ""

def chunk_text(text, chunk_size=2048, overlap=200):
    """Simple chunking"""
    if len(text) < 100:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if len(chunk.strip()) > 100:
            chunks.append(chunk)
        
        start = end - overlap
        if end >= len(text):
            break
    
    return chunks

def process_inline_xbrl_aggressive():
    """
    Process inline XBRL files with aggressive extraction
    """
    print("=" * 70)
    print("AGGRESSIVE INLINE XBRL TEXT EXTRACTION")
    print("=" * 70)
    
    base_dir = Path("/scratch") / os.environ['USER'] / "finverify"
    html_dir = base_dir / "data" / "raw" / "sec_edgar_html"
    
    if not html_dir.exists():
        print(f"❌ HTML directory not found: {html_dir}")
        return []
    
    html_files = list(html_dir.rglob("*.html")) + list(html_dir.rglob("*.htm"))
    print(f"Found {len(html_files)} HTML files")
    
    all_chunks = []
    total_text = 0
    
    print("\nProcessing files...")
    for i, filepath in enumerate(tqdm(html_files, desc="Extracting")):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            # Extract text aggressively
            text = aggressive_extract_text(html_content)
            
            if i == 0:
                print(f"\n--- FIRST FILE SAMPLE ---")
                print(f"File: {filepath.name}")
                print(f"Text extracted: {len(text):,} chars")
                print(f"Preview:\n{text[:500]}\n")
            
            if len(text) < 1000:
                continue
            
            total_text += len(text)
            
            # Chunk
            chunks = chunk_text(text, chunk_size=2048, overlap=200)
            
            if i == 0:
                print(f"Chunks created: {len(chunks)}")
                print(f"First chunk:\n{chunks[0][:300]}\n")
            
            for j, chunk_text in enumerate(chunks):
                all_chunks.append({
                    'chunk_id': f"sec_aggressive_{i}_{j}",
                    'source': 'sec_edgar',
                    'source_file': filepath.name,
                    'text': chunk_text
                })
        
        except Exception as e:
            print(f"Error processing {filepath.name}: {e}")
            continue
    
    print(f"\n" + "=" * 70)
    print(f"EXTRACTION COMPLETE")
    print(f"=" * 70)
    print(f"Files processed: {len(html_files)}")
    print(f"Chunks created: {len(all_chunks):,}")
    print(f"Total text: {total_text:,} chars")
    if all_chunks:
        avg_size = total_text / len(all_chunks)
        print(f"Avg chunk size: {avg_size:.0f} chars")
    
    return all_chunks

def main():
    """Process with aggressive extraction"""
    
    print("This will extract text from inline XBRL files")
    print("Accepting some noise to get more content\n")
    
    # Extract from SEC files
    sec_chunks = process_inline_xbrl_aggressive()
    
    if len(sec_chunks) < 5000:
        print(f"\n⚠️  Only {len(sec_chunks):,} chunks extracted")
        print("Inline XBRL files don't have enough text")
        print("\nRecommendation: Use FinanceBench + TATQA only, or get new data")
        return
    
    # Load FinanceBench and TATQA
    base_dir = Path("/scratch") / os.environ['USER'] / "finverify"
    
    all_chunks = list(sec_chunks)
    
    # Add FinanceBench
    fb_path = base_dir / "data" / "raw" / "financebench" / "financebench_full.json"
    if fb_path.exists():
        fb_data = json.load(open(fb_path))
        for i, item in enumerate(fb_data):
            text = item.get('context', '') or item.get('answer', '')
            if len(text) > 50:
                all_chunks.append({
                    'chunk_id': f"fb_{i}",
                    'source': 'financebench',
                    'text': text
                })
    
    # Add TATQA
    tatqa_dir = base_dir / "data" / "raw" / "tatqa"
    if tatqa_dir.exists():
        for filename in ['tatqa_dataset_train.json', 'tatqa_dataset_dev.json']:
            filepath = tatqa_dir / filename
            if filepath.exists():
                data = json.load(open(filepath))
                for i, item in enumerate(data):
                    paras = item.get('paragraphs', [])
                    if isinstance(paras, list):
                        text = ' '.join([p.get('text', '') if isinstance(p, dict) else str(p) for p in paras])
                        if len(text) > 50:
                            all_chunks.append({
                                'chunk_id': f"tatqa_{filename}_{i}",
                                'source': 'tatqa',
                                'text': text
                            })
    
    # Save
    output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "chunks.json"
    json.dump(all_chunks, open(output_file, 'w'))
    
    print(f"\n" + "=" * 70)
    print(f"FINAL RESULTS")
    print(f"=" * 70)
    print(f"Total chunks: {len(all_chunks):,}")
    print(f"  SEC: {len(sec_chunks):,}")
    print(f"  Others: {len(all_chunks) - len(sec_chunks):,}")
    print(f"\nSaved to: {output_file}")
    print(f"\nNext: Rebuild indexes and test")

if __name__ == "__main__":
    main()
