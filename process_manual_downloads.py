"""
Process manually downloaded SEC 10-K HTML files
Clean, chunk, and combine with existing data
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

def chunk_text(text, chunk_size=2048, overlap=200):
    """Chunk text into segments"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        
        start = end - overlap
        if end >= len(text):
            break
    
    return chunks

def process_manual_html_files():
    """Process manually downloaded HTML files"""
    
    print("=" * 70)
    print("PROCESSING MANUALLY DOWNLOADED SEC 10-K FILES")
    print("=" * 70)
    
    base_dir = Path("/scratch") / os.environ['USER'] / "finverify"
    html_dir = base_dir / "data" / "raw" / "sec_manual"
    
    if not html_dir.exists():
        print(f"\n❌ Directory not found: {html_dir}")
        print("\nPlease run download_targeted_companies.py first!")
        return []
    
    html_files = list(html_dir.glob("*.html")) + list(html_dir.glob("*.htm"))
    
    if not html_files:
        print(f"\n❌ No HTML files found in {html_dir}")
        return []
    
    print(f"\n✓ Found {len(html_files)} HTML files")
    print("\nProcessing files...")
    
    all_chunks = []
    processed = 0
    skipped = 0
    total_text = 0
    
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
            if len(text) < 1000:
                skipped += 1
                if i < 3:
                    print(f"⚠️  Skipped {filepath.name}: text too short ({len(text)} chars)")
                continue
            
            total_text += len(text)
            
            # Chunk
            text_chunks = chunk_text(text, chunk_size=2048, overlap=200)
            
            if i == 0:
                print(f"Chunks created: {len(text_chunks)}")
                print(f"First chunk preview:")
                print(text_chunks[0][:200])
                print("---\n")
            
            for j, chunk_content in enumerate(text_chunks):
                all_chunks.append({
                    'chunk_id': f"sec_manual_{i}_{j}",
                    'source': 'sec_edgar',
                    'source_file': filepath.name,
                    'text': chunk_content
                })
            
            processed += 1
            
        except Exception as e:
            skipped += 1
            if i < 5:
                print(f"\n⚠️  Error: {filepath.name}: {e}")
            continue
    
    print(f"\n" + "=" * 70)
    print("SEC PROCESSING COMPLETE")
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

def load_existing_chunks():
    """Load existing FinanceBench + TATQA chunks"""
    
    base_dir = Path("/scratch") / os.environ['USER'] / "finverify"
    chunks_file = base_dir / "data" / "processed" / "chunks.json"
    
    if not chunks_file.exists():
        print("No existing chunks found")
        return []
    
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    # Keep only FinanceBench and TATQA
    existing = [c for c in chunks if c['source'] in ['financebench', 'tatqa']]
    
    print(f"\nExisting chunks loaded:")
    sources = {}
    for c in existing:
        sources[c['source']] = sources.get(c['source'], 0) + 1
    
    for source, count in sources.items():
        print(f"  {source}: {count:,}")
    
    return existing

def main():
    """Main processing pipeline"""
    
    print("=" * 70)
    print("COMBINING MANUAL SEC + EXISTING DATA")
    print("=" * 70)
    
    # Process manual SEC files
    print("\n📄 STEP 1: Processing manually downloaded SEC files...")
    sec_chunks = process_manual_html_files()
    
    if not sec_chunks:
        print("\n❌ No SEC chunks created!")
        print("Please check that HTML files were downloaded successfully.")
        return
    
    # Load existing data
    print("\n📊 STEP 2: Loading existing FinanceBench + TATQA data...")
    existing_chunks = load_existing_chunks()
    
    # Combine
    all_chunks = list(sec_chunks) + list(existing_chunks)
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    chunk_stats = {}
    for c in all_chunks:
        source = c['source']
        chunk_stats[source] = chunk_stats.get(source, 0) + 1
    
    print(f"\nChunks by source:")
    for source, count in sorted(chunk_stats.items()):
        pct = 100 * count / len(all_chunks) if all_chunks else 0
        print(f"  {source:20s}: {count:8,} chunks ({pct:5.1f}%)")
    print(f"\n  {'TOTAL':20s}: {len(all_chunks):8,} chunks")
    
    # Save
    base_dir = Path("/scratch") / os.environ['USER'] / "finverify"
    output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        'sec_source': 'manual_download_2018',
        'html_cleaned': True
    }
    
    with open(output_dir / 'chunks_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✓ PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nYou now have {len(all_chunks):,} high-quality chunks!")
    
    # Estimate expected improvement
    sec_pct = 100 * chunk_stats.get('sec_edgar', 0) / len(all_chunks)
    
    print("\n📈 Expected Performance Improvement:")
    print(f"  Current: 4% EM, 10% F1 (with 3.5K chunks)")
    print(f"  After:   15-25% EM, 30-40% F1 (with {len(all_chunks):,} chunks)")
    print(f"  SEC coverage: {sec_pct:.1f}% of data")
    
    print("\n🚀 Next steps:")
    print("1. python3 build_bm25.py")
    print("2. python3 generate_embeddings.py")
    print("3. python3 build_faiss.py")
    print("4. python3 bm25_t5.py --eval --num-questions 50")
    
    print("\nExpected new results:")
    print("  EM: 15-25% (was 4%)")
    print("  F1: 30-40% (was 10%)")
    print("  Retrieval will show actual Apple, Microsoft, etc. text!")

if __name__ == "__main__":
    main()
