"""
Enhanced chunk cleaning for SEC EDGAR XBRL/iXBRL files
Handles both HTML and XBRL content
"""

import os
import pickle
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

def extract_text_from_xbrl(text):
    """
    Extract readable text from XBRL/iXBRL documents
    
    XBRL files contain financial data in XML tags. We need to:
    1. Remove SEC headers
    2. Extract text from XBRL tags
    3. Remove namespace declarations and metadata
    """
    
    # Step 1: Remove SEC document headers
    # These are metadata, not content
    text = re.sub(r'<SEC-DOCUMENT>.*?</SEC-DOCUMENT>', '', text, flags=re.DOTALL)
    text = re.sub(r'<SEC-HEADER>.*?</SEC-HEADER>', '', text, flags=re.DOTALL)
    text = re.sub(r'<IMS-HEADER>.*?</IMS-HEADER>', '', text, flags=re.DOTALL)
    
    # Step 2: Remove XBRL namespace declarations
    # These are just XML configuration, not content
    text = re.sub(r'xmlns:\w+="[^"]*"', '', text)
    text = re.sub(r'xsi:schemaLocation="[^"]*"', '', text)
    
    # Step 3: Extract content from <TEXT> section if present
    if '<TEXT>' in text and '</TEXT>' in text:
        match = re.search(r'<TEXT>(.*?)</TEXT>', text, re.DOTALL)
        if match:
            text = match.group(1)
    
    # Step 4: Parse with BeautifulSoup to extract text from tags
    soup = BeautifulSoup(text, 'html.parser')
    
    # Remove script, style, and metadata elements
    for element in soup(['script', 'style', 'meta', 'link', 'head']):
        element.decompose()
    
    # Get all text
    text = soup.get_text(separator=' ', strip=True)
    
    # Step 5: Clean up common XML/XBRL artifacts
    # Remove remaining XML tags that BeautifulSoup missed
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = re.sub(r'&#\d+;', ' ', text)  # Remove numeric entities
    text = re.sub(r'&#x[0-9a-fA-F]+;', ' ', text)  # Remove hex entities
    
    # Step 6: Remove XBRL-specific patterns
    # Remove dimension references
    text = re.sub(r'contextRef="[^"]*"', '', text)
    text = re.sub(r'id="[^"]*"', '', text)
    text = re.sub(r'name="[^"]*"', '', text)
    
    # Remove namespace prefixes (e.g., "us-gaap:", "dei:", "ix:")
    text = re.sub(r'\b\w+:\w+', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://[^\s]+', '', text)
    
    # Step 7: Clean whitespace
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def is_useful_chunk(text):
    """
    Check if cleaned chunk contains useful financial information
    
    Returns True if chunk has actual content, False if just metadata/fragments
    """
    # Too short
    if len(text) < 50:
        return False
    
    # Just numbers or single words
    if len(text.split()) < 5:
        return False
    
    # Mostly URLs or XML artifacts
    url_count = text.count('http')
    if url_count > 3:
        return False
    
    # Check for presence of actual words (not just numbers/symbols)
    words = [w for w in text.split() if w.isalpha() and len(w) > 2]
    if len(words) < 5:
        return False
    
    return True

def clean_html_text(text):
    """
    Clean HTML/XBRL text and extract readable content
    """
    # Quick check - if no tags, return as-is
    if not any(marker in text for marker in ['<', '&gt;', '&lt;', '&amp;', 'xmlns']):
        return text
    
    # Check if this is an XBRL document
    is_xbrl = 'xmlns' in text or 'xbrl' in text.lower() or '<ix:' in text
    
    if is_xbrl:
        # Use specialized XBRL extraction
        text = extract_text_from_xbrl(text)
    else:
        # Standard HTML cleaning
        soup = BeautifulSoup(text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "meta", "link"]):
            element.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_chunks(chunks):
    """Analyze chunks to see what types of content we have"""
    print("\n" + "=" * 70)
    print("ANALYZING CHUNKS")
    print("=" * 70)
    
    total = len(chunks)
    has_markup = 0
    has_xbrl = 0
    by_source = {}
    
    for chunk in chunks:
        source = chunk.get('source', 'unknown')
        if source not in by_source:
            by_source[source] = {
                'total': 0,
                'markup': 0,
                'xbrl': 0,
                'clean': 0
            }
        
        by_source[source]['total'] += 1
        
        text = chunk['text']
        
        # Check for any markup
        if any(marker in text for marker in ['<', '&gt;', '&lt;']):
            has_markup += 1
            by_source[source]['markup'] += 1
        
        # Check for XBRL specifically
        if any(marker in text for marker in ['xmlns', 'xbrl', '<ix:', 'contextRef']):
            has_xbrl += 1
            by_source[source]['xbrl'] += 1
        
        # Check if appears clean
        if not any(marker in text for marker in ['<', '&gt;', 'xmlns']):
            by_source[source]['clean'] += 1
    
    print(f"\nTotal chunks: {total:,}")
    print(f"Chunks with markup: {has_markup:,} ({100*has_markup/total:.1f}%)")
    print(f"Chunks with XBRL: {has_xbrl:,} ({100*has_xbrl/total:.1f}%)")
    
    print("\nBy source:")
    for source, stats in sorted(by_source.items()):
        print(f"\n  {source}:")
        print(f"    Total:  {stats['total']:6,} chunks")
        print(f"    Markup: {stats['markup']:6,} chunks ({100*stats['markup']/stats['total']:.1f}%)")
        print(f"    XBRL:   {stats['xbrl']:6,} chunks ({100*stats['xbrl']/stats['total']:.1f}%)")
        print(f"    Clean:  {stats['clean']:6,} chunks ({100*stats['clean']/stats['total']:.1f}%)")
    
    return by_source

def show_examples(chunks, num_examples=3):
    """Show before/after examples"""
    print("\n" + "=" * 70)
    print("EXAMPLES - BEFORE AND AFTER CLEANING")
    print("=" * 70)
    
    # Find chunks with markup
    markup_chunks = [c for c in chunks if any(m in c['text'] for m in ['<', '&gt;', 'xmlns'])]
    
    for i, chunk in enumerate(markup_chunks[:num_examples], 1):
        print(f"\n--- Example {i} ---")
        print(f"Source: {chunk.get('source', 'unknown')}")
        
        original = chunk['text']
        print(f"\nBEFORE (first 300 chars):")
        print(original[:300])
        
        cleaned = clean_html_text(original)
        print(f"\nAFTER (first 300 chars):")
        if cleaned:
            print(cleaned[:300])
        else:
            print("[Empty after cleaning - will be skipped]")
        
        print(f"\nLength: {len(original)} → {len(cleaned)} chars")
        print(f"Useful content: {'Yes' if is_useful_chunk(cleaned) else 'No (will be skipped)'}")

def clean_chunks(input_path, output_path, show_progress=True):
    """Clean all chunks with enhanced XBRL handling"""
    
    print("=" * 70)
    print("ENHANCED CHUNK CLEANING (XBRL-Aware)")
    print("=" * 70)
    
    # Load chunks
    print(f"\nLoading chunks from: {input_path}")
    with open(input_path, 'rb') as f:
        chunks = pickle.load(f)
    
    print(f"✓ Loaded {len(chunks):,} chunks")
    
    # Analyze
    analyze_chunks(chunks)
    
    # Show examples
    show_examples(chunks)
    
    # Confirm
    print("\n" + "=" * 70)
    print("⚠️  WARNING: Many XBRL chunks may be unusable (just metadata)")
    print("   This is normal for SEC EDGAR XBRL files.")
    print("   Only chunks with actual text content will be kept.")
    response = input("\nProceed with cleaning? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    # Clean chunks
    print("\n" + "=" * 70)
    print("CLEANING CHUNKS")
    print("=" * 70)
    
    cleaned_chunks = []
    skipped_short = 0
    skipped_useless = 0
    
    iterator = tqdm(chunks, desc="Cleaning") if show_progress else chunks
    
    for chunk in iterator:
        original_text = chunk['text']
        cleaned_text = clean_html_text(original_text)
        
        # Skip if too short after cleaning
        if len(cleaned_text.strip()) < 20:
            skipped_short += 1
            continue
        
        # Skip if not useful content
        if not is_useful_chunk(cleaned_text):
            skipped_useless += 1
            continue
        
        # Update chunk with cleaned text
        chunk['text'] = cleaned_text
        chunk['original_length'] = len(original_text)
        chunk['cleaned_length'] = len(cleaned_text)
        
        cleaned_chunks.append(chunk)
    
    print(f"\n✓ Kept {len(cleaned_chunks):,} chunks with useful content")
    print(f"⚠️  Skipped {skipped_short:,} chunks (too short)")
    print(f"⚠️  Skipped {skipped_useless:,} chunks (no useful content)")
    print(f"   Total removed: {skipped_short + skipped_useless:,} chunks")
    
    # Save
    print(f"\nSaving to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(cleaned_chunks, f)
    
    print(f"✓ Saved {len(cleaned_chunks):,} cleaned chunks")
    
    # Statistics
    print("\n" + "=" * 70)
    print("CLEANING STATISTICS")
    print("=" * 70)
    
    if cleaned_chunks:
        avg_original = sum(c['original_length'] for c in cleaned_chunks) / len(cleaned_chunks)
        avg_cleaned = sum(c['cleaned_length'] for c in cleaned_chunks) / len(cleaned_chunks)
        
        print(f"Original total: {len(chunks):,} chunks")
        print(f"Cleaned total:  {len(cleaned_chunks):,} chunks")
        print(f"Retention rate: {100*len(cleaned_chunks)/len(chunks):.1f}%")
        print(f"\nAverage length before: {avg_original:.0f} chars")
        print(f"Average length after:  {avg_cleaned:.0f} chars")
    
    return cleaned_chunks

def verify_cleaning(output_path, num_samples=5):
    """Verify cleaned chunks"""
    
    print("\n" + "=" * 70)
    print("VERIFICATION - CHECKING CLEANED CHUNKS")
    print("=" * 70)
    
    with open(output_path, 'rb') as f:
        chunks = pickle.load(f)
    
    print(f"\nLoaded {len(chunks):,} cleaned chunks")
    
    # Check for remaining markup
    has_tags = sum(1 for c in chunks if '<' in c['text'])
    has_xbrl = sum(1 for c in chunks if 'xmlns' in c['text'] or 'contextRef' in c['text'])
    
    print(f"Chunks with remaining tags: {has_tags} ({100*has_tags/len(chunks):.2f}%)")
    print(f"Chunks with XBRL artifacts: {has_xbrl} ({100*has_xbrl/len(chunks):.2f}%)")
    
    # Show random samples
    import random
    samples = random.sample(chunks, min(num_samples, len(chunks)))
    
    print(f"\n{num_samples} Random Samples:")
    for i, chunk in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Source: {chunk.get('source', 'unknown')}")
        print(f"Length: {len(chunk['text'])} chars")
        print(f"Text preview:\n{chunk['text'][:250]}")

def main():
    """Main cleaning pipeline"""
    
    # Configuration
    base_dir = Path("/scratch") / os.environ.get('USER', 'unknown') / "finverify"
    input_path = base_dir / "data" / "indexes" / "bm25" / "chunks.pkl"
    output_path = base_dir / "data" / "indexes" / "bm25" / "chunks_cleaned.pkl"
    
    # Check input exists
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        print("\nPlease check the path and try again.")
        return
    
    # Clean
    cleaned_chunks = clean_chunks(input_path, output_path)
    
    if cleaned_chunks:
        # Verify
        verify_cleaning(output_path)
        
        print("\n" + "=" * 70)
        print("✓ CLEANING COMPLETE!")
        print("=" * 70)
        print(f"\nOriginal chunks: {input_path}")
        print(f"Cleaned chunks:  {output_path}")
        print("\n⚠️  NOTE: Many SEC XBRL chunks may have been removed.")
        print("   This is expected - XBRL files contain mostly metadata.")
        print("   Your cleaned chunks now contain readable financial text.")
        print("\nNext steps:")
        print("1. Test with cleaned chunks")
        print("2. If still poor quality, consider using HTML 10-K files instead of XBRL")
        print("3. Or extract data from raw SEC filings using proper XBRL parser")

if __name__ == "__main__":
    main()
