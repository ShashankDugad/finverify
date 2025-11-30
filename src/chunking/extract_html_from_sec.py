"""
Extract HTML from SEC EDGAR .txt files
The .txt files contain HTML wrapped in SEC-DOCUMENT tags
"""

import os
import re
from pathlib import Path
from tqdm import tqdm

def extract_html_from_sec_file(filepath):
    """
    Extract HTML content from SEC EDGAR .txt file
    
    SEC .txt files have structure:
    <SEC-DOCUMENT>
      <SEC-HEADER>...</SEC-HEADER>
      <DOCUMENT>
        <TYPE>10-K</TYPE>
        <SEQUENCE>1</SEQUENCE>
        <FILENAME>company-date.htm</FILENAME>
        <TEXT>
          <!DOCTYPE html>
          <html>
            ... actual HTML content ...
          </html>
        </TEXT>
      </DOCUMENT>
    </SEC-DOCUMENT>
    
    We want to extract everything between <TEXT> and </TEXT>
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Method 1: Extract from <TEXT> tags (most common)
        # Look for <TEXT> ... </TEXT> blocks
        text_match = re.search(r'<TEXT>(.*?)</TEXT>', content, re.DOTALL | re.IGNORECASE)
        
        if text_match:
            html_content = text_match.group(1).strip()
            
            # Check if it's actually HTML (starts with <!DOCTYPE or <html)
            if html_content.lower().startswith(('<!doctype', '<html')):
                return html_content, 'html'
            # If it's XML/XBRL, try next method
            elif html_content.startswith('<?xml') or 'xmlns' in html_content[:200]:
                pass  # Fall through to Method 2
            else:
                # It's some other text format, return as-is
                return html_content, 'text'
        
        # Method 2: Look for multiple DOCUMENT sections and find HTML one
        # Some filings have multiple documents (XBRL + HTML)
        documents = re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>', content, re.DOTALL | re.IGNORECASE)
        
        for doc in documents:
            # Check document type
            doc_type = re.search(r'<TYPE>(.*?)\n', doc, re.IGNORECASE)
            if doc_type and 'XML' in doc_type.group(1).upper():
                continue  # Skip XML documents
            
            # Extract TEXT from this document
            text_match = re.search(r'<TEXT>(.*?)</TEXT>', doc, re.DOTALL | re.IGNORECASE)
            if text_match:
                html_content = text_match.group(1).strip()
                if html_content.lower().startswith(('<!doctype', '<html')):
                    return html_content, 'html'
        
        # Method 3: If no <TEXT> tags, entire file might be HTML
        if content.strip().lower().startswith(('<!doctype', '<html')):
            return content, 'html'
        
        # No HTML found
        return None, None
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None

def extract_all_html_from_sec_filings(input_dir, output_dir):
    """
    Process all SEC .txt files and extract HTML
    """
    print("=" * 70)
    print("EXTRACTING HTML FROM SEC EDGAR FILES")
    print("=" * 70)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .txt files recursively
    txt_files = list(input_path.rglob("*.txt"))
    print(f"\nFound {len(txt_files)} SEC filing files (.txt)")
    
    if not txt_files:
        print("❌ No .txt files found!")
        return
    
    # Extract HTML from each
    html_extracted = 0
    xbrl_only = 0
    failed = 0
    
    print("\nExtracting HTML content...")
    for txt_file in tqdm(txt_files, desc="Processing"):
        html_content, content_type = extract_html_from_sec_file(txt_file)
        
        if html_content and content_type == 'html':
            # Create output filename
            # Keep the directory structure
            relative_path = txt_file.relative_to(input_path)
            output_file = output_path / relative_path.parent / f"{relative_path.stem}.html"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save HTML
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            html_extracted += 1
        elif content_type is None:
            xbrl_only += 1
        else:
            failed += 1
    
    # Results
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\n✓ HTML extracted:    {html_extracted:4d} files")
    print(f"⚠ XBRL only:         {xbrl_only:4d} files (no HTML found)")
    print(f"✗ Failed:            {failed:4d} files")
    print(f"\nTotal processed:     {len(txt_files):4d} files")
    print(f"\nHTML files saved to: {output_path}")
    
    return html_extracted

def verify_html_extraction(html_dir, num_samples=3):
    """
    Verify extracted HTML looks good
    """
    print("\n" + "=" * 70)
    print("VERIFYING HTML EXTRACTION")
    print("=" * 70)
    
    html_files = list(Path(html_dir).rglob("*.html"))
    
    if not html_files:
        print("❌ No HTML files found!")
        return
    
    print(f"\nFound {len(html_files)} HTML files")
    print(f"\nShowing {num_samples} samples:\n")
    
    import random
    samples = random.sample(html_files, min(num_samples, len(html_files)))
    
    for i, html_file in enumerate(samples, 1):
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"--- Sample {i}: {html_file.name} ---")
        print(f"Size: {len(content):,} bytes")
        
        # Check for actual financial content
        keywords = ['revenue', 'income', 'expense', 'asset', 'liability', 'earnings']
        found_keywords = [k for k in keywords if k.lower() in content.lower()]
        
        print(f"Financial keywords found: {', '.join(found_keywords) if found_keywords else 'None'}")
        
        # Show preview
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()
        words = text.split()[:100]
        preview = ' '.join(words)
        
        print(f"Text preview: {preview[:200]}...")
        print()

def main():
    """
    Main extraction pipeline
    """
    # Configuration
    user = os.environ.get('USER', 'unknown')
    base_dir = Path("/scratch") / user / "finverify" / "data"
    
    input_dir = base_dir / "raw" / "sec_edgar" / "sec-edgar-filings"
    output_dir = base_dir / "raw" / "sec_edgar_html"
    
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    
    if not input_dir.exists():
        print(f"\n❌ Error: Input directory not found!")
        print(f"   Expected: {input_dir}")
        print(f"\n   Please run download_sec_expanded.py first")
        return
    
    # Extract HTML
    html_count = extract_all_html_from_sec_filings(input_dir, output_dir)
    
    if html_count > 0:
        # Verify
        verify_html_extraction(output_dir)
        
        print("\n" + "=" * 70)
        print("✓ SUCCESS!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Update chunk_docs_fast.py to use HTML files:")
        print(f"   sec_dir = Path(input_dir) / 'sec_edgar_html'")
        print(f"   html_files = list(sec_dir.rglob('*.html'))")
        print("\n2. Re-run chunking with HTML files")
        print("3. Rebuild indexes")
        print("4. Test baselines - should see MUCH better results!")
    else:
        print("\n" + "=" * 70)
        print("⚠ WARNING: No HTML extracted!")
        print("=" * 70)
        print("\nThis means your SEC files are XBRL-only.")
        print("You have two options:")
        print("\n1. Use the XBRL cleaning script (clean_chunks_xbrl.py)")
        print("2. Or download HTML 10-K files directly from SEC EDGAR")

if __name__ == "__main__":
    main()
