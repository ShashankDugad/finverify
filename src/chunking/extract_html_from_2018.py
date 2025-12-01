"""
Extract HTML from 2018-2019 SEC filings
These older filings have better HTML separation from XBRL
"""

import os
import re
from pathlib import Path
from tqdm import tqdm

def extract_html_from_2018_filing(filepath):
    """
    Extract HTML from 2018-2019 SEC filings
    
    These older filings typically have:
    - Primary document: HTML 10-K (what we want)
    - Separate XBRL files (we'll skip these)
    - Better separation between HTML and XBRL
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Find all DOCUMENT sections
        documents = re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>', content, re.DOTALL | re.IGNORECASE)
        
        for doc in documents:
            # Get document type
            doc_type = re.search(r'<TYPE>(.*?)\n', doc, re.IGNORECASE)
            if doc_type:
                dtype = doc_type.group(1).strip().upper()
                
                # Skip XBRL and other non-narrative documents
                skip_types = ['XML', 'EXCEL', 'ZIP', 'GRAPHIC', 'EX-101', 'EX-10', 
                             'EX-21', 'EX-23', 'EX-31', 'EX-32', 'EX-99']
                if any(x in dtype for x in skip_types):
                    continue
                
                # Look for 10-K document
                if '10-K' in dtype or dtype in ['10-K', '10-K/A']:
                    # Extract TEXT section
                    text_match = re.search(r'<TEXT>(.*?)</TEXT>', doc, re.DOTALL | re.IGNORECASE)
                    if text_match:
                        text_content = text_match.group(1).strip()
                        
                        # Check if it's HTML
                        if text_content.lower().startswith(('<!doctype', '<html')):
                            return text_content, 'primary_10k_html'
                        
                        # Sometimes HTML doesn't have DOCTYPE
                        if '<html' in text_content.lower()[:500]:
                            html_match = re.search(r'(<html.*?</html>)', text_content, re.DOTALL | re.IGNORECASE)
                            if html_match:
                                return html_match.group(1), 'primary_10k_html_notag'
        
        # Fallback: Try the improved extraction strategies
        # Strategy 1: Look for any HTML document
        for doc in documents:
            text_match = re.search(r'<TEXT>(.*?)</TEXT>', doc, re.DOTALL | re.IGNORECASE)
            if text_match:
                text_content = text_match.group(1).strip()
                
                # Skip if it's XBRL
                if text_content.startswith('<?xml') or 'xmlns' in text_content[:200]:
                    continue
                
                # If it has HTML markers and is substantial
                if text_content.lower().startswith(('<!doctype', '<html')) and len(text_content) > 10000:
                    return text_content, 'fallback_html'
        
        # Strategy 2: Look for substantial HTML anywhere
        html_match = re.search(r'<!DOCTYPE html.*?</html>', content, re.DOTALL | re.IGNORECASE)
        if html_match and len(html_match.group(0)) > 10000:
            return html_match.group(0), 'extracted_html'
        
        return None, None
        
    except Exception as e:
        print(f"\n  Error reading {filepath.name}: {e}")
        return None, None

def extract_all_html_from_2018(input_dir, output_dir):
    """
    Extract HTML from all 2018-2019 SEC filings
    """
    print("=" * 70)
    print("EXTRACTING HTML FROM 2018-2019 SEC FILINGS")
    print("=" * 70)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .txt files
    txt_files = list(input_path.rglob("*.txt"))
    print(f"\nFound {len(txt_files)} SEC filing files (.txt)")
    
    if not txt_files:
        print("❌ No .txt files found!")
        print(f"   Checked: {input_path}")
        return 0
    
    # Track results
    results = {
        'primary_10k_html': 0,
        'primary_10k_html_notag': 0,
        'fallback_html': 0,
        'extracted_html': 0,
        'failed': 0
    }
    
    failed_files = []
    sample_previews = []
    
    print("\nExtracting HTML content...")
    for i, txt_file in enumerate(tqdm(txt_files, desc="Processing")):
        html_content, strategy = extract_html_from_2018_filing(txt_file)
        
        if html_content and strategy:
            # Save sample preview from first 3 files
            if len(sample_previews) < 3:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                text = soup.get_text()[:300]
                sample_previews.append({
                    'file': txt_file.name,
                    'strategy': strategy,
                    'size': len(html_content),
                    'preview': text
                })
            
            # Create output filename
            relative_path = txt_file.relative_to(input_path)
            output_file = output_path / relative_path.parent / f"{relative_path.stem}.html"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save HTML
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                results[strategy] += 1
            except Exception as e:
                results['failed'] += 1
                failed_files.append((txt_file.name, str(e)))
        else:
            results['failed'] += 1
            failed_files.append((txt_file.name, 'No HTML found'))
    
    # Print results
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    
    total_success = sum(v for k, v in results.items() if k != 'failed')
    
    print(f"\n✓ Successfully extracted: {total_success:3d} files ({100*total_success/len(txt_files):.1f}%)")
    print(f"✗ Failed:                 {results['failed']:3d} files ({100*results['failed']/len(txt_files):.1f}%)")
    
    print("\nExtraction strategies used:")
    print(f"  Primary 10-K HTML:       {results['primary_10k_html']:3d} files")
    print(f"  Primary 10-K (no tag):   {results['primary_10k_html_notag']:3d} files")
    print(f"  Fallback HTML:           {results['fallback_html']:3d} files")
    print(f"  Extracted HTML:          {results['extracted_html']:3d} files")
    
    # Show sample previews
    if sample_previews:
        print("\n" + "=" * 70)
        print("SAMPLE EXTRACTED HTML PREVIEWS")
        print("=" * 70)
        for i, sample in enumerate(sample_previews, 1):
            print(f"\n--- Sample {i}: {sample['file']} ---")
            print(f"Strategy: {sample['strategy']}")
            print(f"Size: {sample['size']:,} bytes")
            print(f"Text preview:")
            print(sample['preview'])
    
    if failed_files and len(failed_files) <= 5:
        print(f"\nFailed files:")
        for fname, reason in failed_files:
            print(f"  - {fname}: {reason}")
    elif failed_files:
        print(f"\nFirst 5 failed files:")
        for fname, reason in failed_files[:5]:
            print(f"  - {fname}: {reason}")
    
    print(f"\nHTML files saved to: {output_path}")
    
    return total_success

def main():
    """Main extraction pipeline"""
    
    user = os.environ.get('USER', 'unknown')
    base_dir = Path("/scratch") / user / "finverify" / "data"
    
    input_dir = base_dir / "raw" / "sec_edgar_2018" / "sec-edgar-filings"
    output_dir = base_dir / "raw" / "sec_edgar_2018_html"
    
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    
    if not input_dir.exists():
        print(f"\n❌ Input directory not found: {input_dir}")
        print("\nPlease run download_sec_2018.py first!")
        return
    
    # Extract HTML
    success_count = extract_all_html_from_2018(input_dir, output_dir)
    
    if success_count > 0:
        print("\n" + "=" * 70)
        print("✓ SUCCESS!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Update chunking to use 2018 HTML:")
        print("   python3 chunk_docs_from_2018.py")
        print("\n2. Build indexes:")
        print("   python3 build_bm25.py")
        print("   python3 generate_embeddings.py")
        print("   python3 build_faiss.py")
        print("\n3. Test baselines:")
        print("   python3 bm25_t5.py --test-mode")
    else:
        print("\n" + "=" * 70)
        print("⚠ WARNING: No HTML extracted!")
        print("=" * 70)
        print("\nPossible issues:")
        print("1. Files might still be inline XBRL")
        print("2. Download might have failed")
        print("3. Check if .txt files exist in input directory")

if __name__ == "__main__":
    main()
