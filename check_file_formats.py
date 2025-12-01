"""
Check which downloaded SEC files have usable HTML vs inline XBRL
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def analyze_sec_file_format(filepath):
    """
    Analyze a single SEC file to determine format
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            # Read first 50KB to check format
            content = f.read(50000)
        
        analysis = {
            'filename': filepath.name,
            'size': filepath.stat().st_size,
            'has_html': False,
            'has_inline_xbrl': False,
            'has_separate_html': False,
            'format': 'unknown',
            'usable': False
        }
        
        # Check for inline XBRL markers
        inline_xbrl_markers = [
            'xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"',
            '<ix:nonNumeric',
            '<ix:nonFraction',
            '<ix:hidden'
        ]
        
        if any(marker in content for marker in inline_xbrl_markers):
            analysis['has_inline_xbrl'] = True
            analysis['format'] = 'inline_xbrl'
            analysis['usable'] = False  # Inline XBRL is hard to parse
        
        # Check for traditional HTML
        if '<html' in content.lower():
            analysis['has_html'] = True
            
            # Check if it's clean HTML (not inline XBRL)
            if not analysis['has_inline_xbrl']:
                # Look for actual content paragraphs
                if '<p>' in content.lower() or '<div>' in content.lower():
                    analysis['has_separate_html'] = True
                    analysis['format'] = 'traditional_html'
                    analysis['usable'] = True  # Traditional HTML is perfect!
        
        return analysis
        
    except Exception as e:
        return {
            'filename': filepath.name,
            'format': 'error',
            'usable': False,
            'error': str(e)
        }

def check_all_downloaded_files(input_dir):
    """
    Check all downloaded files and categorize them
    """
    print("=" * 70)
    print("CHECKING SEC FILE FORMATS")
    print("=" * 70)
    
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"\n❌ Directory not found: {input_path}")
        return
    
    # Find all .txt files
    txt_files = list(input_path.rglob("*.txt"))
    
    if not txt_files:
        print(f"\n❌ No .txt files found in {input_path}")
        return
    
    print(f"\nFound {len(txt_files)} SEC filing files")
    print("Analyzing formats...\n")
    
    # Analyze all files
    results = defaultdict(list)
    
    for filepath in txt_files:
        analysis = analyze_sec_file_format(filepath)
        results[analysis['format']].append(analysis)
    
    # Print summary
    print("=" * 70)
    print("FORMAT ANALYSIS RESULTS")
    print("=" * 70)
    
    print(f"\nTotal files analyzed: {len(txt_files)}")
    print(f"\nBy format:")
    
    format_names = {
        'traditional_html': '✓ Traditional HTML (USABLE!)',
        'inline_xbrl': '✗ Inline XBRL (Not usable)',
        'unknown': '? Unknown format',
        'error': '✗ Error reading'
    }
    
    for format_type in ['traditional_html', 'inline_xbrl', 'unknown', 'error']:
        count = len(results[format_type])
        if count > 0:
            pct = 100 * count / len(txt_files)
            print(f"  {format_names[format_type]:35s}: {count:3d} files ({pct:5.1f}%)")
    
    # Show usable files
    usable_files = [r for r in results['traditional_html']]
    
    if usable_files:
        print(f"\n" + "=" * 70)
        print(f"✓ GOOD NEWS: {len(usable_files)} USABLE FILES FOUND!")
        print("=" * 70)
        
        # Group by company
        by_company = defaultdict(list)
        for analysis in usable_files:
            # Extract company ticker from path
            parts = str(analysis['filename']).split('/')
            if len(parts) > 0:
                company = parts[0] if '/' in str(analysis['filename']) else 'unknown'
                by_company[company].append(analysis)
        
        print(f"\nUsable files by company:")
        for company, files in sorted(by_company.items()):
            print(f"  {company:8s}: {len(files)} files")
        
        print(f"\nSample usable files:")
        for analysis in usable_files[:5]:
            size_mb = analysis['size'] / 1024 / 1024
            print(f"  - {analysis['filename']} ({size_mb:.1f} MB)")
    
    else:
        print(f"\n" + "=" * 70)
        print(f"⚠️  NO USABLE TRADITIONAL HTML FILES FOUND")
        print("=" * 70)
        print("\nAll downloaded files are inline XBRL format.")
        print("\nOptions:")
        print("1. Use the inline XBRL files anyway (with improved extraction)")
        print("2. Try downloading older companies that might still use traditional HTML")
        print("3. Use SEC EDGAR API to download rendered HTML versions")
    
    # Check inline XBRL files
    if results['inline_xbrl']:
        print(f"\n" + "=" * 70)
        print("INLINE XBRL FILES (Harder to Parse)")
        print("=" * 70)
        print(f"\nFound {len(results['inline_xbrl'])} inline XBRL files")
        print("\nThese can still be used but require more complex extraction.")
        print("Sample files:")
        for analysis in results['inline_xbrl'][:5]:
            size_mb = analysis['size'] / 1024 / 1024
            print(f"  - {analysis['filename']} ({size_mb:.1f} MB)")
    
    return results

def main():
    """
    Check downloaded SEC files
    """
    user = os.environ.get('USER', 'unknown')
    
    # Check multiple possible locations
    possible_dirs = [
        Path("/scratch") / user / "finverify" / "data" / "raw" / "sec_edgar_simple" / "sec-edgar-filings",
        Path("/scratch") / user / "finverify" / "data" / "raw" / "sec_edgar_2018" / "sec-edgar-filings",
        Path("/scratch") / user / "finverify" / "data" / "raw" / "sec_edgar" / "sec-edgar-filings",
    ]
    
    print("Searching for SEC filings in:")
    for dir_path in possible_dirs:
        print(f"  - {dir_path}")
        if dir_path.exists():
            print(f"    ✓ Found!")
            results = check_all_downloaded_files(dir_path)
            return results
        else:
            print(f"    ✗ Not found")
    
    print("\n❌ No SEC filing directories found!")
    print("Please run download script first.")

if __name__ == "__main__":
    main()
