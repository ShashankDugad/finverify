"""
Improved SEC HTML extraction with better handling of different file formats
"""

import os
import re
from pathlib import Path
from tqdm import tqdm


def extract_html_improved(filepath):
    """
    Extract HTML with multiple fallback strategies

    Strategy 1: Look for <TEXT>...</TEXT> with HTML
    Strategy 2: Look for <html>...</html> anywhere in file
    Strategy 3: Extract content between <BODY> and </BODY>
    Strategy 4: If file is small, might be pure HTML already
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Strategy 1: Standard <TEXT> extraction
        # Find all DOCUMENT sections
        documents = re.findall(
            r"<DOCUMENT>(.*?)</DOCUMENT>", content, re.DOTALL | re.IGNORECASE
        )

        for doc in documents:
            # Check document type - skip if explicitly XBRL/XML
            doc_type = re.search(r"<TYPE>(.*?)\n", doc, re.IGNORECASE)
            if doc_type:
                dtype = doc_type.group(1).strip().upper()
                # Skip known XBRL/XML types
                if any(
                    x in dtype
                    for x in ["XML", "EXCEL", "ZIP", "GRAPHIC", "EX-10", "EX-3"]
                ):
                    continue

            # Extract TEXT section
            text_match = re.search(
                r"<TEXT>(.*?)</TEXT>", doc, re.DOTALL | re.IGNORECASE
            )
            if text_match:
                text_content = text_match.group(1).strip()

                # Check if it's HTML
                if text_content.lower().startswith(("<!doctype", "<html")):
                    return text_content, "strategy1_text_tag"

                # Sometimes HTML doesn't have DOCTYPE
                if "<html" in text_content.lower()[:500]:
                    # Extract from <html> to </html>
                    html_match = re.search(
                        r"(<html.*?</html>)", text_content, re.DOTALL | re.IGNORECASE
                    )
                    if html_match:
                        return html_match.group(1), "strategy1_html_tag"

        # Strategy 2: Look for HTML anywhere in file (not in TEXT tags)
        # Sometimes HTML is not wrapped properly
        html_match = re.search(
            r"<!DOCTYPE html.*?</html>", content, re.DOTALL | re.IGNORECASE
        )
        if html_match:
            return html_match.group(0), "strategy2_doctype"

        html_match = re.search(r"<html.*?</html>", content, re.DOTALL | re.IGNORECASE)
        if html_match:
            return html_match.group(0), "strategy2_html"

        # Strategy 3: Look for substantial content between <BODY> tags
        body_match = re.search(
            r"<body.*?>(.*?)</body>", content, re.DOTALL | re.IGNORECASE
        )
        if body_match:
            body_content = body_match.group(1)
            # If body has substantial content (> 1000 chars), consider it valid
            if len(body_content.strip()) > 1000:
                # Reconstruct minimal HTML
                html_content = f"<html><body>{body_content}</body></html>"
                return html_content, "strategy3_body"

        # Strategy 4: Check if entire file is HTML (small files)
        if content.strip().lower().startswith(("<!doctype", "<html")):
            return content, "strategy4_pure_html"

        # Strategy 5: Last resort - look for any TEXT section with substantial content
        text_matches = re.findall(
            r"<TEXT>(.*?)</TEXT>", content, re.DOTALL | re.IGNORECASE
        )
        for text_content in text_matches:
            text_content = text_content.strip()
            # Skip if it's clearly XBRL
            if text_content.startswith("<?xml") or "xmlns" in text_content[:200]:
                continue
            # Skip if too short
            if len(text_content) < 1000:
                continue
            # If it has some HTML-like tags and substantial content
            if any(
                tag in text_content.lower()
                for tag in ["<p>", "<div>", "<table>", "<span>"]
            ):
                return text_content, "strategy5_html_fragments"

        return None, None

    except Exception as e:
        print(f"\n  Error reading {filepath.name}: {e}")
        return None, None


def extract_all_html_improved(input_dir, output_dir):
    """
    Process all SEC .txt files with improved extraction
    """
    print("=" * 70)
    print("IMPROVED SEC HTML EXTRACTION")
    print("=" * 70)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all .txt files
    txt_files = list(input_path.rglob("*.txt"))
    print(f"\nFound {len(txt_files)} SEC filing files (.txt)")

    if not txt_files:
        print("❌ No .txt files found!")
        return

    # Track results by strategy
    results = {
        "strategy1_text_tag": 0,
        "strategy1_html_tag": 0,
        "strategy2_doctype": 0,
        "strategy2_html": 0,
        "strategy3_body": 0,
        "strategy4_pure_html": 0,
        "strategy5_html_fragments": 0,
        "failed": 0,
    }

    failed_files = []

    print("\nExtracting HTML content...")
    for txt_file in tqdm(txt_files, desc="Processing"):
        html_content, strategy = extract_html_improved(txt_file)

        if html_content and strategy:
            # Create output filename
            relative_path = txt_file.relative_to(input_path)
            output_file = (
                output_path / relative_path.parent / f"{relative_path.stem}.html"
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save HTML
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(html_content)
                results[strategy] += 1
            except Exception as e:
                results["failed"] += 1
                failed_files.append((txt_file.name, str(e)))
        else:
            results["failed"] += 1
            failed_files.append((txt_file.name, "No HTML found"))

    # Print results
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    total_success = sum(v for k, v in results.items() if k != "failed")

    print(
        f"\n✓ Successfully extracted: {total_success:3d} files ({100*total_success/len(txt_files):.1f}%)"
    )
    print(
        f"✗ Failed:                 {results['failed']:3d} files ({100*results['failed']/len(txt_files):.1f}%)"
    )

    print("\nExtraction strategies used:")
    print(f"  Strategy 1 (TEXT tag):       {results['strategy1_text_tag']:3d} files")
    print(f"  Strategy 1 (HTML in TEXT):   {results['strategy1_html_tag']:3d} files")
    print(f"  Strategy 2 (DOCTYPE found):  {results['strategy2_doctype']:3d} files")
    print(f"  Strategy 2 (HTML tag found): {results['strategy2_html']:3d} files")
    print(f"  Strategy 3 (BODY content):   {results['strategy3_body']:3d} files")
    print(f"  Strategy 4 (Pure HTML):      {results['strategy4_pure_html']:3d} files")
    print(
        f"  Strategy 5 (HTML fragments): {results['strategy5_html_fragments']:3d} files"
    )

    if failed_files and len(failed_files) <= 10:
        print(f"\nFailed files:")
        for fname, reason in failed_files[:10]:
            print(f"  - {fname}: {reason}")
    elif failed_files:
        print(f"\nFirst 5 failed files:")
        for fname, reason in failed_files[:5]:
            print(f"  - {fname}: {reason}")
        print(f"  ... and {len(failed_files)-5} more")

    print(f"\nHTML files saved to: {output_path}")

    return total_success


def analyze_failed_file(filepath):
    """
    Analyze a failed file to understand its structure
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING: {filepath.name}")
    print("=" * 70)

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        print(f"File size: {len(content):,} bytes")

        # Check for various markers
        markers = {
            "<SEC-DOCUMENT>": content.count("<SEC-DOCUMENT>"),
            "<DOCUMENT>": content.count("<DOCUMENT>"),
            "<TEXT>": content.count("<TEXT>"),
            "<html": content.lower().count("<html"),
            "<!DOCTYPE": content.count("<!DOCTYPE"),
            "<?xml": content.count("<?xml"),
            "xmlns": content.count("xmlns"),
            "<body>": content.lower().count("<body"),
        }

        print("\nMarkers found:")
        for marker, count in markers.items():
            print(f"  {marker:20s}: {count}")

        # Show first 500 chars
        print(f"\nFirst 500 characters:")
        print(content[:500])

        # Check document types
        doc_types = re.findall(r"<TYPE>(.*?)</TYPE>", content, re.IGNORECASE)
        if doc_types:
            print(f"\nDocument types found:")
            for dtype in set(doc_types):
                print(f"  - {dtype.strip()}")

    except Exception as e:
        print(f"Error analyzing file: {e}")


def main():
    """Main pipeline with diagnostics"""

    user = os.environ.get("USER", "unknown")
    base_dir = Path("/scratch") / user / "finverify" / "data"

    input_dir = base_dir / "raw" / "sec_edgar" / "sec-edgar-filings"
    output_dir = base_dir / "raw" / "sec_edgar_html"

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    if not input_dir.exists():
        print(f"\n❌ Input directory not found: {input_dir}")
        return

    # Extract HTML
    success_count = extract_all_html_improved(input_dir, output_dir)

    if success_count == 0:
        print("\n" + "=" * 70)
        print("❌ NO FILES EXTRACTED!")
        print("=" * 70)
        print("\nLet's analyze a sample file to understand the structure:")

        txt_files = list(input_dir.rglob("*.txt"))
        if txt_files:
            analyze_failed_file(txt_files[0])

            if len(txt_files) > 1:
                print("\n\nAnalyzing another file:")
                analyze_failed_file(txt_files[1])

    elif success_count < len(list(input_dir.rglob("*.txt"))) * 0.5:
        print("\n" + "=" * 70)
        print("⚠️  LOW SUCCESS RATE")
        print("=" * 70)
        print("\nLet's analyze a failed file:")

        # Find a failed file
        txt_files = list(input_dir.rglob("*.txt"))
        html_files = list(output_dir.rglob("*.html"))
        html_stems = {f.stem for f in html_files}

        failed = [f for f in txt_files if f.stem not in html_stems]
        if failed:
            analyze_failed_file(failed[0])

    else:
        print("\n" + "=" * 70)
        print("✓ SUCCESS!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Run: python3 chunk_docs_html.py")
        print("2. Rebuild indexes")
        print("3. Test baselines")


if __name__ == "__main__":
    main()
