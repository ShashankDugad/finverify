"""
Clean HTML/XML from existing chunks
Fixes SEC EDGAR HTML markup issue
"""

import os
import pickle
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm


def clean_html_text(text):
    """
    Remove HTML/XML markup and clean text

    Handles:
    - HTML tags (<div>, <span>, etc.)
    - HTML entities (&gt;, &lt;, etc.)
    - Excessive whitespace
    - SEC EDGAR headers
    """
    # Quick check - if no HTML, return as-is
    if not any(marker in text for marker in ["<", "&gt;", "&lt;", "&amp;"]):
        return text

    # Remove SEC EDGAR headers
    if "<SEC-DOCUMENT>" in text:
        text = re.sub(r"<SEC-DOCUMENT>.*?</SEC-DOCUMENT>", "", text, flags=re.DOTALL)
    if "<SEC-HEADER>" in text:
        text = re.sub(r"<SEC-HEADER>.*?</SEC-HEADER>", "", text, flags=re.DOTALL)

    # Extract text from <TEXT> tags if present
    if "<TEXT>" in text:
        match = re.search(r"<TEXT>(.*?)</TEXT>", text, re.DOTALL)
        if match:
            text = match.group(1)

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "meta", "link"]):
        element.decompose()

    # Get text
    text = soup.get_text(separator=" ", strip=True)

    # Clean up HTML entities that BeautifulSoup might have missed
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#8217;", "'")
    text = text.replace("&#8220;", '"')
    text = text.replace("&#8221;", '"')
    text = text.replace("&#x2013;", "-")
    text = text.replace("&#x2014;", "-")

    # Clean up whitespace
    # Split into lines and clean each
    lines = (line.strip() for line in text.splitlines())
    # Break multi-spaces into single spaces
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Join and collapse multiple spaces
    text = " ".join(chunk for chunk in chunks if chunk)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def analyze_chunks(chunks):
    """Analyze chunks to see HTML prevalence"""
    print("\n" + "=" * 70)
    print("ANALYZING CHUNKS")
    print("=" * 70)

    total = len(chunks)
    has_html = 0
    by_source = {}

    for chunk in chunks:
        source = chunk.get("source", "unknown")
        if source not in by_source:
            by_source[source] = {"total": 0, "html": 0}

        by_source[source]["total"] += 1

        text = chunk["text"]
        if any(marker in text for marker in ["<", "&gt;", "&lt;"]):
            has_html += 1
            by_source[source]["html"] += 1

    print(f"\nTotal chunks: {total:,}")
    print(f"Chunks with HTML: {has_html:,} ({100*has_html/total:.1f}%)")

    print("\nBy source:")
    for source, stats in sorted(by_source.items()):
        pct = 100 * stats["html"] / stats["total"] if stats["total"] > 0 else 0
        print(
            f"  {source:20s}: {stats['total']:6,} chunks, {stats['html']:6,} with HTML ({pct:.1f}%)"
        )

    return by_source


def show_examples(chunks, num_examples=3):
    """Show before/after examples"""
    print("\n" + "=" * 70)
    print("EXAMPLES - BEFORE AND AFTER CLEANING")
    print("=" * 70)

    # Find chunks with HTML
    html_chunks = [c for c in chunks if any(m in c["text"] for m in ["<", "&gt;"])]

    for i, chunk in enumerate(html_chunks[:num_examples], 1):
        print(f"\n--- Example {i} ---")
        print(f"Source: {chunk.get('source', 'unknown')}")
        print(f"\nBEFORE (first 300 chars):")
        print(chunk["text"][:300])

        cleaned = clean_html_text(chunk["text"])
        print(f"\nAFTER (first 300 chars):")
        print(cleaned[:300])
        print(f"\nLength: {len(chunk['text'])} → {len(cleaned)} chars")


def clean_chunks(input_path, output_path, show_progress=True):
    """Clean all chunks"""

    print("=" * 70)
    print("CHUNK CLEANING SCRIPT")
    print("=" * 70)

    # Load chunks
    print(f"\nLoading chunks from: {input_path}")
    with open(input_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"✓ Loaded {len(chunks):,} chunks")

    # Analyze
    analyze_chunks(chunks)

    # Show examples
    show_examples(chunks)

    # Confirm
    print("\n" + "=" * 70)
    response = input("Proceed with cleaning? (yes/no): ")
    if response.lower() != "yes":
        print("Aborted.")
        return

    # Clean chunks
    print("\n" + "=" * 70)
    print("CLEANING CHUNKS")
    print("=" * 70)

    cleaned_chunks = []
    skipped = 0

    iterator = tqdm(chunks, desc="Cleaning") if show_progress else chunks

    for chunk in iterator:
        original_text = chunk["text"]
        cleaned_text = clean_html_text(original_text)

        # Skip if cleaning resulted in empty/very short text
        if len(cleaned_text.strip()) < 20:
            skipped += 1
            continue

        # Update chunk with cleaned text
        chunk["text"] = cleaned_text
        chunk["original_length"] = len(original_text)
        chunk["cleaned_length"] = len(cleaned_text)

        cleaned_chunks.append(chunk)

    print(f"\n✓ Cleaned {len(cleaned_chunks):,} chunks")
    print(f"⚠ Skipped {skipped:,} chunks (too short after cleaning)")

    # Save
    print(f"\nSaving to: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(cleaned_chunks, f)

    print(f"✓ Saved {len(cleaned_chunks):,} cleaned chunks")

    # Statistics
    print("\n" + "=" * 70)
    print("CLEANING STATISTICS")
    print("=" * 70)

    avg_original = sum(c["original_length"] for c in cleaned_chunks) / len(
        cleaned_chunks
    )
    avg_cleaned = sum(c["cleaned_length"] for c in cleaned_chunks) / len(cleaned_chunks)
    reduction = (1 - avg_cleaned / avg_original) * 100

    print(f"Average length before: {avg_original:.0f} chars")
    print(f"Average length after:  {avg_cleaned:.0f} chars")
    print(f"Size reduction:        {reduction:.1f}%")

    return cleaned_chunks


def verify_cleaning(output_path, num_samples=5):
    """Verify cleaned chunks look good"""

    print("\n" + "=" * 70)
    print("VERIFICATION - CHECKING CLEANED CHUNKS")
    print("=" * 70)

    with open(output_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"\nLoaded {len(chunks):,} cleaned chunks")

    # Check for remaining HTML
    has_html = sum(
        1 for c in chunks if any(m in c["text"] for m in ["<div", "<span", "<td"])
    )
    print(f"Chunks still with HTML tags: {has_html} ({100*has_html/len(chunks):.2f}%)")

    # Show random samples
    import random

    samples = random.sample(chunks, min(num_samples, len(chunks)))

    print(f"\n{num_samples} Random Samples:")
    for i, chunk in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Source: {chunk.get('source', 'unknown')}")
        print(f"Length: {len(chunk['text'])} chars")
        print(f"Text preview:\n{chunk['text'][:200]}")


def main():
    """Main cleaning pipeline"""

    # Configuration
    base_dir = Path("/scratch") / os.environ.get("USER", "unknown") / "finverify"
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
        print("\nNext steps:")
        print("1. Update your baseline code to use chunks_cleaned.pkl")
        print("2. Or rebuild BM25 index with cleaned chunks")
        print("3. Or copy chunks_cleaned.pkl to chunks.pkl (backup original first!)")


if __name__ == "__main__":
    main()
