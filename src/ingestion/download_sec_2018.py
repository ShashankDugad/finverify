"""
Download SEC filings from 2018-2019 (pre-inline XBRL era)
These filings have separate HTML documents, not inline XBRL
"""

import os
from pathlib import Path
from sec_edgar_downloader import Downloader

def download_older_sec_filings():
    """
    Download SEC 10-K filings from 2018-2019
    
    Why 2018-2019?
    - These years have traditional HTML 10-K filings
    - Pre-inline XBRL mandate (which started ~2019-2020)
    - HTML contains readable narrative text
    - Separate from XBRL data files
    """
    
    output_dir = Path("/scratch") / os.environ['USER'] / "finverify" / "data" / "raw" / "sec_edgar_2018"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DOWNLOADING SEC 10-K FILINGS (2018-2019)")
    print("=" * 70)
    print("\nWhy 2018-2019?")
    print("  - Traditional HTML format (readable text)")
    print("  - Pre-inline XBRL mandate")
    print("  - Better for text-based RAG")
    print("\nDownloading to:", output_dir)
    
    # Initialize downloader
    dl = Downloader("NYU", "ua2152@nyu.edu", str(output_dir))
    
    # Top 50 companies by market cap
    companies = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "V", "UNH",
        "JNJ", "WMT", "XOM", "JPM", "LLY", "MA", "PG", "AVGO", "HD", "CVX",
        "MRK", "ABBV", "KO", "COST", "PEP", "ADBE", "TMO", "MCD", "CSCO", "ACN",
        "ABT", "NKE", "CRM", "DHR", "VZ", "LIN", "DIS", "TXN", "NEE", "CMCSA",
        "PM", "WFC", "ORCL", "BMY", "INTC", "UPS", "AMD", "RTX", "HON", "AMGN"
    ]
    
    total_downloaded = 0
    total_failed = 0
    failed_companies = []
    
    print(f"\nDownloading 10-Ks for {len(companies)} companies...")
    print("(2 filings per company = ~100 total filings)")
    print()
    
    for i, ticker in enumerate(companies, 1):
        try:
            print(f"[{i:2d}/{len(companies)}] {ticker:6s} ... ", end='', flush=True)
            
            # Download 2 filings from 2018-2019
            dl.get(
                "10-K", 
                ticker, 
                limit=2,
                after_date="2018-01-01",   # Start of 2018
                before_date="2019-12-31"   # End of 2019
            )
            
            total_downloaded += 2
            print(f"✓ Downloaded 2 filings")
            
        except Exception as e:
            total_failed += 1
            failed_companies.append(ticker)
            print(f"✗ Failed: {str(e)[:50]}")
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"\n✓ Successfully downloaded: {total_downloaded} filings")
    print(f"✗ Failed:                  {total_failed} companies")
    
    if failed_companies:
        print(f"\nFailed companies ({len(failed_companies)}):")
        for ticker in failed_companies[:10]:
            print(f"  - {ticker}")
        if len(failed_companies) > 10:
            print(f"  ... and {len(failed_companies) - 10} more")
    
    print(f"\nFiles saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Extract HTML:   python3 extract_html_from_2018.py")
    print("2. Chunk data:     python3 chunk_docs_diagnostic.py")
    print("3. Build indexes:  python3 build_bm25.py")
    
    return total_downloaded

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SEC EDGAR DOWNLOADER - 2018-2019 FILINGS")
    print("=" * 70)
    print("\nThis will download ~100 SEC 10-K filings from 2018-2019")
    print("These older filings have better HTML format for RAG.")
    print("\nEstimated time: 15-30 minutes")
    print("Estimated size: 2-3 GB")
    
    response = input("\nProceed with download? (yes/no): ")
    
    if response.lower() == 'yes':
        download_older_sec_filings()
    else:
        print("\nDownload cancelled.")
