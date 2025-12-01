"""
Download SEC filings - Version compatible with sec-edgar-downloader
Downloads recent filings and we'll manually check which are good
"""

import os
from pathlib import Path
from sec_edgar_downloader import Downloader

def download_sec_filings_simple():
    """
    Download SEC 10-K filings without date filters
    
    Strategy: Download recent filings and check which format they are
    Some companies still file traditional HTML even in recent years
    """
    
    output_dir = Path("/scratch") / os.environ['USER'] / "finverify" / "data" / "raw" / "sec_edgar_simple"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DOWNLOADING SEC 10-K FILINGS (Simple Method)")
    print("=" * 70)
    print("\nStrategy: Download 2 recent filings per company")
    print("We'll check which ones have usable HTML format")
    print("\nDownloading to:", output_dir)
    
    # Initialize downloader
    dl = Downloader("NYU", "ua2152@nyu.edu", str(output_dir))
    
    # Top 50 companies
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
    print("(Requesting 2 most recent filings per company)")
    print()
    
    for i, ticker in enumerate(companies, 1):
        try:
            print(f"[{i:2d}/{len(companies)}] {ticker:6s} ... ", end='', flush=True)
            
            # Download without date filters
            # Library will get the most recent filings
            dl.get(
                "10-K", 
                ticker, 
                limit=2  # Get 2 most recent
            )
            
            total_downloaded += 1
            print(f"✓ Downloaded")
            
        except Exception as e:
            total_failed += 1
            failed_companies.append(ticker)
            print(f"✗ Failed: {str(e)[:50]}")
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"\n✓ Successfully downloaded: {total_downloaded} companies")
    print(f"✗ Failed:                  {total_failed} companies")
    
    if failed_companies:
        print(f"\nFailed companies ({len(failed_companies)}):")
        for ticker in failed_companies[:10]:
            print(f"  - {ticker}")
        if len(failed_companies) > 10:
            print(f"  ... and {len(failed_companies) - 10} more")
    
    print(f"\nFiles saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Check file formats: python3 check_file_formats.py")
    print("2. Extract HTML:       python3 extract_html_improved.py")
    print("3. Chunk data:         python3 chunk_docs_diagnostic.py")
    
    return total_downloaded

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SEC EDGAR DOWNLOADER - Simple Method")
    print("=" * 70)
    print("\nThis will download ~100 SEC 10-K filings (2 per company)")
    print("We'll download recent filings and check which format they are")
    print("\nEstimated time: 15-30 minutes")
    print("Estimated size: 2-3 GB")
    
    response = input("\nProceed with download? (yes/no): ")
    
    if response.lower() == 'yes':
        download_sec_filings_simple()
    else:
        print("\nDownload cancelled.")
