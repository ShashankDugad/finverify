"""
Download expanded SEC filings (50 companies × 2 filings = 100 total)
"""

import os
from pathlib import Path
import subprocess

def download_expanded_sec():
    """Download SEC filings for 50 companies"""
    
    output_dir = Path("/scratch") / os.environ['USER'] / "finverify" / "data" / "raw" / "sec_edgar"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Downloading Expanded SEC Filings (50 companies)")
    print("=" * 60)
    
    from sec_edgar_downloader import Downloader
    
    dl = Downloader("NYU", "sd5957@nyu.edu", str(output_dir))
    
    # Top 50 companies by market cap
    companies = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "V", "UNH",
        "JNJ", "WMT", "XOM", "JPM", "LLY", "MA", "PG", "AVGO", "HD", "CVX",
        "MRK", "ABBV", "KO", "COST", "PEP", "ADBE", "TMO", "MCD", "CSCO", "ACN",
        "ABT", "NKE", "CRM", "DHR", "VZ", "LIN", "DIS", "TXN", "NEE", "CMCSA",
        "PM", "WFC", "ORCL", "BMY", "INTC", "UPS", "AMD", "RTX", "HON", "AMGN"
    ]
    
    total = 0
    failed = []
    
    for i, ticker in enumerate(companies, 1):
        try:
            print(f"\n[{i}/50] Downloading {ticker}...")
            dl.get("10-K", ticker, limit=2)
            total += 2
            print(f"  ✓ {ticker}")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")
            failed.append(ticker)
            continue
    
    print(f"\n" + "=" * 60)
    print(f"✓ Downloaded {total} SEC filings")
    print(f"✗ Failed: {len(failed)} companies")
    if failed:
        print(f"  {', '.join(failed[:10])}")
    print("=" * 60)
    
    return total

if __name__ == "__main__":
    download_expanded_sec()
