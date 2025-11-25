"""
Download REAL SEC filings using sec-edgar-downloader library
"""

import os
from pathlib import Path

def download_real_sec():
    """Download real SEC filings"""
    
    output_dir = Path("/scratch") / os.environ['USER'] / "finverify" / "data" / "raw" / "sec_edgar"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Installing sec-edgar-downloader...")
    print("=" * 60)
    
    import subprocess
    subprocess.run(["pip", "install", "sec-edgar-downloader", "--break-system-packages", "-q"])
    
    from sec_edgar_downloader import Downloader
    
    print("\n" + "=" * 60)
    print("Downloading Real SEC Filings")
    print("=" * 60)
    
    # Initialize downloader
    dl = Downloader("NYU", "sd5957@nyu.edu", str(output_dir))
    
    # Download sample filings - top tech companies
    companies = [
        ("AAPL", "Apple Inc"),
        ("MSFT", "Microsoft"),
        ("GOOGL", "Alphabet"),
        ("AMZN", "Amazon"),
        ("TSLA", "Tesla")
    ]
    
    total_downloaded = 0
    
    for ticker, name in companies:
        try:
            print(f"\nDownloading {name} ({ticker}) 10-K filings...")
            # Download latest 2 10-K filings
            dl.get("10-K", ticker, limit=2)
            total_downloaded += 2
            print(f"  ✓ Downloaded 2 filings for {name}")
        except Exception as e:
            print(f"  ✗ Failed for {ticker}: {e}")
            continue
    
    print(f"\n✓ Total downloaded: {total_downloaded} SEC filings")
    
    return total_downloaded

if __name__ == "__main__":
    download_real_sec()
