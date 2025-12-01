"""
Download specific company 10-Ks that your eval questions need
Manual/targeted approach for companies in your test set
"""

import os
import requests
from pathlib import Path
import time
from bs4 import BeautifulSoup

def get_company_cik(ticker):
    """Get CIK for a company ticker - All 50 companies"""
    cik_map = {
        'AAPL': '0000320193',
        'MSFT': '0000789019',
        'GOOGL': '0001652044',
        'AMZN': '0001018724',
        'NVDA': '0001045810',
        'META': '0001326801',
        'TSLA': '0001318605',
        'BRK-B': '0001067983',  # Berkshire Hathaway
        'V': '0001403161',
        'UNH': '0000731766',
        'JNJ': '0000200406',
        'WMT': '0000104169',
        'XOM': '0000034088',
        'JPM': '0000019617',
        'LLY': '0000059478',
        'MA': '0001141391',
        'PG': '0000080424',
        'AVGO': '0001730168',
        'HD': '0000354950',
        'CVX': '0000093410',
        'MRK': '0000310158',
        'ABBV': '0001551152',
        'KO': '0000021344',
        'COST': '0000909832',
        'PEP': '0000077476',
        'ADBE': '0000796343',
        'TMO': '0000097745',
        'MCD': '0000063908',
        'CSCO': '0000858877',
        'ACN': '0001467373',
        'ABT': '0000001800',
        'NKE': '0000320187',
        'CRM': '0001108524',
        'DHR': '0000313616',
        'VZ': '0000732712',
        'LIN': '0001707925',
        'DIS': '0001001039',
        'TXN': '0000097476',
        'NEE': '0000753308',
        'CMCSA': '0001166691',
        'PM': '0001413329',
        'WFC': '0000072971',
        'ORCL': '0001341439',
        'BMY': '0000014272',
        'INTC': '0000050863',
        'UPS': '0001090727',
        'AMD': '0000002488',
        'RTX': '0000101829',
        'HON': '0000773840',
        'AMGN': '0000318154',
    }
    return cik_map.get(ticker.upper())

def download_filing_html(ticker, year=2018):
    """
    Download 10-K HTML for a specific company and year
    
    Strategy: Use SEC EDGAR full-text search to find filing,
    then download the primary HTML document
    """
    print(f"\n{'='*60}")
    print(f"Downloading {ticker} 10-K from {year}")
    print(f"{'='*60}")
    
    cik = get_company_cik(ticker)
    if not cik:
        print(f"❌ CIK not found for {ticker}")
        return None
    
    # Search for filings
    search_url = f"https://www.sec.gov/cgi-bin/browse-edgar"
    params = {
        'action': 'getcompany',
        'CIK': cik,
        'type': '10-K',
        'dateb': f'{year}1231',
        'owner': 'exclude',
        'count': '10'
    }
    
    headers = {
        'User-Agent': 'NYU ua2152@nyu.edu'
    }
    
    try:
        print(f"1. Searching for {ticker} 10-K filings around {year}...")
        response = requests.get(search_url, params=params, headers=headers)
        time.sleep(0.2)  # Rate limit
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find filing links
        filing_links = soup.find_all('a', id='documentsbutton')
        
        if not filing_links:
            print(f"❌ No filings found for {ticker} in {year}")
            return None
        
        print(f"✓ Found {len(filing_links)} filings")
        
        # Get first filing's document page
        first_link = filing_links[0]['href']
        doc_url = f"https://www.sec.gov{first_link}"
        
        print(f"2. Fetching document list...")
        doc_response = requests.get(doc_url, headers=headers)
        time.sleep(0.2)
        
        doc_soup = BeautifulSoup(doc_response.text, 'html.parser')
        
        # Find the primary 10-K HTML document
        # Look for Type "10-K" and Format "HTML"
        doc_table = doc_soup.find('table', class_='tableFile')
        
        if not doc_table:
            print(f"❌ Could not find document table")
            return None
        
        rows = doc_table.find_all('tr')[1:]  # Skip header
        
        primary_doc_url = None
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 4:
                doc_type = cols[3].text.strip()
                description = cols[1].text.strip()
                
                # Look for primary 10-K document
                if doc_type == '10-K' or '10-K' in description:
                    link = cols[2].find('a')
                    if link:
                        primary_doc_url = f"https://www.sec.gov{link['href']}"
                        break
        
        if not primary_doc_url:
            # Fallback: get first HTML document
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 4:
                    link = cols[2].find('a')
                    if link and '.htm' in link['href']:
                        primary_doc_url = f"https://www.sec.gov{link['href']}"
                        break
        
        if not primary_doc_url:
            print(f"❌ Could not find HTML document")
            return None
        
        print(f"3. Downloading HTML from: {primary_doc_url}")
        html_response = requests.get(primary_doc_url, headers=headers)
        time.sleep(0.2)
        
        if html_response.status_code == 200:
            print(f"✓ Downloaded {len(html_response.text):,} bytes")
            return html_response.text
        else:
            print(f"❌ Download failed: {html_response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def download_multiple_companies():
    """Download 10-Ks for all 50 companies"""
    
    # All 50 companies from original script
    companies = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "V", "UNH",
        "JNJ", "WMT", "XOM", "JPM", "LLY", "MA", "PG", "AVGO", "HD", "CVX",
        "MRK", "ABBV", "KO", "COST", "PEP", "ADBE", "TMO", "MCD", "CSCO", "ACN",
        "ABT", "NKE", "CRM", "DHR", "VZ", "LIN", "DIS", "TXN", "NEE", "CMCSA",
        "PM", "WFC", "ORCL", "BMY", "INTC", "UPS", "AMD", "RTX", "HON", "AMGN"
    ]
    
    output_dir = Path("/scratch") / os.environ['USER'] / "finverify" / "data" / "raw" / "sec_manual"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MANUAL SEC 10-K DOWNLOAD - ALL 50 COMPANIES")
    print("=" * 60)
    print(f"\nDownloading {len(companies)} companies")
    print(f"Year: 2018 (traditional HTML format)")
    print(f"Output: {output_dir}")
    print(f"\nEstimated time: 30-45 minutes")
    print(f"Estimated size: 300-500 MB")
    
    success_count = 0
    failed = []
    
    for i, ticker in enumerate(companies, 1):
        print(f"\n[{i}/{len(companies)}] {ticker}")
        
        html_content = download_filing_html(ticker, year=2018)
        
        if html_content:
            # Save
            output_file = output_dir / f"{ticker}_10K_2018.html"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✓ Saved to: {output_file.name}")
            success_count += 1
        else:
            failed.append(ticker)
            print(f"✗ Failed: {ticker}")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"✓ Success: {success_count}/{len(companies)} companies")
    print(f"✗ Failed:  {len(failed)} companies")
    
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
    
    print(f"\nFiles saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Process downloaded HTML files")
    print("2. Chunk and add to existing chunks")
    print("3. Rebuild indexes")
    
    return success_count

if __name__ == "__main__":
    print("\nThis will download 50 company 10-K filings from 2018")
    print("Estimated time: 30-45 minutes (0.5-1 min per company)")
    print("Estimated size: 300-500 MB")
    print("\nNote: SEC rate limits to ~10 requests/second")
    print("Script includes delays to respect rate limits")
    
    response = input("\nProceed? (yes/no): ")
    
    if response.lower() == 'yes':
        download_multiple_companies()
    else:
        print("\nCancelled.")
