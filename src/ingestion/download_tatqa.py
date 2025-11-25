"""
Download TATQA dataset from GitHub
"""

import os
import json
import requests
from pathlib import Path

def download_tatqa():
    """Download TATQA from GitHub repository"""
    
    output_dir = Path("/scratch") / os.environ['USER'] / "finverify" / "data" / "raw" / "tatqa"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Downloading TATQA Dataset")
    print("=" * 60)
    
    base_url = "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw"
    
    files = ["tatqa_dataset_train.json", "tatqa_dataset_dev.json"]
    
    total = 0
    
    for filename in files:
        url = f"{base_url}/{filename}"
        print(f"\nDownloading {filename}...")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            with open(output_dir / filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            count = len(data) if isinstance(data, list) else len(data.get('data', []))
            total += count
            print(f"✓ {count} examples")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    print(f"\n✓ Total: {total} TATQA examples")
    
    return total

if __name__ == "__main__":
    download_tatqa()
