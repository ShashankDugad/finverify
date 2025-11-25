"""
Download FULL TATQA dataset (all files)
"""

import os
import json
import requests
from pathlib import Path

def download_full_tatqa():
    """Download complete TATQA dataset"""
    
    output_dir = Path("/scratch") / os.environ['USER'] / "finverify" / "data" / "raw" / "tatqa"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Downloading FULL TATQA Dataset")
    print("=" * 60)
    
    base_url = "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw"
    
    files = [
        "tatqa_dataset_train.json",
        "tatqa_dataset_dev.json",
        "tatqa_dataset_test.json"
    ]
    
    total = 0
    
    for filename in files:
        url = f"{base_url}/{filename}"
        output_file = output_dir / filename
        
        print(f"\nDownloading {filename}...")
        
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            count = len(data) if isinstance(data, list) else len(data.get('data', []))
            total += count
            print(f"✓ {count} examples")
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"✗ File not found (may not exist): {filename}")
            else:
                print(f"✗ Failed: {e}")
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    print(f"\n✓ Total TATQA: {total} examples")
    
    return total

if __name__ == "__main__":
    download_full_tatqa()
