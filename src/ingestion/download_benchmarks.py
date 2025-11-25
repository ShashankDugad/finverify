"""
Download FinanceBench and TATQA benchmark datasets
"""

import os
from pathlib import Path
from datasets import load_dataset
import json

def download_financebench(output_dir):
    """Download FinanceBench dataset from HuggingFace"""
    
    print("\n" + "=" * 60)
    print("Downloading FinanceBench")
    print("=" * 60)
    
    try:
        # FinanceBench is on HuggingFace
        dataset = load_dataset("PatronusAI/financebench", split="train")
        
        # Save as JSON
        output_file = output_dir / "financebench.json"
        
        data = []
        for item in dataset:
            data.append({
                'question': item['question'],
                'answer': item['answer'],
                'context': item.get('context', ''),
                'doc_name': item.get('doc_name', '')
            })
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Downloaded {len(data)} FinanceBench examples")
        print(f"  Saved to: {output_file}")
        
        return len(data)
        
    except Exception as e:
        print(f"✗ Failed to download FinanceBench: {e}")
        return 0

def download_tatqa(output_dir):
    """Download TATQA dataset from HuggingFace"""
    
    print("\n" + "=" * 60)
    print("Downloading TATQA")
    print("=" * 60)
    
    try:
        # TATQA dataset
        dataset = load_dataset("cais/tatqa", split="train")
        
        # Save as JSON
        output_file = output_dir / "tatqa.json"
        
        data = []
        for item in dataset:
            data.append({
                'question': item['question'],
                'answer': item.get('answer', ''),
                'table': item.get('table', {}),
                'paragraphs': item.get('paragraphs', [])
            })
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Downloaded {len(data)} TATQA examples")
        print(f"  Saved to: {output_file}")
        
        return len(data)
        
    except Exception as e:
        print(f"✗ Failed to download TATQA: {e}")
        print("  Note: TATQA may require manual download")
        return 0

def main():
    """Main download function"""
    
    # FinanceBench
    fb_dir = Path("/scratch") / os.environ['USER'] / "finverify" / "data" / "raw" / "financebench"
    fb_dir.mkdir(parents=True, exist_ok=True)
    
    # TATQA
    tatqa_dir = Path("/scratch") / os.environ['USER'] / "finverify" / "data" / "raw" / "tatqa"
    tatqa_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    fb_count = download_financebench(fb_dir)
    tatqa_count = download_tatqa(tatqa_dir)
    
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"FinanceBench: {fb_count} examples")
    print(f"TATQA: {tatqa_count} examples")

if __name__ == "__main__":
    main()
