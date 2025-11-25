"""
Download FULL FinanceBench dataset (all splits)
"""

import os
import json
from pathlib import Path
from datasets import load_dataset

def download_full_financebench():
    """Download complete FinanceBench dataset"""
    
    output_dir = Path("/scratch") / os.environ['USER'] / "finverify" / "data" / "raw" / "financebench"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Downloading FULL FinanceBench Dataset")
    print("=" * 60)
    
    try:
        # Load all splits
        dataset = load_dataset("PatronusAI/financebench")
        
        all_data = []
        
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            print(f"\n{split_name}: {len(split_data)} examples")
            
            for item in split_data:
                all_data.append({
                    'question': item['question'],
                    'answer': item['answer'],
                    'context': item.get('context', ''),
                    'doc_name': item.get('doc_name', ''),
                    'split': split_name
                })
        
        # Save combined
        output_file = output_dir / "financebench_full.json"
        with open(output_file, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"\n✓ Total: {len(all_data)} FinanceBench examples")
        print(f"✓ Saved to: {output_file}")
        
        return len(all_data)
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return 0

if __name__ == "__main__":
    download_full_financebench()
