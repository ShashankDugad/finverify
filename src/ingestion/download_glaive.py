"""
Download Glaive RAG-v1 dataset for fine-tuning
"""

import os
from pathlib import Path
from datasets import load_dataset
import json

def download_glaive(output_dir, max_samples=10000):
    """Download Glaive RAG-v1 dataset"""
    
    print("\n" + "=" * 60)
    print("Downloading Glaive RAG-v1")
    print("=" * 60)
    
    try:
        # Load dataset
        dataset = load_dataset("glaiveai/RAG-v1", split="train")
        
        # Take subset
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        # Save as JSON
        output_file = output_dir / "glaive_rag_v1.json"
        
        data = []
        for item in dataset:
            data.append({
                'system': item.get('system', ''),
                'user': item.get('user', ''),
                'assistant': item.get('assistant', '')
            })
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Downloaded {len(data)} Glaive RAG examples")
        print(f"  Saved to: {output_file}")
        
        return len(data)
        
    except Exception as e:
        print(f"✗ Failed to download Glaive: {e}")
        return 0

def main():
    """Main download function"""
    
    output_dir = Path("/scratch") / os.environ['USER'] / "finverify" / "data" / "raw" / "glaive"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = download_glaive(output_dir, max_samples=10000)
    
    print(f"\n✓ Total: {count} examples")

if __name__ == "__main__":
    main()
