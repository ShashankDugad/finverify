import json
from pathlib import Path

base_dir = Path("/scratch/ua2152/finverify")
chunks_path = base_dir / "data/processed/chunks.json"

# Load existing
chunks = json.load(open(chunks_path))

# Keep only FinanceBench and TATQA
good_chunks = [c for c in chunks if c["source"] in ["financebench", "tatqa"]]

# Stats
sources = {}
for c in good_chunks:
    sources[c["source"]] = sources.get(c["source"], 0) + 1

print(f"Total chunks: {len(good_chunks):,}")
for source, count in sources.items():
    print(f"  {source}: {count:,}")

# Save
json.dump(good_chunks, open(chunks_path, "w"))
print(f"\n✓ Saved {len(good_chunks):,} chunks")
print("\nNext: Rebuild indexes")
