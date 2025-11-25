"""
Basic Gradio demo for FinVERIFY retrieval
"""

import gradio as gr
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Load models and indexes
print("Loading models and indexes...")

base_dir = Path("/scratch/sd5957/finverify")

# Load BM25
with open(base_dir / "data/indexes/bm25/bm25_index.pkl", 'rb') as f:
    bm25 = pickle.load(f)

with open(base_dir / "data/indexes/bm25/chunks.pkl", 'rb') as f:
    chunks = pickle.load(f)

# Load FAISS
faiss_index = faiss.read_index(str(base_dir / "data/indexes/faiss/faiss_index.bin"))

# Load embedding model
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

print("✓ Models loaded")

def retrieve_bm25(query, top_k=5):
    """Retrieve using BM25"""
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    top_indices = scores.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        results.append({
            'source': chunk['source'],
            'text': chunk['text'][:300],
            'score': float(scores[idx])
        })
    
    return results

def retrieve_faiss(query, top_k=5):
    """Retrieve using FAISS"""
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    scores, indices = faiss_index.search(query_embedding, top_k)
    
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        chunk = chunks[idx]
        results.append({
            'source': chunk['source'],
            'text': chunk['text'][:300],
            'score': float(scores[0][i])
        })
    
    return results

def search(query, method):
    """Main search function"""
    if method == "BM25":
        results = retrieve_bm25(query, top_k=5)
    else:  # FAISS
        results = retrieve_faiss(query, top_k=5)
    
    output = f"## Query: {query}\n\n"
    output += f"**Method:** {method}\n\n"
    
    for i, result in enumerate(results, 1):
        output += f"### Result {i}\n"
        output += f"**Source:** {result['source']}  \n"
        output += f"**Score:** {result['score']:.3f}  \n"
        output += f"**Text:** {result['text']}...\n\n"
        output += "---\n\n"
    
    return output

# Create Gradio interface
demo = gr.Interface(
    fn=search,
    inputs=[
        gr.Textbox(label="Query", placeholder="Enter your financial question..."),
        gr.Radio(["BM25", "FAISS"], label="Retrieval Method", value="BM25")
    ],
    outputs=gr.Markdown(label="Results"),
    title="FinVERIFY - Financial Fact Retrieval Demo",
    description="Test BM25 and FAISS retrieval on SEC filings, FinanceBench, and TATQA datasets.",
    examples=[
        ["What is Apple's total revenue?", "BM25"],
        ["Microsoft cash position", "FAISS"],
        ["Tesla gross margin analysis", "BM25"],
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
