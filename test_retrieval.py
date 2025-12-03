"""
Simple test script for the retrieval system
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_faiss_retriever():
    """Test FAISS retriever"""
    print("=" * 60)
    print("Testing FAISS Retriever")
    print("=" * 60)
    
    try:
        from src.retrieval import FaissRetriever
        
        print("\nInitializing FAISS retriever...")
        retriever = FaissRetriever()
        
        query = "What is Apple's total revenue?"
        print(f"\nQuery: {query}")
        print("\nRetrieving results...")
        
        results = retriever.retrieve(query, top_k=3)
        
        print(f"\n✓ Retrieved {len(results)} results\n")
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Document: {result['metadata'].get('doc_name', 'N/A')}")
            print(f"  Content: {result['content'][:150]}...")
            print()
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you've completed Steps 1-4:")
        print("  1. Download FinanceBench PDFs")
        print("  2. Chunk PDFs (python src/chunking/pdf_chunker.py)")
        print("  3. Generate embeddings (python src/embeddings/generate_embeddings.py)")
        print("  4. Build FAISS index (python src/embeddings/build_faiss_index.py)")
        return False


def test_unified_retriever():
    """Test unified retriever"""
    print("\n" + "=" * 60)
    print("Testing Unified Retriever (Multi-Aspect)")
    print("=" * 60)
    
    try:
        from src.retrieval import UnifiedRetriever
        
        print("\nInitializing unified retriever...")
        retriever = UnifiedRetriever(
            methods=['faiss', 'entity', 'temporal'],
            use_reranker=True
        )
        
        query = "What is Microsoft's cash position in 2023?"
        print(f"\nQuery: {query}")
        print("\nRetrieving results (with RRF and reranking)...")
        
        results = retriever.retrieve(query, top_k=5)
        
        print(f"\n✓ Retrieved {len(results)} results\n")
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Document: {result['metadata'].get('doc_name', 'N/A')}")
            print(f"  Content: {result['content'][:150]}...")
            print()
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FinanceBench Retrieval System - Test Script")
    print("=" * 60)
    
    # Test FAISS
    faiss_ok = test_faiss_retriever()
    
    # Test Unified (only if FAISS works)
    if faiss_ok:
        unified_ok = test_unified_retriever()
    else:
        unified_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"FAISS Retriever: {'✓ PASS' if faiss_ok else '✗ FAIL'}")
    print(f"Unified Retriever: {'✓ PASS' if unified_ok else '✗ FAIL'}")
    print()

