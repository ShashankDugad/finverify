"""
FinanceBench Test Suite
Tests T5 model with RAG (using embeddings/chunks) on FinanceBench questions
Calculates F1 and EM scores

Author: FinVERIFY Team
Date: 2025
"""

import os
import json
import pickle
import time
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Import evaluation metrics
from evaluation_metrics import exact_match, f1_score, evaluate_predictions, print_metrics

# Conditional import for BM25 (only needed if using BM25 retrieval)
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None


def get_base_dir():
    """Determine base directory (local workspace or HPC scratch)"""
    workspace_path = Path(__file__).parent.parent.parent
    # Check if we're in a scratch directory
    if '/scratch/' in str(workspace_path) or 'USER' in str(workspace_path):
        import os
        return Path("/scratch") / os.environ.get('USER', 'user') / "finverify"
    return workspace_path


def load_config():
    """Load config.yaml"""
    base_dir = get_base_dir()
    config_path = base_dir / "config.yaml"
    if not config_path.exists():
        # Try relative to workspace
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class FinanceBenchTester:
    """
    Test suite for FinanceBench evaluation using T5 + RAG
    """
    
    def __init__(
        self,
        base_dir: str = None,
        embedding_model: str = None,  # If None, uses config.yaml
        generator_model: str = None,  # If None, uses config.yaml
        top_k: int = 5,
        max_evidence_length: int = 1024,
        device: str = None,
        retrieval_method: str = "faiss",  # "faiss", "bm25", or "both"
    ):
        """
        Initialize FinanceBench tester
        
        Args:
            base_dir: Base directory (if None, auto-detects from workspace)
            embedding_model: Model for semantic retrieval (if None, uses config.yaml)
            generator_model: T5 model for answer generation (if None, uses config.yaml)
            top_k: Number of chunks to retrieve
            max_evidence_length: Max tokens for evidence context
            device: Device to use (if None, auto-detects: cuda/mps/cpu)
            retrieval_method: "faiss", "bm25", or "both"
        """
        # Load config
        config = load_config()
        
        # Setup paths
        if base_dir is None:
            base_dir = get_base_dir()
        else:
            base_dir = Path(base_dir)
        
        self.base_dir = base_dir
        self.faiss_dir = base_dir / "data" / "indexes" / "faiss"
        self.bm25_dir = base_dir / "data" / "indexes" / "bm25"
        self.top_k = top_k
        self.max_evidence_length = max_evidence_length
        self.retrieval_method = retrieval_method.lower()
        
        # Validate retrieval method
        if self.retrieval_method not in ["faiss", "bm25", "both"]:
            raise ValueError(f"retrieval_method must be 'faiss', 'bm25', or 'both', got: {retrieval_method}")
        
        # Get model names from config if not provided
        if embedding_model is None:
            embedding_model = config['models']['embedder']['name']
        if generator_model is None:
            generator_model = config['models']['generator']['name']
        
        # Auto-detect best device if not specified (same logic as embedding generation)
        if device is None:
            device = config['models']['embedder'].get('device', None)
            if device is None:
                # Auto-detect: prefer CUDA, then MPS, then CPU
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
        
        # Handle device fallback (same as embedding generation)
        if device == "cuda" and not torch.cuda.is_available():
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("CUDA not available, using MPS (Apple Silicon GPU) instead")
                device = "mps"
            else:
                print("CUDA not available, using CPU")
                device = "cpu"
        elif device == "mps":
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                print("MPS not available, using CPU")
                device = "cpu"
        
        self.device = device
        
        print(f"Device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            # Enable optimizations for CUDA (same as embedding generation)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        elif device == "mps":
            print("GPU: Apple Silicon (MPS)")
        print(f"Retrieval method: {self.retrieval_method.upper()}")
        
        # Load retrieval systems
        if self.retrieval_method in ["faiss", "both"]:
            # Load embedding model for FAISS (same approach as embedding generation)
            print(f"\nLoading embedding model: {embedding_model}...")
            self.embedder = SentenceTransformer(embedding_model, device=self.device)
            
            # Verify model is on correct device (same as embedding generation)
            if device == "cuda":
                model_device = next(self.embedder[0].parameters()).device
                if model_device.type != "cuda":
                    print(f"⚠ Warning: Model loaded on {model_device}, expected cuda")
                else:
                    print(f"✓ Model confirmed on GPU: {model_device}")
            
            # Load FAISS index
            print("\nLoading FAISS index...")
            self._load_faiss_index()
        
        if self.retrieval_method in ["bm25", "both"]:
            if not BM25_AVAILABLE:
                raise ImportError(
                    "rank_bm25 package is required for BM25 retrieval. "
                    "Install it with: pip install rank-bm25"
                )
            # Load BM25 index
            print("\nLoading BM25 index...")
            self._load_bm25_index()
        
        # Load chunks (needed for both methods)
        if not hasattr(self, 'chunks'):
            self._load_chunks()
        
        # Load T5 generator
        print(f"\nLoading generator: {generator_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.generator = T5ForConditionalGeneration.from_pretrained(generator_model)
        self.generator.to(self.device)
        self.generator.eval()
        
        # Verify generator is on correct device
        if device == "cuda":
            gen_device = next(self.generator.parameters()).device
            if gen_device.type != "cuda":
                print(f"⚠ Warning: Generator loaded on {gen_device}, expected cuda")
            else:
                print(f"✓ Generator confirmed on GPU: {gen_device}")
        
        print("✓ All models loaded successfully")
    
    def _load_chunks(self):
        """Load chunks from multiple possible locations"""
        possible_chunk_paths = [
            self.bm25_dir / "chunks.pkl",
            self.base_dir / "data" / "processed" / "chunks.jsonl",
            self.base_dir / "data" / "processed" / "chunks.json",
            self.base_dir / "data" / "files" / "processed" / "chunks.jsonl",
        ]
        
        chunks_path = None
        for path in possible_chunk_paths:
            if path.exists():
                chunks_path = path
                break
        
        if chunks_path is None:
            raise FileNotFoundError(
                f"Chunks not found. Tried: {possible_chunk_paths}"
            )
        
        # Load chunks based on file extension
        print(f"Loading chunks from: {chunks_path}")
        if chunks_path.suffix == ".pkl":
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
        elif chunks_path.suffix == ".jsonl":
            self.chunks = []
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.chunks.append(json.loads(line))
        else:  # .json
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        
        print(f"✓ Loaded chunks: {len(self.chunks):,} chunks")
    
    def _load_faiss_index(self):
        """Load FAISS index"""
        # Load FAISS index
        index_path = self.faiss_dir / "faiss_index.bin"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        self.faiss_index = faiss.read_index(str(index_path))
        print(f"✓ Loaded FAISS index: {self.faiss_index.ntotal:,} vectors, dimension: {self.faiss_index.d}")
    
    def _load_bm25_index(self):
        """Load BM25 index"""
        # Try multiple locations for BM25 index
        possible_bm25_paths = [
            self.bm25_dir / "bm25_index.pkl",
            self.bm25_dir / "index.pkl",
        ]
        
        bm25_path = None
        for path in possible_bm25_paths:
            if path.exists():
                bm25_path = path
                break
        
        if bm25_path is None:
            raise FileNotFoundError(
                f"BM25 index not found. Tried: {possible_bm25_paths}"
            )
        
        with open(bm25_path, "rb") as f:
            self.bm25_index = pickle.load(f)
        
        # BM25Okapi doesn't expose corpus directly, but we can check if it's loaded
        # The number of documents is stored internally
        print(f"✓ Loaded BM25 index successfully")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve top-K chunks using specified retrieval method
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve (default: self.top_k)
        
        Returns:
            List of chunks with similarity scores
        """
        if top_k is None:
            top_k = self.top_k
        
        if self.retrieval_method == "faiss":
            return self._retrieve_faiss(query, top_k)
        elif self.retrieval_method == "bm25":
            return self._retrieve_bm25(query, top_k)
        elif self.retrieval_method == "both":
            return self._retrieve_hybrid(query, top_k)
        else:
            raise ValueError(f"Unknown retrieval method: {self.retrieval_method}")
    
    def _retrieve_faiss(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using FAISS semantic search"""
        # Encode query (same approach as embedding generation)
        # Use torch.no_grad() for inference to save memory and speed up
        with torch.no_grad():
            query_embedding = self.embedder.encode(
                [query],
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,  # Normalize for cosine similarity (same as embedding generation)
                device=self.device  # Explicitly specify device
            )
        
        # Note: normalize_embeddings=True already normalizes, but FAISS index expects normalized vectors
        # Double-check normalization (embeddings in index are already normalized)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            chunk["rank"] = rank
            chunk["method"] = "faiss"
            results.append(chunk)
        
        return results
    
    def _retrieve_bm25(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using BM25 keyword search"""
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-K indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        # Prepare results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            if idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            chunk["rank"] = rank
            chunk["method"] = "bm25"
            results.append(chunk)
        
        return results
    
    def _retrieve_hybrid(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using both FAISS and BM25, then merge results"""
        # Get results from both methods
        faiss_results = self._retrieve_faiss(query, top_k * 2)  # Get more for merging
        bm25_results = self._retrieve_bm25(query, top_k * 2)
        
        # Simple merge: combine and deduplicate by chunk_id
        seen_ids = set()
        merged = []
        
        # Add FAISS results first (higher weight for semantic)
        for result in faiss_results:
            chunk_id = result.get("chunk_id", str(result.get("text", ""))[:50])
            if chunk_id not in seen_ids:
                result["score"] = result["score"] * 1.2  # Boost FAISS scores slightly
                merged.append(result)
                seen_ids.add(chunk_id)
        
        # Add BM25 results
        for result in bm25_results:
            chunk_id = result.get("chunk_id", str(result.get("text", ""))[:50])
            if chunk_id not in seen_ids:
                merged.append(result)
                seen_ids.add(chunk_id)
        
        # Sort by score and return top_k
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:top_k]
    
    def format_prompt(self, query: str, evidence: List[Dict]) -> str:
        """
        Format prompt for T5 model
        
        Args:
            query: Question
            evidence: Retrieved chunks
        
        Returns:
            Formatted prompt string
        """
        # Build evidence text
        evidence_texts = []
        for i, chunk in enumerate(evidence, 1):
            text = chunk["text"].strip()
            source = chunk.get("source", "unknown")
            evidence_texts.append(f"[{i}] {text} (Source: {source})")
        
        evidence_str = "\n\n".join(evidence_texts)
        
        # Truncate if too long
        if len(evidence_str) > self.max_evidence_length * 4:  # Rough char estimate
            evidence_str = evidence_str[: self.max_evidence_length * 4] + "..."
        
        # Create prompt
        prompt = f"""Answer the following question based on the provided evidence.

Question: {query}

Evidence:
{evidence_str}

Answer:"""
        
        return prompt
    
    def generate(self, prompt: str, max_length: int = 128) -> str:
        """
        Generate answer using T5
        
        Args:
            prompt: Input prompt
            max_length: Max length of generated answer
        
        Returns:
            Generated answer
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_evidence_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate (use torch.no_grad() for inference to save memory)
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()
    
    def predict(self, query: str) -> Dict:
        """
        End-to-end prediction pipeline
        
        Args:
            query: Question
        
        Returns:
            Prediction dictionary with answer and timing
        """
        start_time = time.time()
        
        # Retrieve
        retrieve_start = time.time()
        evidence = self.retrieve(query)
        retrieve_time = time.time() - retrieve_start
        
        # Generate
        generate_start = time.time()
        prompt = self.format_prompt(query, evidence)
        answer = self.generate(prompt)
        generate_time = time.time() - generate_start
        
        total_time = time.time() - start_time
        
        return {
            "question": query,
            "answer": answer,
            "evidence": evidence,
            "retrieve_time_ms": retrieve_time * 1000,
            "generate_time_ms": generate_time * 1000,
            "total_time_ms": total_time * 1000,
        }
    
    def load_financebench(self, num_questions: Optional[int] = None) -> List[Dict]:
        """
        Load FinanceBench questions
        
        Args:
            num_questions: Limit number of questions (None = all)
        
        Returns:
            List of FinanceBench question dictionaries
        """
        # Try multiple possible locations
        possible_paths = [
            self.base_dir / "data" / "raw" / "financebench" / "financebench.json",
            self.base_dir / "data" / "raw" / "financebench" / "financebench_full.json",
            self.base_dir / "data" / "raw" / "financebench" / "financebench_open_source.jsonl",
            self.base_dir / "financebench" / "data" / "financebench_open_source.jsonl",
            self.base_dir / "financebench" / "data" / "financebench.json",
            self.base_dir.parent / "financebench" / "data" / "financebench_open_source.jsonl",
        ]
        
        questions = []
        data_path = None
        
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(
                f"FinanceBench data not found. Tried: {possible_paths}"
            )
        
        print(f"\nLoading FinanceBench from: {data_path}")
        
        # Load JSON or JSONL
        if data_path.suffix == ".jsonl":
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    questions = data
                elif isinstance(data, dict) and "questions" in data:
                    questions = data["questions"]
                else:
                    questions = [data]
        
        if num_questions:
            questions = questions[:num_questions]
        
        print(f"✓ Loaded {len(questions)} FinanceBench questions")
        
        return questions
    
    def evaluate(
        self, 
        questions: List[Dict], 
        save_results: bool = True,
        output_file: Optional[Path] = None
    ) -> Dict:
        """
        Evaluate on FinanceBench questions
        
        Args:
            questions: List of FinanceBench question dictionaries
            save_results: Whether to save results to file
            output_file: Path to save results (default: outputs/results/financebench_results.json)
        
        Returns:
            Dictionary with metrics and predictions
        """
        print("\n" + "=" * 70)
        print("FINANCEBENCH EVALUATION")
        print("=" * 70)
        print(f"Questions: {len(questions)}")
        print(f"Retrieval: {self.retrieval_method.upper()} (top-{self.top_k})")
        print(f"Generator: T5")
        print("=" * 70)
        
        # Run predictions
        predictions = []
        ground_truth = []
        
        print("\nGenerating predictions...")
        for i, question_data in enumerate(tqdm(questions, desc="Processing")):
            query = question_data.get("question", "")
            if not query:
                continue
            
            # Get ground truth answer
            gt_answer = question_data.get("answer", "")
            if not gt_answer or gt_answer.lower() in ["unknown", "n/a", "none"]:
                # Skip if no ground truth
                continue
            
            # Generate prediction
            try:
                result = self.predict(query)
                result["ground_truth"] = gt_answer
                predictions.append(result)
                ground_truth.append({"question": query, "answer": gt_answer})
            except Exception as e:
                import traceback
                print(f"\n⚠️  Error processing question {i}: {e}")
                print(f"   Question: {query[:100]}...")
                traceback.print_exc()
                continue
        
        print(f"\n✓ Generated {len(predictions)} predictions")
        
        if len(predictions) == 0:
            print("⚠️  No valid predictions generated!")
            return {}
        
        # Evaluate
        print("\nEvaluating results...")
        metrics = evaluate_predictions(predictions, ground_truth)
        
        # Print results
        print_metrics(metrics, "FinanceBench T5+RAG")
        
        # Save results
        if save_results:
            if output_file is None:
                output_dir = self.base_dir / "outputs" / "results"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / "financebench_results.json"
            
            # Get actual model names used
            config = load_config()
            results = {
                "metrics": metrics,
                "predictions": predictions,
                "config": {
                    "top_k": self.top_k,
                    "max_evidence_length": self.max_evidence_length,
                    "generator_model": config['models']['generator']['name'],
                    "embedding_model": config['models']['embedder']['name'],
                    "retrieval_method": self.retrieval_method,
                    "device": str(self.device),
                }
            }
            
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✓ Results saved to: {output_file}")
        
        return metrics


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="FinanceBench Test Suite")
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions to evaluate (default: all)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory (default: /scratch/$USER/finverify)",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default=None,
        help="T5 model for generation (default: from config.yaml)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model for FAISS (default: from config.yaml)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for results (default: outputs/results/financebench_results.json)",
    )
    parser.add_argument(
        "--retrieval-method",
        type=str,
        default="faiss",
        choices=["faiss", "bm25", "both"],
        help="Retrieval method: 'faiss' (semantic), 'bm25' (keyword), or 'both' (hybrid)",
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = FinanceBenchTester(
        base_dir=args.base_dir,
        top_k=args.top_k,
        generator_model=args.generator_model,
        embedding_model=args.embedding_model,
        retrieval_method=args.retrieval_method,
    )
    
    # Load FinanceBench questions
    questions = tester.load_financebench(num_questions=args.num_questions)
    
    # Evaluate
    output_file = Path(args.output_file) if args.output_file else None
    metrics = tester.evaluate(questions, save_results=True, output_file=output_file)
    
    print("\n" + "=" * 70)
    print("✓ EVALUATION COMPLETE")
    print("=" * 70)
    if metrics:
        print(f"\nFinal Scores:")
        print(f"  Exact Match: {metrics['exact_match']:.2f}%")
        print(f"  F1 Score:    {metrics['f1_score']:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()

