"""
DPR (Dense Passage Retrieval) + Flan-T5-base Baseline
Uses FAISS semantic search with T5 generation

Author: Utkarsh
Date: November 2025
"""

import os
import json
import pickle
import time
import argparse
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Import evaluation metrics
try:
    from evaluation_metrics import evaluate_predictions, print_metrics
except ImportError:
    print("⚠️  evaluation_metrics.py not found. Using basic metrics only.")
    evaluate_predictions = None
    print_metrics = None


class DPRT5Baseline:
    """DPR + T5 baseline for financial QA"""

    def __init__(
        self,
        base_dir: str = None,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        generator_model: str = "google/flan-t5-base",
        top_k: int = 5,
        max_evidence_length: int = 1024,
        device: str = None,
    ):
        """
        Initialize DPR+T5 baseline

        Args:
            base_dir: Base directory (defaults to /scratch/$USER/finverify)
            embedding_model: Embedding model name
            generator_model: T5 model name
            top_k: Number of chunks to retrieve
            max_evidence_length: Max tokens for evidence
            device: Device to use (cuda/cpu)
        """
        # Setup paths
        if base_dir is None:
            user = os.environ.get("USER", "unknown")
            base_dir = Path("/scratch") / user / "finverify"
        else:
            base_dir = Path(base_dir)

        self.base_dir = base_dir
        self.faiss_dir = base_dir / "data" / "indexes" / "faiss"
        self.bm25_dir = base_dir / "data" / "indexes" / "bm25"  # Chunks stored here
        self.top_k = top_k
        self.max_evidence_length = max_evidence_length

        # Setup device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load embedding model
        print(f"\nLoading embedding model: {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model)
        self.embedder.to(self.device)

        # Load FAISS index
        print("\nLoading FAISS index...")
        self._load_faiss_index()

        # Load T5 model
        print(f"\nLoading generator: {generator_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.generator = T5ForConditionalGeneration.from_pretrained(generator_model)
        self.generator.to(self.device)
        self.generator.eval()

        print("✓ All models loaded successfully")

    def _load_faiss_index(self):
        """Load FAISS index and chunks"""
        # Load FAISS index
        index_path = self.faiss_dir / "faiss_index.bin"
        self.index = faiss.read_index(str(index_path))

        # Load chunks (stored with BM25 index)
        with open(self.bm25_dir / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        print(f"✓ Loaded FAISS index: {self.index.ntotal:,} vectors")
        print(f"✓ Loaded chunks: {len(self.chunks):,} chunks")

        assert self.index.ntotal == len(self.chunks), "Index size mismatch!"

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve top-K chunks using FAISS semantic search

        Args:
            query: Query string
            top_k: Number of chunks to retrieve (default: self.top_k)

        Returns:
            List of chunks with similarity scores
        """
        if top_k is None:
            top_k = self.top_k

        # Encode query
        query_embedding = self.embedder.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Prepare results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            chunk["rank"] = rank
            results.append(chunk)

        return results

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

        # Generate
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

    def predict(self, query: str, return_evidence: bool = False) -> Dict:
        """
        End-to-end prediction pipeline

        Args:
            query: Question
            return_evidence: Whether to return retrieved evidence

        Returns:
            Prediction dictionary
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

        result = {
            "question": query,
            "answer": answer,
            "retrieve_time_ms": retrieve_time * 1000,
            "generate_time_ms": generate_time * 1000,
            "total_time_ms": total_time * 1000,
        }

        if return_evidence:
            result["evidence"] = [
                {
                    "text": e["text"][:200] + "...",
                    "source": e.get("source", "unknown"),
                    "score": e["score"],
                }
                for e in evidence
            ]

        return result


def load_financebench(data_path: Path, num_questions: int = None) -> List[Dict]:
    """Load FinanceBench dataset"""
    fb_path = data_path / "raw" / "financebench"

    # Check different possible file locations
    possible_files = [
        fb_path / "financebench.json",
        fb_path / "questions.json",
        fb_path / "test.json",
    ]

    questions = []
    for file_path in possible_files:
        if file_path.exists():
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    questions = data
                elif isinstance(data, dict) and "questions" in data:
                    questions = data["questions"]
                break

    if not questions:
        print(f"⚠️  Could not find FinanceBench data in {fb_path}")
        print("Creating sample questions for testing...")
        questions = [
            {
                "question": "What is Apple's total revenue for fiscal year 2024?",
                "answer": "Unknown",
            },
            {"question": "What was Microsoft's operating income?", "answer": "Unknown"},
            {"question": "How much cash does Tesla have?", "answer": "Unknown"},
        ]

    if num_questions:
        questions = questions[:num_questions]

    return questions


def evaluate_predictions_basic(
    predictions: List[Dict], ground_truth: List[Dict]
) -> Dict:
    """
    Basic evaluation (latency only) - fallback if evaluation_metrics.py not available

    Note: This is a simplified evaluation. For full metrics (EM, F1, Recall@10),
    use the evaluation_metrics module.
    """
    metrics = {
        "num_questions": len(predictions),
        "avg_retrieve_time_ms": 0,
        "avg_generate_time_ms": 0,
        "avg_total_time_ms": 0,
    }

    # Calculate average times
    for pred in predictions:
        metrics["avg_retrieve_time_ms"] += pred["retrieve_time_ms"]
        metrics["avg_generate_time_ms"] += pred["generate_time_ms"]
        metrics["avg_total_time_ms"] += pred["total_time_ms"]

    if metrics["num_questions"] > 0:
        metrics["avg_retrieve_time_ms"] /= metrics["num_questions"]
        metrics["avg_generate_time_ms"] /= metrics["num_questions"]
        metrics["avg_total_time_ms"] /= metrics["num_questions"]

    print("\n⚠️  Using basic metrics (latency only)")
    print(
        "   For full metrics (EM, F1, Recall@10), ensure evaluation_metrics.py is present"
    )

    return metrics


def test_mode():
    """Test on sample questions"""
    print("=" * 70)
    print("DPR+T5 BASELINE - TEST MODE")
    print("=" * 70)

    # Initialize model
    baseline = DPRT5Baseline(top_k=3)

    # Test queries
    test_queries = [
        "What is Apple's total revenue?",
        "How much cash does Microsoft have?",
        "What is Tesla's gross margin?",
        "What are Amazon's operating expenses?",
        "What is Google's advertising revenue?",
    ]

    print("\n" + "=" * 70)
    print("TESTING ON SAMPLE QUESTIONS")
    print("=" * 70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] {query}")
        print("-" * 70)

        result = baseline.predict(query, return_evidence=True)

        print(f"Answer: {result['answer']}")
        print(
            f"Time: {result['total_time_ms']:.0f}ms (retrieve: {result['retrieve_time_ms']:.0f}ms, generate: {result['generate_time_ms']:.0f}ms)"
        )
        print(f"\nTop Retrieved Chunks:")
        for j, ev in enumerate(result["evidence"], 1):
            print(f"  [{j}] Similarity: {ev['score']:.3f} | Source: {ev['source']}")
            print(f"      {ev['text']}")

    print("\n" + "=" * 70)
    print("✓ Test complete!")
    print("=" * 70)


def eval_mode(num_questions: int = 50):
    """Evaluate on FinanceBench dataset"""
    print("=" * 70)
    print(f"DPR+T5 BASELINE - EVALUATION MODE ({num_questions} questions)")
    print("=" * 70)

    # Initialize model
    baseline = DPRT5Baseline()

    # Load questions
    questions = load_financebench(baseline.base_dir / "data", num_questions)
    print(f"\n✓ Loaded {len(questions)} questions")

    # Run predictions
    print("\nGenerating predictions...")
    predictions = []

    for question_data in tqdm(questions):
        query = question_data["question"]
        result = baseline.predict(query, return_evidence=True)

        # Add ground truth if available
        if "answer" in question_data:
            result["ground_truth"] = question_data["answer"]

        predictions.append(result)

    # Evaluate with comprehensive or basic metrics
    print("\nEvaluating results...")

    if evaluate_predictions is not None:
        # Use comprehensive metrics (EM, F1, Recall@10)
        metrics = evaluate_predictions(
            predictions, questions, compute_retrieval_metrics=False
        )

        # Print formatted results
        if print_metrics is not None:
            print_metrics(metrics, "DPR+T5")
        else:
            # Manual print if print_metrics not available
            print("\n" + "=" * 70)
            print("EVALUATION RESULTS")
            print("=" * 70)
            print(f"Exact Match: {metrics.get('exact_match', 0):.2f}%")
            print(f"F1 Score: {metrics.get('f1_score', 0):.2f}%")
            print(f"Avg Retrieval Time: {metrics['avg_retrieve_time_ms']:.1f}ms")
            print(f"Avg Generation Time: {metrics['avg_generate_time_ms']:.1f}ms")
            print(f"Avg Total Time: {metrics['avg_total_time_ms']:.1f}ms")
            print("=" * 70)
    else:
        # Fall back to basic metrics (latency only)
        metrics = evaluate_predictions_basic(predictions, questions)

        # Print basic metrics
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS (Basic)")
        print("=" * 70)
        print(f"Questions: {metrics['num_questions']}")
        print(f"Avg Retrieval Time: {metrics['avg_retrieve_time_ms']:.1f}ms")
        print(f"Avg Generation Time: {metrics['avg_generate_time_ms']:.1f}ms")
        print(f"Avg Total Time: {metrics['avg_total_time_ms']:.1f}ms")
        print("=" * 70)

    # Save results
    output_dir = baseline.base_dir / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "dpr_t5_results.json"
    with open(results_file, "w") as f:
        json.dump({"metrics": metrics, "predictions": predictions}, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="DPR+T5 Baseline")
    parser.add_argument(
        "--test-mode", action="store_true", help="Run test on sample questions"
    )
    parser.add_argument(
        "--eval", action="store_true", help="Run evaluation on FinanceBench"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=50,
        help="Number of questions for evaluation",
    )
    parser.add_argument("--base-dir", type=str, default=None, help="Base directory")

    args = parser.parse_args()

    if args.test_mode:
        test_mode()
    elif args.eval:
        eval_mode(args.num_questions)
    else:
        print("Usage:")
        print("  python dpr_t5.py --test-mode          # Test on 5 sample questions")
        print("  python dpr_t5.py --eval               # Evaluate on 50 questions")
        print(
            "  python dpr_t5.py --eval --num-questions 100  # Evaluate on 100 questions"
        )


if __name__ == "__main__":
    main()
