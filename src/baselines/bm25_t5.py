"""
BM25 + Flan-T5-base Baseline
Combines keyword retrieval with T5 generation

Author: Utkarsh
Date: November 2025
"""

import os
import json
import pickle
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


class BM25T5Baseline:
    """BM25 + T5 baseline for financial QA"""

    def __init__(
        self,
        base_dir: str = None,
        model_name: str = "google/flan-t5-base",
        top_k: int = 5,
        max_evidence_length: int = 1024,
        device: str = None,
    ):
        """
        Initialize BM25+T5 baseline

        Args:
            base_dir: Base directory (defaults to /scratch/$USER/finverify)
            model_name: T5 model name
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
        self.index_dir = base_dir / "data" / "indexes" / "bm25"
        self.top_k = top_k
        self.max_evidence_length = max_evidence_length

        # Setup device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load BM25 index
        print("\nLoading BM25 index...")
        self._load_bm25_index()

        # Load T5 model
        print(f"\nLoading {model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("✓ Model loaded successfully")

    def _load_bm25_index(self):
        """Load BM25 index and chunks"""
        with open(self.index_dir / "bm25_index.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        with open(self.index_dir / "chunk_ids.pkl", "rb") as f:
            self.chunk_ids = pickle.load(f)

        with open(self.index_dir / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        print(f"✓ Loaded BM25 index: {len(self.chunks):,} chunks")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve top-K chunks using BM25

        Args:
            query: Query string
            top_k: Number of chunks to retrieve (default: self.top_k)

        Returns:
            List of chunks with scores
        """
        if top_k is None:
            top_k = self.top_k

        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(query_tokens)

        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            chunk["rank"] = len(results) + 1
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

        evidence_texts = []
        for i, chunk in enumerate(evidence, 1):
            text = chunk["text"].strip()
            source = chunk.get("source", "unknown")
            evidence_texts.append(f"[{i}] {text} (Source: {source})")

        evidence_str = "\n\n".join(evidence_texts)

        if len(evidence_str) > self.max_evidence_length * 4:  # Rough char estimate
            evidence_str = evidence_str[: self.max_evidence_length * 4] + "..."

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

        inputs = self.tokenizer(
            prompt,
            max_length=self.max_evidence_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

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


def evaluate_predictions(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """
    Evaluate predictions against ground truth

    Note: This is a simplified evaluation. For full evaluation,
    you should use proper metrics like F1, EM, etc.
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

    metrics["avg_retrieve_time_ms"] /= len(predictions)
    metrics["avg_generate_time_ms"] /= len(predictions)
    metrics["avg_total_time_ms"] /= len(predictions)

    # TODO: Add F1, EM metrics when ground truth is available
    # This would require implementing string matching and F1 calculation

    return metrics


def test_mode():
    """Test on sample questions"""
    print("=" * 70)
    print("BM25+T5 BASELINE - TEST MODE")
    print("=" * 70)

    # Initialize model
    baseline = BM25T5Baseline(top_k=3)

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
            print(f"  [{j}] Score: {ev['score']:.2f} | Source: {ev['source']}")
            print(f"      {ev['text']}")

    print("\n" + "=" * 70)
    print("✓ Test complete!")
    print("=" * 70)


def eval_mode(num_questions: int = 50):
    """Evaluate on FinanceBench dataset"""
    print("=" * 70)
    print(f"BM25+T5 BASELINE - EVALUATION MODE ({num_questions} questions)")
    print("=" * 70)

    # Initialize model
    baseline = BM25T5Baseline()

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

    # Evaluate
    print("\nEvaluating results...")
    metrics = evaluate_predictions(predictions, questions)

    # Print metrics
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Questions: {metrics['num_questions']}")
    print(f"Avg Retrieval Time: {metrics['avg_retrieve_time_ms']:.1f}ms")
    print(f"Avg Generation Time: {metrics['avg_generate_time_ms']:.1f}ms")
    print(f"Avg Total Time: {metrics['avg_total_time_ms']:.1f}ms")
    print("=" * 70)

    # Save results
    output_dir = baseline.base_dir / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "bm25_t5_results.json"
    with open(results_file, "w") as f:
        json.dump({"metrics": metrics, "predictions": predictions}, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="BM25+T5 Baseline")
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
        print("  python bm25_t5.py --test-mode          # Test on 5 sample questions")
        print("  python bm25_t5.py --eval               # Evaluate on 50 questions")
        print(
            "  python bm25_t5.py --eval --num-questions 100  # Evaluate on 100 questions"
        )


if __name__ == "__main__":
    main()
