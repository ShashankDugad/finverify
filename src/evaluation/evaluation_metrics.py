"""
Evaluation Metrics for FinVERIFY Baselines
Implements: Exact Match, F1 Score, Recall@K

Author: Utkarsh
Date: November 2025
"""

import re
from typing import List, Dict, Set
from collections import Counter


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison
    
    Steps:
    1. Lowercase
    2. Remove articles (a, an, the)
    3. Remove punctuation
    4. Remove extra whitespace
    """
    # Lowercase
    text = text.lower()
    
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Exact Match (EM) metric
    
    Returns 1.0 if normalized strings match exactly, else 0.0
    
    Args:
        prediction: Model's predicted answer
        ground_truth: Correct answer
        
    Returns:
        1.0 or 0.0
        
    Example:
        prediction = "The revenue was $394.3 billion"
        ground_truth = "revenue was $394.3 billion"
        → normalize both → match! → 1.0
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 Score
    
    Computes overlap between predicted and ground truth tokens
    
    Formula:
        Precision = |pred ∩ truth| / |pred|
        Recall = |pred ∩ truth| / |truth|
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
    Args:
        prediction: Model's predicted answer
        ground_truth: Correct answer
        
    Returns:
        F1 score (0.0 to 1.0)
        
    Example:
        prediction = "revenue was 394 billion"     [revenue, was, 394, billion]
        ground_truth = "revenue 394 billion usd"   [revenue, 394, billion, usd]
        
        Common tokens: {revenue, 394, billion} = 3 tokens
        Precision: 3/4 = 0.75
        Recall: 3/4 = 0.75
        F1: 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75
    """
    # Normalize and tokenize
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    # Handle empty predictions
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(len(pred_tokens) == len(truth_tokens))
    
    # Count token frequencies
    pred_counts = Counter(pred_tokens)
    truth_counts = Counter(truth_tokens)
    
    # Calculate overlap (common tokens with min frequency)
    common = pred_counts & truth_counts  # Intersection of counters
    num_common = sum(common.values())
    
    # Calculate precision and recall
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    
    # Calculate F1
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def recall_at_k(retrieved_docs: List[Dict], ground_truth_docs: List[str], k: int = 10) -> float:
    """
    Recall@K metric for retrieval
    
    Measures what fraction of relevant documents are in top-K results
    
    Formula:
        Recall@K = |relevant docs in top-K| / |total relevant docs|
        
    Args:
        retrieved_docs: List of retrieved documents with 'source' or 'doc_id'
        ground_truth_docs: List of relevant document IDs/sources
        k: Number of top results to consider
        
    Returns:
        Recall score (0.0 to 1.0)
        
    Example:
        retrieved = [doc1, doc2, doc3, doc4, doc5]
        ground_truth = [doc2, doc3, doc7]
        
        In top-5: doc2 ✓, doc3 ✓, doc7 ✗
        Recall@5 = 2/3 = 0.667
    """
    if not ground_truth_docs:
        return 0.0
    
    # Get top-K retrieved document IDs
    retrieved_ids = set()
    for doc in retrieved_docs[:k]:
        # Try different possible ID fields
        doc_id = doc.get('source') or doc.get('doc_id') or doc.get('id') or str(doc)
        retrieved_ids.add(doc_id)
    
    # Count how many ground truth docs are in retrieved set
    ground_truth_set = set(ground_truth_docs)
    num_relevant_retrieved = len(retrieved_ids & ground_truth_set)
    
    # Calculate recall
    recall = num_relevant_retrieved / len(ground_truth_set)
    
    return recall


def evaluate_predictions(
    predictions: List[Dict], 
    ground_truth: List[Dict],
    compute_retrieval_metrics: bool = False
) -> Dict:
    """
    Comprehensive evaluation of predictions
    
    Computes:
    - Exact Match (EM)
    - F1 Score
    - Recall@10 (if retrieval info available)
    - Latency metrics
    
    Args:
        predictions: List of prediction dictionaries
            Format: {'question': str, 'answer': str, 'evidence': [...], ...}
        ground_truth: List of ground truth dictionaries
            Format: {'question': str, 'answer': str, 'evidence_docs': [...]}
        compute_retrieval_metrics: Whether to compute Recall@K
        
    Returns:
        Dictionary with all metrics
    """
    num_questions = len(predictions)
    
    if num_questions == 0:
        return {
            'num_questions': 0,
            'exact_match': 0.0,
            'f1_score': 0.0,
            'recall@10': None,
            'avg_retrieve_time_ms': 0.0,
            'avg_generate_time_ms': 0.0,
            'avg_total_time_ms': 0.0,
            'per_question': []
        }
    
    # Initialize accumulators
    total_em = 0.0
    total_f1 = 0.0
    total_recall_at_10 = 0.0
    total_retrieve_time = 0.0
    total_generate_time = 0.0
    total_time = 0.0
    valid_questions = 0
    
    # Track per-question metrics for analysis
    per_question_metrics = []
    
    for pred, truth in zip(predictions, ground_truth):
        # Get answers
        pred_answer = pred.get('answer', '')
        truth_answer = truth.get('answer', '')
        
        # Skip if no ground truth
        if not truth_answer or truth_answer.lower() in ['unknown', 'n/a', 'none']:
            continue
        
        valid_questions += 1
        
        # Compute answer metrics
        em = exact_match(pred_answer, truth_answer)
        f1 = f1_score(pred_answer, truth_answer)
        
        total_em += em
        total_f1 += f1
        
        # Compute retrieval metrics if requested
        recall_10 = 0.0
        if compute_retrieval_metrics:
            pred_evidence = pred.get('evidence', [])
            truth_evidence = truth.get('evidence_docs', [])
            if pred_evidence and truth_evidence:
                recall_10 = recall_at_k(pred_evidence, truth_evidence, k=10)
                total_recall_at_10 += recall_10
        
        # Accumulate timing
        total_retrieve_time += pred.get('retrieve_time_ms', 0)
        total_generate_time += pred.get('generate_time_ms', 0)
        total_time += pred.get('total_time_ms', 0)
        
        # Store per-question metrics
        per_question_metrics.append({
            'question': pred.get('question', ''),
            'em': em,
            'f1': f1,
            'recall@10': recall_10 if compute_retrieval_metrics else None
        })
    
    # Calculate averages
    if valid_questions == 0:
        valid_questions = 1  # Avoid division by zero
    
    metrics = {
        'num_questions': num_questions,
        'valid_questions': valid_questions,
        
        # Answer quality metrics
        'exact_match': (total_em / valid_questions) * 100,  # Convert to percentage
        'f1_score': (total_f1 / valid_questions) * 100,      # Convert to percentage
        
        # Retrieval quality
        'recall@10': (total_recall_at_10 / valid_questions) * 100 if compute_retrieval_metrics else None,
        
        # Latency metrics
        'avg_retrieve_time_ms': total_retrieve_time / num_questions if num_questions > 0 else 0.0,
        'avg_generate_time_ms': total_generate_time / num_questions if num_questions > 0 else 0.0,
        'avg_total_time_ms': total_time / num_questions if num_questions > 0 else 0.0,
        
        # Per-question breakdown
        'per_question': per_question_metrics
    }
    
    return metrics


def print_metrics(metrics: Dict, baseline_name: str = "Baseline"):
    """
    Pretty print evaluation metrics
    
    Args:
        metrics: Dictionary from evaluate_predictions
        baseline_name: Name to display (e.g., "BM25+T5")
    """
    print("\n" + "=" * 70)
    print(f"{baseline_name} EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\n📊 Answer Quality Metrics:")
    print(f"  Exact Match:  {metrics['exact_match']:.2f}%")
    print(f"  F1 Score:     {metrics['f1_score']:.2f}%")
    
    if metrics['recall@10'] is not None:
        print(f"\n🔍 Retrieval Quality:")
        print(f"  Recall@10:    {metrics['recall@10']:.2f}%")
    
    print(f"\n⚡ Latency Metrics:")
    print(f"  Retrieval:    {metrics['avg_retrieve_time_ms']:.1f}ms")
    print(f"  Generation:   {metrics['avg_generate_time_ms']:.1f}ms")
    print(f"  Total:        {metrics['avg_total_time_ms']:.1f}ms")
    
    print(f"\n📈 Questions Evaluated: {metrics['num_questions']} (Valid: {metrics.get('valid_questions', metrics['num_questions'])})")
    print("=" * 70)

