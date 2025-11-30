"""
Compare BM25+T5 vs DPR+T5 baselines

Author: Utkarsh
Date: November 2025
"""

import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(base_dir: Path) -> tuple:
    """Load results from both baselines"""
    results_dir = base_dir / "outputs" / "results"

    bm25_file = results_dir / "bm25_t5_results.json"
    dpr_file = results_dir / "dpr_t5_results.json"

    if not bm25_file.exists():
        raise FileNotFoundError(f"BM25 results not found: {bm25_file}")
    if not dpr_file.exists():
        raise FileNotFoundError(f"DPR results not found: {dpr_file}")

    with open(bm25_file, "r") as f:
        bm25_results = json.load(f)

    with open(dpr_file, "r") as f:
        dpr_results = json.load(f)

    return bm25_results, dpr_results


def compare_metrics(bm25_results: dict, dpr_results: dict) -> dict:
    """Compare metrics between baselines"""
    bm25_metrics = bm25_results["metrics"]
    dpr_metrics = dpr_results["metrics"]

    comparison = {"bm25": bm25_metrics, "dpr": dpr_metrics, "improvements": {}}

    # Calculate improvements
    for metric in ["avg_retrieve_time_ms", "avg_generate_time_ms", "avg_total_time_ms"]:
        bm25_val = bm25_metrics[metric]
        dpr_val = dpr_metrics[metric]
        improvement = ((bm25_val - dpr_val) / bm25_val) * 100
        comparison["improvements"][metric] = improvement

    return comparison


def create_comparison_plots(comparison: dict, output_dir: Path):
    """Create visualization comparing both methods"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["avg_retrieve_time_ms", "avg_generate_time_ms", "avg_total_time_ms"]
    titles = ["Retrieval Time", "Generation Time", "Total Time"]

    for ax, metric, title in zip(axes, metrics, titles):
        bm25_val = comparison["bm25"][metric]
        dpr_val = comparison["dpr"][metric]

        x = ["BM25+T5", "DPR+T5"]
        y = [bm25_val, dpr_val]

        bars = ax.bar(x, y, color=["#3498db", "#e74c3c"])
        ax.set_ylabel("Time (ms)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}ms",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()

    # Save plot
    plot_file = output_dir / "baseline_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"✓ Plot saved to: {plot_file}")

    plt.close()


def create_comparison_table(comparison: dict, output_dir: Path):
    """Create CSV comparison table"""
    csv_file = output_dir / "baseline_comparison.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["Metric", "BM25+T5", "DPR+T5", "Improvement (%)"])

        # Rows
        metrics = [
            ("Retrieval Time (ms)", "avg_retrieve_time_ms"),
            ("Generation Time (ms)", "avg_generate_time_ms"),
            ("Total Time (ms)", "avg_total_time_ms"),
        ]

        for label, metric in metrics:
            bm25_val = comparison["bm25"][metric]
            dpr_val = comparison["dpr"][metric]
            improvement = comparison["improvements"][metric]

            writer.writerow(
                [label, f"{bm25_val:.2f}", f"{dpr_val:.2f}", f"{improvement:+.1f}%"]
            )

    print(f"✓ CSV saved to: {csv_file}")


def create_detailed_results(bm25_results: dict, dpr_results: dict, output_dir: Path):
    """Create detailed per-question comparison"""
    csv_file = output_dir / "detailed_results.csv"

    bm25_preds = bm25_results["predictions"]
    dpr_preds = dpr_results["predictions"]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            [
                "Question",
                "BM25+T5 Answer",
                "DPR+T5 Answer",
                "BM25 Time (ms)",
                "DPR Time (ms)",
                "Faster Method",
            ]
        )

        for bm25_pred, dpr_pred in zip(bm25_preds, dpr_preds):
            question = bm25_pred["question"]
            bm25_answer = bm25_pred["answer"]
            dpr_answer = dpr_pred["answer"]
            bm25_time = bm25_pred["total_time_ms"]
            dpr_time = dpr_pred["total_time_ms"]
            faster = "BM25+T5" if bm25_time < dpr_time else "DPR+T5"

            writer.writerow(
                [
                    question,
                    bm25_answer,
                    dpr_answer,
                    f"{bm25_time:.1f}",
                    f"{dpr_time:.1f}",
                    faster,
                ]
            )

    print(f"✓ Detailed results saved to: {csv_file}")


def print_summary(comparison: dict):
    """Print summary to console"""
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 70)

    print("\nBM25+T5 Performance:")
    print(f"  Retrieval: {comparison['bm25']['avg_retrieve_time_ms']:.1f}ms")
    print(f"  Generation: {comparison['bm25']['avg_generate_time_ms']:.1f}ms")
    print(f"  Total: {comparison['bm25']['avg_total_time_ms']:.1f}ms")

    print("\nDPR+T5 Performance:")
    print(f"  Retrieval: {comparison['dpr']['avg_retrieve_time_ms']:.1f}ms")
    print(f"  Generation: {comparison['dpr']['avg_generate_time_ms']:.1f}ms")
    print(f"  Total: {comparison['dpr']['avg_total_time_ms']:.1f}ms")

    print("\nRelative Performance:")
    for metric, improvement in comparison["improvements"].items():
        label = metric.replace("avg_", "").replace("_ms", "").replace("_", " ").title()
        if improvement > 0:
            print(f"  {label}: BM25+T5 is {improvement:.1f}% faster")
        else:
            print(f"  {label}: DPR+T5 is {abs(improvement):.1f}% faster")

    print("\n" + "=" * 70)

    bm25_total = comparison["bm25"]["avg_total_time_ms"]
    dpr_total = comparison["dpr"]["avg_total_time_ms"]

    if bm25_total < dpr_total:
        winner = "BM25+T5"
        margin = ((dpr_total - bm25_total) / dpr_total) * 100
    else:
        winner = "DPR+T5"
        margin = ((bm25_total - dpr_total) / bm25_total) * 100

    print(f"\n🏆 Winner: {winner} ({margin:.1f}% faster overall)")
    print("=" * 70)


def main():
    import os

    user = os.environ.get("USER", "unknown")
    base_dir = Path("/scratch") / user / "finverify"

    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        print("Please ensure you've copied the data and run both baselines.")
        return

    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    print("\nLoading results...")
    try:
        bm25_results, dpr_results = load_results(base_dir)
        print("✓ Results loaded successfully")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run both baselines first:")
        print("  python src/baselines/bm25_t5.py --eval")
        print("  python src/baselines/dpr_t5.py --eval")
        return

    print("\nComparing metrics...")
    comparison = compare_metrics(bm25_results, dpr_results)

    output_dir = base_dir / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = base_dir / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating comparison plots...")
    create_comparison_plots(comparison, figures_dir)

    print("\nCreating comparison tables...")
    create_comparison_table(comparison, output_dir)

    print("\nCreating detailed results...")
    create_detailed_results(bm25_results, dpr_results, output_dir)

    comparison_file = output_dir / "baseline_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n✓ Comparison saved to: {comparison_file}")

    print_summary(comparison)

    print("\n✓ Comparison complete!")
    print("\nGenerated files:")
    print(f"  - {figures_dir}/baseline_comparison.png")
    print(f"  - {output_dir}/baseline_comparison.json")
    print(f"  - {output_dir}/baseline_comparison.csv")
    print(f"  - {output_dir}/detailed_results.csv")


if __name__ == "__main__":
    main()
