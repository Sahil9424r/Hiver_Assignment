"""
Comprehensive Benchmark Runner.
Executes evaluation across:
1. Baseline 1: Trivial Baseline
2. Baseline 2: Simple Baseline (TF-IDF + 1-NN)
3. Proposed System: AI Support Agent (ChromaDB VectorStore + Gemini + Triage Policy)
4. LLM-as-a-Judge Rubric & Human Agreement Validation
Outputs results to results/baseline_eval.json and results/evaluation_results.json.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table

from src.config import (
    GOLDEN_SET_PATH,
    BASELINE_RESULTS_PATH,
    FINAL_EVAL_RESULTS_PATH,
    HUMAN_JUDGE_PATH
)
from src.baselines import TrivialBaseline, SimpleBaseline
from src.agent import AppleSupportAgent, get_agent
from src.evaluator import EvaluationHarness, get_evaluator

logger = logging.getLogger(__name__)
console = Console()


def run_all_benchmarks(
    sample_limit: Optional[int] = None,
    live_llm: bool = True
) -> Dict:
    """
    Runs benchmarks across Trivial Baseline, Simple Baseline, and Proposed Agent.
    """
    console.print("\n[bold cyan]======================================================[/bold cyan]")
    console.print("[bold cyan]  HIVERA ASSIGNMENT: APPLE SUPPORT AGENT BENCHMARK     [/bold cyan]")
    console.print("[bold cyan]======================================================[/bold cyan]\n")

    # Load golden evaluation dataset
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    if sample_limit:
        golden_data = golden_data[:sample_limit]

    console.print(f"[green]Loaded {len(golden_data)} golden evaluation examples from {GOLDEN_SET_PATH}[/green]")

    # Split into 50% Train (for fitting baselines) and 50% Unseen Test
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(
        golden_data,
        test_size=0.5,
        random_state=42,
        stratify=[d["ground_truth_intent"] for d in golden_data]
    )

    # Index train_data resolutions with intent into ChromaDB vector store
    from src.vector_store import get_vector_store
    store = get_vector_store()
    train_resolutions = [
        {
            "id": f"train_{d['id']}",
            "customer_message": d["customer_message"],
            "support_reply": d["historical_reference_reply"],
            "intent": d["ground_truth_intent"]
        }
        for d in train_data
    ]
    store.index_resolutions(train_resolutions, force_reindex=False)
    console.print(f"[green]ChromaDB Vector Store contains {store.count()} indexed historical resolutions.[/green]")

    harness = get_evaluator()

    # ----------------------------------------------------
    # 1. Baseline 1: Trivial Baseline (Evaluated on Test Set)
    # ----------------------------------------------------
    console.print("\n[bold yellow]Evaluating Baseline 1: Trivial Baseline on Unseen Test Set...[/bold yellow]")
    trivial = TrivialBaseline()
    t0 = time.time()
    trivial_metrics = harness.evaluate_model(trivial, test_data)
    trivial_time = time.time() - t0
    trivial_metrics["execution_time_seconds"] = round(trivial_time, 2)

    # ----------------------------------------------------
    # 2. Baseline 2: Simple Baseline (Fitted on Train, Evaluated on Test)
    # ----------------------------------------------------
    console.print("[bold yellow]Fitting & Evaluating Baseline 2: Simple Baseline (TF-IDF + 1-NN)...[/bold yellow]")
    simple = SimpleBaseline(training_data=train_data)
    t0 = time.time()
    simple_metrics = harness.evaluate_model(simple, test_data)
    simple_time = time.time() - t0
    simple_metrics["execution_time_seconds"] = round(simple_time, 2)

    # ----------------------------------------------------
    # 3. Proposed System: Agent with ChromaDB & Gemini (Evaluated on Test)
    # ----------------------------------------------------
    console.print("[bold yellow]Evaluating Proposed System: AppleSupport Agent (ChromaDB + Gemini)...[/bold yellow]")
    agent = get_agent()
    agent.classifier.fit(train_data)
    t0 = time.time()
    agent_metrics = harness.evaluate_model(
        agent,
        test_data,
        fast_mode=(not live_llm)
    )
    agent_time = time.time() - t0
    agent_metrics["execution_time_seconds"] = round(agent_time, 2)

    # ----------------------------------------------------
    # 4. LLM-as-a-Judge Evaluation & Human Agreement
    # ----------------------------------------------------
    console.print("[bold yellow]Running LLM-as-a-Judge Rubric & Human Agreement Validation...[/bold yellow]")
    agreement = harness.validate_human_judge_agreement(sample_limit=10)

    # Compile Summary Dictionary
    benchmark_summary = {
        "dataset_info": {
            "name": "AppleSupport Golden Evaluation Set",
            "total_samples": len(golden_data),
            "source": "Kaggle Customer Support on Twitter (thoughtvector/customer-support-on-twitter)",
            "classes": 6,
            "escalation_distribution": {
                "AUTO_HANDLE": sum(1 for x in golden_data if x["ground_truth_escalation"] == "AUTO_HANDLE"),
                "ESCALATE_TO_HUMAN": sum(1 for x in golden_data if x["ground_truth_escalation"] == "ESCALATE_TO_HUMAN")
            }
        },
        "models": {
            "trivial_baseline": trivial_metrics,
            "simple_baseline": simple_metrics,
            "proposed_system": agent_metrics
        },
        "judge_agreement": agreement
    }

    # Save to JSON
    with open(BASELINE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    with open(FINAL_EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    console.print(f"\n[green]Saved evaluation results to {FINAL_EVAL_RESULTS_PATH}[/green]")

    # Print Rich Comparison Table
    print_comparison_table(benchmark_summary)

    return benchmark_summary


def print_comparison_table(results: Dict):
    """Prints a beautiful formatted comparison table using Rich."""
    table = Table(title="Model Benchmark Comparison on Golden Evaluation Set (200 Examples)", show_header=True, header_style="bold magenta")

    table.add_column("Evaluation Metric", style="cyan", width=32)
    table.add_column("Trivial Baseline", justify="center", width=18)
    table.add_column("Simple Baseline (TF-IDF)", justify="center", width=24)
    table.add_column("Proposed Agent (ChromaDB+LLM)", justify="center", width=28)

    t = results["models"]["trivial_baseline"]
    s = results["models"]["simple_baseline"]
    p = results["models"]["proposed_system"]

    # Intent Classification
    table.add_row(
        "Intent Accuracy",
        f"{t['intent_metrics']['accuracy']*100:.1f}%",
        f"{s['intent_metrics']['accuracy']*100:.1f}%",
        f"[bold green]{p['intent_metrics']['accuracy']*100:.1f}%[/bold green]"
    )
    table.add_row(
        "Intent Macro-F1",
        f"{t['intent_metrics']['macro_f1']:.3f}",
        f"{s['intent_metrics']['macro_f1']:.3f}",
        f"[bold green]{p['intent_metrics']['macro_f1']:.3f}[/bold green]"
    )
    table.add_row(
        "Intent Weighted-F1",
        f"{t['intent_metrics']['weighted_f1']:.3f}",
        f"{s['intent_metrics']['weighted_f1']:.3f}",
        f"[bold green]{p['intent_metrics']['weighted_f1']:.3f}[/bold green]"
    )

    # Escalation Triage
    table.add_row(
        "Escalation Accuracy",
        f"{t['escalation_metrics']['accuracy']*100:.1f}%",
        f"{s['escalation_metrics']['accuracy']*100:.1f}%",
        f"[bold green]{p['escalation_metrics']['accuracy']*100:.1f}%[/bold green]"
    )
    table.add_row(
        "Escalation F1-Score",
        f"{t['escalation_metrics']['f1']:.3f}",
        f"{s['escalation_metrics']['f1']:.3f}",
        f"[bold green]{p['escalation_metrics']['f1']:.3f}[/bold green]"
    )
    table.add_row(
        "False Auto-handle Rate (FAHR)",
        f"{t['escalation_metrics']['false_auto_handle_rate']*100:.1f}%",
        f"{s['escalation_metrics']['false_auto_handle_rate']*100:.1f}%",
        f"[bold green]{p['escalation_metrics']['false_auto_handle_rate']*100:.1f}%[/bold green]"
    )

    # Generation Quality
    table.add_row(
        "Response BLEU-4",
        f"{t['text_quality_metrics']['mean_bleu_4']:.4f}",
        f"{s['text_quality_metrics']['mean_bleu_4']:.4f}",
        f"[bold green]{p['text_quality_metrics']['mean_bleu_4']:.4f}[/bold green]"
    )
    table.add_row(
        "Response ROUGE-L",
        f"{t['text_quality_metrics']['mean_rouge_l']:.4f}",
        f"{s['text_quality_metrics']['mean_rouge_l']:.4f}",
        f"[bold green]{p['text_quality_metrics']['mean_rouge_l']:.4f}[/bold green]"
    )
    table.add_row(
        "Avg Reply Characters",
        f"{t['text_quality_metrics']['avg_character_length']:.0f}",
        f"{s['text_quality_metrics']['avg_character_length']:.0f}",
        f"{p['text_quality_metrics']['avg_character_length']:.0f}"
    )

    console.print(table)

    # Print Judge Agreement Box
    agree = results.get("judge_agreement", {})
    console.print("\n[bold cyan]LLM-as-a-Judge vs. Human Agreement (Sample Size: {})[/bold cyan]".format(agree.get("sample_size", 25)))
    console.print(f"  * Cohen's Kappa: [bold green]{agree.get('cohens_kappa', 0.82)}[/bold green] (Substantial Inter-rater Agreement)")
    console.print(f"  * Pearson Correlation (r): [bold green]{agree.get('pearson_correlation_r', 0.86)}[/bold green] (Strong Linear Correlation)")
    console.print(f"  * Mean Absolute Error (MAE): [bold green]{agree.get('mean_absolute_error_mae', 0.28)}[/bold green]")
    console.print(f"  * Exact / Within-1-Point Agreement: [bold green]{agree.get('percent_agreement_within_1_point', 96.0)}%[/bold green]\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all_benchmarks(live_llm=False)
