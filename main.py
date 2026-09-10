"""
AppleSupport AI Agent - Unified Command Line Interface.
Supports:
  - run: Process sample or custom queries through the end-to-end pipeline
  - eval: Run evaluation benchmark across baselines and proposed model
  - chat: Interactive real-time support session
  - demo: Side-by-side comparison against baselines
"""

import argparse
import json
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.config import (
    FINAL_EVAL_RESULTS_PATH,
    GOLDEN_SET_PATH,
    GEMINI_MODEL
)
from src.agent import get_agent
from src.baselines import TrivialBaseline, SimpleBaseline
from src.run_benchmarks import run_all_benchmarks, print_comparison_table

console = Console()


def run_pipeline_cmd(args):
    """Executes the pipeline on sample queries or a single user query."""
    agent = get_agent()

    queries = []
    if args.query:
        queries = [args.query]
    else:
        queries = [
            "My iPhone 14 battery dropped 25% in 30 minutes and gets scorching hot.",
            "Someone in Russia logged into my Apple ID account, please help!",
            "How do I set up Face ID with a medical face mask?",
            "Apple charged my credit card $79.99 for ITUNES.COM/BILL and I want a refund.",
            "I waited 2 hours at Regent Street Apple Store and the staff was extremely rude."
        ]

    console.print("\n[bold cyan]=== AppleSupport AI Agent: Processing Inbound Queries ===[/bold cyan]\n")

    for i, q in enumerate(queries, 1):
        console.print(f"[bold yellow]Query {i}:[/bold yellow] \"{q}\"")
        out = agent.process(q, fast_mode=args.fast)

        # Color-coded action
        action_style = "bold red" if out.escalation_action == "ESCALATE_TO_HUMAN" else "bold green"

        table = Table(show_header=False, box=None)
        table.add_row("[bold]Detected Intent:[/bold]", f"{out.intent} (Confidence: {out.intent_confidence:.2f})")
        table.add_row("[bold]Intent Rationale:[/bold]", f"{out.intent_reasoning}")
        table.add_row("[bold]Triage Action:[/bold]", f"[{action_style}]{out.escalation_action}[/{action_style}] (Confidence: {out.escalation_confidence:.2f})")
        table.add_row("[bold]Escalation Reason:[/bold]", f"{out.escalation_reason}")
        table.add_row("[bold]Drafted Reply:[/bold]", f"[bold white]{out.drafted_reply}[/bold white]")

        if out.retrieved_resolutions:
            top_ev = out.retrieved_resolutions[0]
            table.add_row(
                "[bold]ChromaDB Grounding:[/bold]",
                f"[dim]Matched: \"{top_ev['matched_customer_query'][:60]}...\" (Sim: {top_ev['similarity_score']})[/dim]"
            )

        console.print(Panel(table, border_style="cyan"))
        console.print()


def eval_cmd(args):
    """Executes evaluation benchmarks and displays headline results."""
    if args.cached and FINAL_EVAL_RESULTS_PATH.exists():
        console.print(f"[green]Loading precomputed headline evaluation results from {FINAL_EVAL_RESULTS_PATH}...[/green]")
        with open(FINAL_EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print_comparison_table(data)
    else:
        run_all_benchmarks(sample_limit=args.sample_size, live_llm=args.live)


def chat_cmd(args):
    """Launches interactive chat loop."""
    agent = get_agent()
    console.print("\n[bold green]====================================================[/bold green]")
    console.print("[bold green]  Interactive AppleSupport AI Agent CLI            [/bold green]")
    console.print("[dim]Type your customer support tweet below. Type 'exit' to quit.[/dim]")
    console.print("[bold green]====================================================[/bold green]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]Customer Tweet > [/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            out = agent.process(user_input, fast_mode=args.fast)
            action_style = "bold red" if out.escalation_action == "ESCALATE_TO_HUMAN" else "bold green"

            console.print(f"\n[bold]Intent:[/bold] {out.intent} ({out.intent_confidence}) | [bold]Triage:[/bold] [{action_style}]{out.escalation_action}[/{action_style}]")
            console.print(f"[bold]Triage Reason:[/bold] {out.escalation_reason}")
            console.print(f"[bold]AppleSupport Draft:[/bold] [bold white]{out.drafted_reply}[/bold white]\n")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting...[/yellow]")
            break


def demo_cmd(args):
    """Compares Trivial Baseline vs Simple Baseline vs Proposed Agent."""
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    trivial = TrivialBaseline()
    simple = SimpleBaseline(training_data=golden_data[:100])
    agent = get_agent()

    query = args.query or "My iPad screen is physically bulging outward and popping out of the chassis!"
    console.print(f"\n[bold yellow]Comparing Models on Query:[/bold yellow] \"{query}\"\n")

    t_out = trivial.process(query)
    s_out = simple.process(query)
    p_out = agent.process(query, fast_mode=args.fast)

    comp_table = Table(title="Live Multi-Model Comparison", show_header=True, header_style="bold magenta")
    comp_table.add_column("Dimension", style="cyan", width=22)
    comp_table.add_column("Trivial Baseline", width=26)
    comp_table.add_column("Simple Baseline (TF-IDF)", width=28)
    comp_table.add_column("Proposed Agent (ChromaDB+LLM)", width=35)

    comp_table.add_row("Predicted Intent", t_out["intent"], s_out["intent"], f"[bold green]{p_out.intent}[/bold green]")
    comp_table.add_row("Escalation Decision", t_out["escalation_action"], s_out["escalation_action"], f"[bold green]{p_out.escalation_action}[/bold green]")
    comp_table.add_row("Escalation Reason", t_out["escalation_reason"][:60]+"...", s_out["escalation_reason"], p_out.escalation_reason)
    comp_table.add_row("Drafted Reply", t_out["drafted_reply"], s_out["drafted_reply"], f"[bold white]{p_out.drafted_reply}[/bold white]")

    console.print(comp_table)
    console.print()


def main():
    parser = argparse.ArgumentParser(description="AppleSupport AI Agent Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # run command
    run_p = subparsers.add_parser("run", help="Run agent pipeline on sample or custom queries")
    run_p.add_argument("--query", "-q", type=str, help="Single query text to process")
    run_p.add_argument("--fast", action="store_true", help="Use fast local heuristic/vector inference")

    # eval command
    eval_p = subparsers.add_parser("eval", help="Run benchmark evaluation")
    eval_p.add_argument("--cached", action="store_true", default=True, help="Display precomputed headline results instantly")
    eval_p.add_argument("--recompute", action="store_true", help="Re-run the evaluation benchmark suite")
    eval_p.add_argument("--sample-size", type=int, default=None, help="Limit number of samples")
    eval_p.add_argument("--live", action="store_true", help="Use live LLM API calls during benchmark")

    # chat command
    chat_p = subparsers.add_parser("chat", help="Launch interactive support chat")
    chat_p.add_argument("--fast", action="store_true", help="Use fast local heuristic/vector inference")

    # demo command
    demo_p = subparsers.add_parser("demo", help="Side-by-side comparison against baselines")
    demo_p.add_argument("--query", "-q", type=str, help="Query text to compare")
    demo_p.add_argument("--fast", action="store_true", help="Use fast local heuristic/vector inference")

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline_cmd(args)
    elif args.command == "eval":
        if args.recompute:
            args.cached = False
        eval_cmd(args)
    elif args.command == "chat":
        chat_cmd(args)
    elif args.command == "demo":
        demo_cmd(args)
    else:
        # Default to showing evaluation results
        parser.print_help()


if __name__ == "__main__":
    main()
