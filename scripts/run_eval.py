"""Section 10: run closed-book, vanilla RAG and the agentic pipeline on the evaluation sets.

Needs Qdrant and Ollama running; the agentic systems also use Gemini lite quota (resumable:
finished questions are skipped and LLM replies are cached). Stop Ollama-free GPU work first.

    python scripts/run_eval.py --limit 5                       # pilot
    python scripts/run_eval.py                                  # everything
    python scripts/run_eval.py --datasets hotpotqa --systems vanilla agentic
    python scripts/run_eval.py --summary-only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src.agents.critic import judge  # noqa: E402
from src.agents.reformulator import reformulate  # noqa: E402
from src.classifier.predict import should_retrieve  # noqa: E402
from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.eval.harness import (  # noqa: E402
    DATASETS, SYSTEMS, evaluate, load_eval_set, run_agentic, run_closed_book, run_vanilla, save_summary, summarize_all,
)
from src.generation.grounded import grounded_generate, short_cited_generate  # noqa: E402
from src.generation.verifier import NLIScorer, verify_answer  # noqa: E402
from src.graph.pipeline import build_graph  # noqa: E402
from src.index.embedder import Embedder  # noqa: E402
from src.index.qdrant_store import QdrantStore, make_client  # noqa: E402
from src.llm import DailyLimitReached  # noqa: E402


def hybrid_store(dataset: str, dense_store, config: dict, rerank: bool):
    """Dense + BM25 (+ reranker) over the same chunks, with the fusion weights chosen by scripts/eval_retrieval.py."""
    import json

    from src.eval.benchmarks import benchmark_paths
    from src.index.hybrid import HybridStore
    from src.ingest.io_utils import read_jsonl

    chunks_path = (
        PROJECT_ROOT / config["paths"]["processed_dir"] / config["ingest"]["chunks_file"]
        if dataset == "arxiv" else benchmark_paths(dataset)[1]
    )
    store = HybridStore.from_config(dense_store, list(read_jsonl(chunks_path)), rerank, config)
    chosen = PROJECT_ROOT / config["hybrid"]["output"]
    if chosen.is_file():
        store.weights = json.loads(chosen.read_text(encoding="utf-8")).get(dataset, {}).get("weights", store.weights)
    return store


def print_summary(summary: dict) -> None:
    """One table per dataset (datasets are never averaged together)."""
    columns = [
        ("relaxed_accuracy", "relaxed"), ("strict_accuracy", "strict"), ("skip_rate", "skip"),
        ("false_abstention_rate", "false_abst"), ("null_abstention_rate", "null_abst"),
        ("gold_title_recall", "gold_rec"), ("unsupported_claim_rate", "unsupp"),
        ("mean_model_seconds", "model_s"), ("mean_wait_seconds", "wait_s"), ("mean_lite_calls", "lite"),
    ]
    for dataset, systems in summary.items():
        print(f"\n== {dataset}")
        print(f"{'system':24s} {'n':>4s} " + " ".join(f"{label:>10s}" for _, label in columns))
        for system, values in systems.items():
            cells = []
            for key, _ in columns:
                value = values[key]
                cells.append(f"{'-':>10s}" if value is None else f"{value:10.3f}")
            paired = values.get("paired_vs_vanilla")
            note = ""
            if paired and paired["n"]:
                note = f"   vs vanilla {paired['difference']:+.3f} [{paired['ci95'][0]:+.3f}, {paired['ci95'][1]:+.3f}] W{paired['wins']}/L{paired['losses']}"
            print(f"{system:24s} {values['n']:4d} " + " ".join(cells) + note)


def main() -> None:
    """Build every system for every dataset and evaluate."""
    parser = argparse.ArgumentParser(description="Run the section 10 evaluation.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--limit", type=int, default=None, help="At most this many NEW questions per dataset and system.")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--reverify", action="store_true", help="Recompute claim counts of stored answers with the current verifier (no LLM calls).")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config()

    if args.reverify:
        from src.eval.harness import reverify

        scorer = NLIScorer.from_config(config)
        for dataset in args.datasets:
            for system in args.systems:
                print(f"reverified {dataset}/{system}: {reverify(config, dataset, system, scorer)} records")
    elif not args.summary_only:
        embedder = Embedder.from_config(config)
        scorer = NLIScorer.from_config(config)
        verify_fn = lambda answer, contexts, question=None: verify_answer(answer, contexts, scorer, question=question)  # noqa: E731
        vector_config = config["vector_store"]
        try:
            for dataset in args.datasets:
                collection = config["graph"]["collection"] if dataset == "arxiv" else config["benchmarks"][dataset]["collection"]
                store = QdrantStore(make_client(vector_config), collection, embedder, vector_config, False)
                questions = load_eval_set(dataset, config)
                print(f"\n{dataset}: {len(questions)} questions, collection {collection}")
                for system in args.systems:
                    if system == "closed_book":
                        runner, runner_args = run_closed_book, {}
                    elif system.startswith("vanilla"):
                        runner = run_vanilla
                        retrieval_store = store
                        if system != "vanilla":
                            retrieval_store = hybrid_store(dataset, store, config, rerank=system.endswith("rerank"))
                        runner_args = {"store": retrieval_store, "top_k": config["retrieval"]["top_k"], "scorer": scorer,
                                       "verifier_settings": config["verifier"]}
                    else:
                        generate_fn = short_cited_generate if system.startswith("agentic_v2") else grounded_generate
                        app = build_graph(
                            store, classifier=should_retrieve, reformulate_fn=reformulate, judge_fn=judge,
                            generate_fn=generate_fn, verify_fn=verify_fn, use_classifier=not system.endswith("no_classifier"),
                        )
                        runner, runner_args = run_agentic, {"app": app}
                    new = evaluate(config, dataset, system, questions, runner, runner_args, args.limit)
                    print(f"  {system}: {new} new questions")
        except DailyLimitReached as error:
            print(f"\nStopped: {error}\nProgress is saved; re-run after midnight Pacific Time to continue.")

    summary = summarize_all(config)
    print_summary(summary)
    print(f"\nSaved {save_summary(config, summary)}")


if __name__ == "__main__":
    main()
