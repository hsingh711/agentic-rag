"""Section 11: run the pipeline ablations on the arXiv evaluation questions.

Needs Qdrant and Ollama running, and uses Gemini lite quota for the reformulator and critic
(resumable: finished questions are skipped and identical LLM calls are served from the cache).

    python scripts/run_ablations.py --estimate-only        # print the call budget and stop
    python scripts/run_ablations.py --limit 5              # pilot: 5 questions per configuration
    python scripts/run_ablations.py                         # all configurations
    python scripts/run_ablations.py --only int8 top_k_10
    python scripts/run_ablations.py --summary-only
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
from src.eval.ablations import (  # noqa: E402
    ABLATIONS, accept_all_judge, format_table, identity_reformulate, no_decompose_reformulate,
    run_ablation, save_ablation_summary, save_table, summarize_ablations,
)
from src.eval.harness import load_eval_set, run_agentic  # noqa: E402
from src.generation.grounded import short_cited_generate  # noqa: E402
from src.generation.verifier import NLIScorer, verify_answer  # noqa: E402
from src.graph.pipeline import build_graph  # noqa: E402
from src.index.embedder import Embedder  # noqa: E402
from src.index.qdrant_store import QdrantStore, make_client  # noqa: E402
from src.llm import DailyLimitReached  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description="Run the section 11 ablations.")
    parser.add_argument("--dataset", default="arxiv", help="Evaluation set to ablate on (default arxiv).")
    parser.add_argument("--only", nargs="+", choices=list(ABLATIONS), default=list(ABLATIONS))
    parser.add_argument("--limit", type=int, default=None, help="Override ablations.questions (new questions per configuration).")
    parser.add_argument("--estimate-only", action="store_true", help="Print the lite-call budget and stop.")
    parser.add_argument("--summary-only", action="store_true", help="Summarise existing results, run nothing.")
    return parser.parse_args()


def estimate(names, questions: int, config) -> None:
    """Print a rough upper bound of Gemini lite calls (the cache makes repeats free)."""
    max_retries = config["critic"]["max_retries"]
    print(f"Call budget (worst case, nothing cached), {questions} questions per configuration:")
    total = 0
    for name in names:
        overrides = ABLATIONS[name]["overrides"]
        retries = overrides.get("max_retries", max_retries)
        reformulations = 0 if overrides.get("reformulate") == "identity" else 1 + retries
        critiques = 0 if overrides.get("judge") == "accept_all" else 1 + retries
        calls = (reformulations + critiques) * questions
        total += calls
        print(f"  {name:18s} <= {calls:5d} lite calls")
    print(f"  {'TOTAL':18s} <= {total:5d} (classifier 'skip' questions and the cache cut this a lot)")


def main() -> None:
    """Build one graph per ablation, run it, then print and save the comparison table."""
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config()
    questions_limit = args.limit or config["ablations"]["questions"]

    if args.estimate_only:
        estimate(args.only, questions_limit, config)
        return

    if not args.summary_only:
        embedder = Embedder.from_config(config)
        scorer = NLIScorer.from_config(config)
        verify_fn = lambda answer, contexts, question=None: verify_answer(answer, contexts, scorer, question=question)  # noqa: E731
        vector_config = config["vector_store"]
        client = make_client(vector_config)
        collection = config["graph"]["collection"] if args.dataset == "arxiv" else config["benchmarks"][args.dataset]["collection"]
        stores = {"fp32": QdrantStore(client, collection, embedder, vector_config, use_quantization=False)}
        if args.dataset == "arxiv":
            # Only the arXiv corpus has an int8 collection; the benchmark corpora are fp32 only.
            stores["int8"] = QdrantStore(
                client, vector_config["collection_int8"], embedder, vector_config, use_quantization=True
            )
        elif "int8" in args.only:
            print(f"  (skipping the int8 ablation: {args.dataset} has no int8 collection)")
            args.only = [name for name in args.only if name != "int8"]
        # Slice here, instead of passing a limit, so every configuration runs the SAME questions
        # however often the sweep is restarted (the limit would mean "this many NEW ones").
        all_questions = load_eval_set(args.dataset, config)
        questions = all_questions[:questions_limit]
        print(f"{args.dataset}: {len(all_questions)} questions available, running the first {len(questions)} per configuration")
        estimate(args.only, questions_limit, config)

        try:
            for name in args.only:
                overrides = dict(ABLATIONS[name]["overrides"])
                store = stores[overrides.pop("collection", "fp32")]
                reformulate_mode = overrides.pop("reformulate", None)
                if reformulate_mode == "identity":
                    reformulate_fn = identity_reformulate
                elif reformulate_mode == "no_decompose":
                    reformulate_fn = no_decompose_reformulate(reformulate)
                else:
                    reformulate_fn = reformulate
                judge_fn = accept_all_judge if overrides.pop("judge", None) == "accept_all" else judge

                app = build_graph(
                    store, classifier=should_retrieve, reformulate_fn=reformulate_fn, judge_fn=judge_fn,
                    generate_fn=short_cited_generate, verify_fn=verify_fn, **overrides,
                )
                new = run_ablation(config, args.dataset, name, questions, run_agentic, {"app": app})
                print(f"  {name}: {new} new questions ({ABLATIONS[name]['description']})")
        except DailyLimitReached as error:
            print(f"\nStopped: {error}\nProgress is saved; re-run after midnight Pacific Time to continue.")

    summary = summarize_ablations(config, args.dataset, list(ABLATIONS))
    table = format_table(summary, list(ABLATIONS))
    print(f"\n== {args.dataset} ablations\n{table}")
    print(f"\nSaved {save_ablation_summary(config, summary)}")
    print(f"Saved {save_table(config, table)}")


if __name__ == "__main__":
    main()
