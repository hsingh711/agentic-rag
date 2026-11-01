"""Section 5: build the query pool and label retrieve/skip by experiment.

Needs: Qdrant server (scripts/start_qdrant.ps1) with papers_fp32 built, Ollama running,
GOOGLE_API_KEY in .env (for question generation).

Run from the project root:
    python scripts/make_labels.py --estimate-only   # print calls/time, do nothing
    python scripts/make_labels.py --limit 50        # pilot: label only 50 queries
    python scripts/make_labels.py                   # full run (resumes where it stopped)
    python scripts/make_labels.py --report-only     # balance + 10 random samples
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.make_labels import estimate, report, rescore, run_labelling  # noqa: E402
from src.classifier.query_pool import build_hard_examples, build_pool, load_state  # noqa: E402
from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.ingest.io_utils import read_jsonl  # noqa: E402
from src.llm import DailyLimitReached  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description="Generate retrieve/skip labels by experiment.")
    parser.add_argument("--estimate-only", action="store_true", help="Print the estimate and exit.")
    parser.add_argument("--pool-only", action="store_true", hekflp="Build the query pool, do not label.")
    parser.add_argument(
        "--hard-examples",
        action="store_true",
        help="Also add concept questions and plain-language chunk questions (labels.hard_examples).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Label at most this many new queries.")
    parser.add_argument("--report-only", action="store_true", help="Print the label report and exit.")
    parser.add_argument(
        "--rescore", action="store_true", help="Re-score stored answers with the current scoring rules (no LLM calls)."
    )
    return parser.parse_args()


def main() -> None:
    """Estimate, build the pool, label, report."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy_logger in ["httpx", "google_genai", "urllib3", "datasets", "huggingface_hub"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    config = load_config()
    labels_config = config["labels"]
    labels_dir = PROJECT_ROOT / labels_config["dir"]
    pool_path = labels_dir / labels_config["pool_file"]
    state_path = labels_dir / labels_config["generation_state_file"]
    details_path = labels_dir / labels_config["details_file"]
    csv_path = labels_dir / labels_config["output_csv"]

    if args.rescore:
        chunks_file = PROJECT_ROOT / config["paths"]["processed_dir"] / config["ingest"]["chunks_file"]
        chunk_text_by_id = {chunk["chunk_id"]: chunk["text"] for chunk in read_jsonl(chunks_file)}
        changes = rescore(pool_path, details_path, csv_path, labels_config["f1_correct_threshold"], chunk_text_by_id)
        print(f"Re-scored with F1 threshold {labels_config['f1_correct_threshold']}. Label changes (old -> new):")
        for (old_label, new_label), count in sorted(changes.items()):
            print(f"  {old_label} -> {new_label}: {count}")

    if args.report_only or args.rescore:
        print(report(details_path))
        return

    chunks_path = PROJECT_ROOT / config["paths"]["processed_dir"] / config["ingest"]["chunks_file"]
    chunks = list(read_jsonl(chunks_path))
    if labels_config["max_chunks"]:
        chunks = chunks[: labels_config["max_chunks"]]

    print(estimate(config, chunks, pool_path, load_state(state_path), details_path))
    if args.estimate_only:
        return

    try:
        pool_size = build_pool(config, chunks, pool_path, state_path)
        if args.hard_examples:
            pool_size = build_hard_examples(config, chunks, pool_path, state_path)
        print(f"Query pool: {pool_size} queries -> {pool_path}")
        if args.pool_only:
            return

        # Imported here so --estimate-only / --pool-only do not load the embedding model.
        from src.index.embedder import Embedder
        from src.index.qdrant_store import QdrantStore, make_client

        vector_config = config["vector_store"]
        store = QdrantStore(
            make_client(vector_config), labels_config["collection"], Embedder.from_config(config), vector_config, False
        )
        run_labelling(config, store, pool_path, details_path, csv_path, limit=args.limit)
    except DailyLimitReached as error:
        print(f"\nStopped: {error}\nProgress is saved. Re-run after midnight Pacific Time to continue.")

    print()
    print(report(details_path))
    print(f"\nTraining CSV: {csv_path}")


if __name__ == "__main__":
    main()
