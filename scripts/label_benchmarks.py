"""Section 10a: add retrieve/skip labels from the benchmark questions (never the evaluation questions).

For each benchmark the generator answers closed-book and with the top-k chunks of the
benchmark's OWN collection; the usual rule decides retrieve / skip / dropped. Output goes to
data/labels/necessity_<name>_details.jsonl and necessity_<name>.csv (resumable). No Gemini calls.
Needs Qdrant and Ollama running.

    python scripts/label_benchmarks.py
    python scripts/label_benchmarks.py --limit 20     # pilot
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.make_labels import run_labelling  # noqa: E402
from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.eval.benchmarks import label_pool, write_jsonl  # noqa: E402
from src.index.embedder import Embedder  # noqa: E402
from src.index.qdrant_store import QdrantStore, make_client  # noqa: E402
from src.ingest.io_utils import read_jsonl  # noqa: E402
from src.llm import DailyLimitReached  # noqa: E402


def main() -> None:
    """Label the benchmark questions of both datasets."""
    parser = argparse.ArgumentParser(description="Label benchmark questions for the classifier.")
    parser.add_argument("--limit", type=int, default=None, help="Label at most this many new queries per dataset.")
    args = parser.parse_args()

    config = load_config()
    labels_dir = PROJECT_ROOT / config["labels"]["dir"]
    vector_config = config["vector_store"]
    embedder = Embedder.from_config(config)
    try:
        for name in ("hotpotqa", "multihoprag"):
            pool_path = labels_dir / f"pool_{name}.jsonl"
            details_path = labels_dir / f"necessity_{name}_details.jsonl"
            csv_path = labels_dir / f"necessity_{name}.csv"
            write_jsonl(pool_path, label_pool(name, config))
            store = QdrantStore(
                make_client(vector_config), config["benchmarks"][name]["collection"], embedder, vector_config, False
            )
            run_labelling(config, store, pool_path, details_path, csv_path, limit=args.limit)
            counts = Counter(record["label"] for record in read_jsonl(details_path))
            print(f"{name}: {dict(counts)} -> {csv_path}")
    except DailyLimitReached as error:
        print(f"\nStopped: {error}\nProgress is saved; re-run to continue.")


if __name__ == "__main__":
    main()
