"""Section 11: compare dense, BM25, hybrid (RRF) and hybrid + cross-encoder reranker retrieval.

No LLM calls. Needs the Qdrant server. The RRF weights are chosen on development questions
(never the evaluation questions); the table is on the evaluation questions at top-k = retrieval.top_k.

    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --datasets hotpotqa multihoprag
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.eval.benchmarks import benchmark_paths  # noqa: E402
from src.eval.retrieval_eval import evaluate_retriever, load_questions  # noqa: E402
from src.index.embedder import Embedder  # noqa: E402
from src.index.hybrid import BM25Index, CrossEncoderReranker, HybridStore  # noqa: E402
from src.index.qdrant_store import QdrantStore, make_client  # noqa: E402
from src.ingest.io_utils import read_jsonl  # noqa: E402


def chunk_file(dataset: str, config: dict) -> Path:
    """Chunks that were indexed for a dataset."""
    if dataset == "arxiv":
        return PROJECT_ROOT / config["paths"]["processed_dir"] / config["ingest"]["chunks_file"]
    return benchmark_paths(dataset)[1]


def main() -> None:
    """Pick RRF weights on development questions, then report every method on the evaluation questions."""
    parser = argparse.ArgumentParser(description="Compare retrieval methods.")
    parser.add_argument("--datasets", nargs="+", default=["arxiv", "hotpotqa", "multihoprag"])
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config = load_config()
    settings = config["hybrid"]
    top_k = config["retrieval"]["top_k"]
    vector_config = config["vector_store"]
    embedder = Embedder.from_config(config)
    reranker = CrossEncoderReranker(settings["reranker_model"])
    report = {}

    for dataset in args.datasets:
        collection = config["graph"]["collection"] if dataset == "arxiv" else config["benchmarks"][dataset]["collection"]
        dense = QdrantStore(make_client(vector_config), collection, embedder, vector_config, False)
        chunks = list(read_jsonl(chunk_file(dataset, config)))
        bm25 = BM25Index(chunks, settings["bm25_k1"], settings["bm25_b"])
        dev = load_questions(dataset, config, "dev")
        test = load_questions(dataset, config, "eval")

        # Choose the fusion weights on the development questions only.
        best_weights, best_score = None, -1.0
        for weights in settings["weight_grid"]:
            hybrid = HybridStore(dense, bm25, settings["candidates"], weights, settings["rrf_k"])
            score = evaluate_retriever(hybrid.search, dev, dataset, top_k)["gold_recall"]
            print(f"{dataset} dev: weights {weights} gold_recall {score:.3f} ({len(dev)} questions)")
            if score > best_score:
                best_weights, best_score = weights, score

        methods = {
            "dense": dense.search,
            "bm25": lambda query, k: bm25.search(query, k),
            "hybrid": HybridStore(dense, bm25, settings["candidates"], best_weights, settings["rrf_k"]).search,
            "hybrid_rerank": HybridStore(
                dense, bm25, settings["candidates"], best_weights, settings["rrf_k"], reranker, settings["rerank_candidates"]
            ).search,
        }
        report[dataset] = {"weights": best_weights, "dev_n": len(dev), "eval_n": len(test), "methods": {}}
        print(f"\n== {dataset} (eval n={len(test)}, weights {best_weights}, top-{top_k})")
        print(f"{'method':15s} {'hit':>6s} {'gold_rec':>9s} {'all_gold':>9s} {'ms':>7s}")
        for name, search in methods.items():
            result = evaluate_retriever(search, test, dataset, top_k)
            report[dataset]["methods"][name] = result
            print(f"{name:15s} {result['hit']:6.3f} {result['gold_recall']:9.3f} {result['all_gold']:9.3f} {result['median_ms']:7.1f}")

    output = PROJECT_ROOT / settings["output"]
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
