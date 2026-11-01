"""Embed the chunks, build the fp32 and int8 Qdrant collections, and run a smoke test.

Start the Qdrant server first:  powershell -File scripts/start_qdrant.ps1

Examples (run from the project root):
    python scripts/build_index.py
    python scripts/build_index.py --skip-smoke
    python scripts/build_index.py --chunks data/processed/chunks_scale.jsonl   # large corpus (section 11)
"""

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.index.embedder import Embedder  # noqa: E402
from src.index.qdrant_store import QdrantStore, make_client  # noqa: E402
from src.ingest.io_utils import read_jsonl  # noqa: E402

SMOKE_QUERIES = [
    "retrieval-augmented generation for question answering",
    "dense passage retrieval with contrastive learning",
    "efficient fine-tuning of large language models",
    "reducing hallucination in language model outputs",
    "approximate nearest neighbor search",
]


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description="Build Qdrant collections and run a smoke test.")
    parser.add_argument("--chunks", type=str, default=None, help="Chunks JSONL (default: processed chunks_file).")
    parser.add_argument("--skip-smoke", action="store_true", help="Build only, no smoke test.")
    return parser.parse_args()


def median_latency_ms(store: QdrantStore, query_vector, top_k: int, repeats: int) -> float:
    """Median time of one vector search in milliseconds (after one warm-up call)."""
    store.search_by_vector(query_vector, top_k)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        store.search_by_vector(query_vector, top_k)
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def run_smoke_test(stores, embedder, config) -> str:
    """Run the sample queries on every store; return the report text."""
    top_k = config["retrieval"]["top_k"]
    repeats = config["smoke_test"]["repeats"]
    lines = []
    for query in SMOKE_QUERIES:
        query_vector = embedder.encode_query(query)
        lines.append(f"Query: {query}")
        for name, store in stores.items():
            latency = median_latency_ms(store, query_vector, top_k, repeats)
            hits = store.search_by_vector(query_vector, 3)
            lines.append(f"  [{name}] median latency {latency:.2f} ms over {repeats} runs")
            for rank, hit in enumerate(hits, start=1):
                lines.append(f"    {rank}. ({hit['score']:.3f}) {hit['title']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Build both collections and optionally run the smoke test."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    vector_config = config["vector_store"]

    chunks_path = Path(args.chunks) if args.chunks else (
        PROJECT_ROOT / config["paths"]["processed_dir"] / config["ingest"]["chunks_file"]
    )
    chunks = list(read_jsonl(chunks_path))
    if not chunks:
        raise SystemExit(f"No chunks found in {chunks_path}. Run scripts/run_ingest.py first.")
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    embedder = Embedder.from_config(config)
    start = time.perf_counter()
    vectors = embedder.encode([chunk["text"] for chunk in chunks], show_progress=True)
    print(f"Embedded in {time.perf_counter() - start:.1f}s, vector shape {vectors.shape}")

    client = make_client(vector_config)
    stores = {
        vector_config["collection_fp32"]: QdrantStore(
            client, vector_config["collection_fp32"], embedder, vector_config, use_quantization=False
        ),
        vector_config["collection_int8"]: QdrantStore(
            client, vector_config["collection_int8"], embedder, vector_config,
            use_quantization=vector_config["use_quantization"],
        ),
    }

    report_lines = [f"Qdrant mode: {vector_config['mode']}"]
    for name, store in stores.items():
        build_seconds = store.build_index(chunks, vectors=vectors)
        info = store.describe()
        report_lines.append(
            f"{name}: {info['points']} points, indexed_vectors={info['indexed_vectors']}, "
            f"status={info['status']}, built in {build_seconds:.1f}s, "
            f"quantization={'int8' if info['quantization'] else 'none'}"
        )
    if vector_config["mode"] != "server":
        report_lines.append("WARNING: local mode ignores HNSW and quantization; latencies are NOT meaningful.")
    report_lines.append(f"Corpus is small ({len(chunks)} chunks): fp32/int8 speed differences are expected to be tiny.")
    report_lines.append("")

    if not args.skip_smoke:
        report_lines.append(run_smoke_test(stores, embedder, config))

    report = "\n".join(report_lines)
    print(report)
    output_path = PROJECT_ROOT / config["smoke_test"]["output_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
