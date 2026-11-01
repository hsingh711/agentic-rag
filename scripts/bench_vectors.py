"""Section 11: benchmark FAISS exact search against Qdrant HNSW (fp32 and int8).

Needs the scale collections built (``scripts/build_scale_index.py``) and the Qdrant server
running. Makes no LLM calls, so it costs nothing and needs no quota.

    python scripts/bench_vectors.py                       # scale corpus, both collections
    python scripts/bench_vectors.py --corpus dev          # the 528-chunk dev corpus instead
    python scripts/bench_vectors.py --queries 50 --ef 64 128
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.index.embedder import Embedder  # noqa: E402
from src.index.qdrant_store import QdrantStore, make_client  # noqa: E402
from src.index.scale_index import load_vectors, scale_paths  # noqa: E402
from src.index.vector_bench import benchmark, format_table, load_bench_queries, save_results  # noqa: E402
from src.ingest.io_utils import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description="FAISS vs Qdrant HNSW fp32/int8 benchmark.")
    parser.add_argument("--corpus", choices=["scale", "dev"], default="scale", help="Which corpus/collections to measure.")
    parser.add_argument("--queries", type=int, default=None, help="Override bench.n_queries.")
    parser.add_argument("--top-k", type=int, default=None, help="Override bench.top_k.")
    parser.add_argument("--ef", nargs="+", type=int, default=None, help="Override bench.ef_search.")
    parser.add_argument("--repeats", type=int, default=None, help="Override bench.repeats.")
    return parser.parse_args()


def dev_vectors(config, embedder):
    """Embed the small dev corpus on the fly; returns ``(vectors, chunk_ids)``."""
    chunks_path = PROJECT_ROOT / config["paths"]["processed_dir"] / config["ingest"]["chunks_file"]
    chunks = list(read_jsonl(chunks_path))
    vectors = embedder.encode([chunk["text"] for chunk in chunks])
    return vectors, [chunk["chunk_id"] for chunk in chunks]


def main() -> None:
    """Run the benchmark and print the markdown table."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config = load_config()
    bench_config = config["bench"]
    vector_config = config["vector_store"]

    top_k = args.top_k or bench_config["top_k"]
    ef_values = args.ef or bench_config["ef_search"]
    repeats = args.repeats or bench_config["repeats"]
    if args.queries:
        bench_config["n_queries"] = args.queries

    embedder = Embedder.from_config(config)
    if args.corpus == "scale":
        _, vectors_path, ids_path = scale_paths(config)
        if not vectors_path.is_file():
            raise SystemExit(f"{vectors_path} not found. Run scripts/build_scale_index.py first.")
        vectors, chunk_ids = load_vectors(vectors_path, ids_path)
        collections = [(vector_config["collection_scale_fp32"], False), (vector_config["collection_scale_int8"], True)]
    else:
        vectors, chunk_ids = dev_vectors(config, embedder)
        collections = [(vector_config["collection_fp32"], False), (vector_config["collection_int8"], True)]

    queries = load_bench_queries(config)
    query_vectors = embedder.encode(queries)
    print(f"{args.corpus} corpus: {len(chunk_ids)} vectors, {len(queries)} queries, top_k {top_k}, ef {ef_values}")

    client = make_client(vector_config)
    stores = {}
    for collection_name, quantize in collections:
        if not client.collection_exists(collection_name):
            print(f"  (skipping {collection_name}: collection does not exist)")
            continue
        label = "int8" if quantize else "fp32"
        stores[label] = QdrantStore(client, collection_name, embedder, vector_config, use_quantization=quantize)
    if not stores:
        raise SystemExit("No collections to measure.")

    # With a real server, searches go over plain HTTP so the server's own search time is
    # recorded next to the client-side latency (the loopback round trip dominates the latter).
    rest_url = vector_config["url"] if vector_config["mode"] == "server" else None
    results = benchmark(vectors, chunk_ids, query_vectors, stores, top_k, ef_values, repeats, rest_url)
    results["settings"] = {
        "corpus": args.corpus,
        "n_vectors": len(chunk_ids),
        "n_queries": len(queries),
        "top_k": top_k,
        "ef_search": list(ef_values),
        "repeats": repeats,
        "embedding_model": config["embedding"]["model_name"],
        "hnsw_m": vector_config["hnsw_m"],
        "hnsw_ef_construct": vector_config["hnsw_ef_construct"],
    }
    table = format_table(results, top_k)
    print("\n" + table)
    results["table"] = table
    suffix = "" if args.corpus == "scale" else args.corpus
    print(f"\nSaved {save_results(config, results, suffix)}")


if __name__ == "__main__":
    main()
