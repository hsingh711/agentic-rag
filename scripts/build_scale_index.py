"""Section 11: embed the large arXiv corpus and index it in its own Qdrant collections.

The corpus (``data/processed/chunks_scale.jsonl``) is embedded once into a float32 ``.npy``
memory map, which both the Qdrant builds and the FAISS baseline read, and uploaded into
``scale_fp32`` and ``scale_int8``. The dev collections (``papers_fp32`` / ``papers_int8``),
which the pipeline and the evaluation use, are never touched.

Start Qdrant first and stop Ollama to free memory:
    powershell -File scripts/start_qdrant.ps1
    ollama stop qwen2.5:3b-instruct

Examples (run from the project root):
    python scripts/build_scale_index.py --embed-only     # just the vectors file (resumable)
    python scripts/build_scale_index.py                   # embed, then build both collections
    python scripts/build_scale_index.py --skip-embed --collections int8
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.index.embedder import Embedder  # noqa: E402
from src.index.qdrant_store import QdrantStore, make_client  # noqa: E402
from src.index.scale_index import build_scale_collection, embed_corpus, load_vectors, scale_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description="Embed and index the scale corpus (section 11).")
    parser.add_argument("--embed-only", action="store_true", help="Write the vectors file and stop.")
    parser.add_argument("--skip-embed", action="store_true", help="Reuse the existing vectors file.")
    parser.add_argument(
        "--collections", nargs="+", choices=["fp32", "int8"], default=["fp32", "int8"],
        help="Which scale collections to build.",
    )
    parser.add_argument("--report", type=str, default="results/scale_index.json", help="Where to write the build report.")
    return parser.parse_args()


def main() -> None:
    """Embed the corpus, then build the requested collections and save a build report."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config = load_config()
    vector_config = config["vector_store"]
    chunks_path, vectors_path, ids_path = scale_paths(config)
    if not chunks_path.is_file():
        raise SystemExit(f"{chunks_path} not found. Run scripts/run_ingest.py --scale first.")

    report = {"chunks_file": str(chunks_path), "embedding_model": config["embedding"]["model_name"]}
    embedder = Embedder.from_config(config)

    if not args.skip_embed:
        report["embedding"] = embed_corpus(
            chunks_path, vectors_path, ids_path, embedder, block_size=config["bench"]["embed_batch_size"]
        )
        print(f"Vectors: {vectors_path} ({report['embedding']['chunks']} x {report['embedding']['dimension']})")
    if args.embed_only:
        save_report(args.report, report)
        return

    vectors, chunk_ids = load_vectors(vectors_path, ids_path)
    print(f"Loaded {len(chunk_ids)} vectors of dimension {vectors.shape[1]}")

    client = make_client(vector_config)
    wanted = {
        "fp32": (vector_config["collection_scale_fp32"], False),
        "int8": (vector_config["collection_scale_int8"], True),
    }
    report["collections"] = {}
    for key in args.collections:
        collection_name, quantize = wanted[key]
        store = QdrantStore(client, collection_name, embedder, vector_config, use_quantization=quantize)
        print(f"\nBuilding {collection_name} (quantization: {'int8' if quantize else 'none'}) ...")
        started = time.perf_counter()
        result = build_scale_collection(
            store,
            chunks_path,
            vectors,
            batch_size=vector_config["scale_upsert_batch_size"],
            final_threshold_kb=vector_config["scale_indexing_threshold_kb"],
            index_timeout_seconds=vector_config["scale_wait_for_index_seconds"],
        )
        result["total_seconds"] = time.perf_counter() - started
        report["collections"][collection_name] = result
        print(
            f"{collection_name}: {result['points']} points, upload {result['upload_seconds']:.0f} s, "
            f"index {result['index_seconds']:.0f} s, indexed_vectors {result['indexed_vectors']}, "
            f"status {result['status']}"
        )

    save_report(args.report, report)


def save_report(relative_path: str, report: dict) -> None:
    """Write the build report JSON under the project root."""
    path = PROJECT_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved build report to {path}")


if __name__ == "__main__":
    main()
