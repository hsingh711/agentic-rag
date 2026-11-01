"""Section 10a: prepare HotpotQA and MultiHop-RAG and index each into its own Qdrant collection.

Needs the Qdrant server (scripts/start_qdrant.ps1). No LLM calls.

    python scripts/build_benchmarks.py                 # both
    python scripts/build_benchmarks.py --only hotpotqa
    python scripts/build_benchmarks.py --skip-index    # write the JSONL files only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.eval.benchmarks import (  # noqa: E402
    benchmark_paths,
    hotpotqa_records,
    load_hotpotqa_rows,
    load_multihoprag_files,
    multihoprag_records,
    write_jsonl,
)
from src.index.embedder import Embedder  # noqa: E402
from src.index.qdrant_store import QdrantStore, make_client  # noqa: E402
from src.ingest.chunker import load_tokenizer  # noqa: E402


def prepare(name: str, config: dict):
    """Build (questions, chunks) for one benchmark."""
    settings = config["benchmarks"][name]
    if name == "hotpotqa":
        return hotpotqa_records(load_hotpotqa_rows(settings["questions"]))
    queries, articles = load_multihoprag_files(settings)
    ingest = config["ingest"]
    tokenizer = load_tokenizer(ingest["tokenizer"], ingest["tokenizer_model"])
    return multihoprag_records(
        queries, articles, settings["questions"], config["benchmarks"]["seed"],
        tokenizer, settings["chunk_size_tokens"], settings["chunk_overlap_tokens"],
    )


def main() -> None:
    """Write the question/chunk files and build one fp32 collection per benchmark."""
    parser = argparse.ArgumentParser(description="Prepare and index the multi-hop benchmarks.")
    parser.add_argument("--only", choices=["hotpotqa", "multihoprag"])
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    config = load_config()
    names = [args.only] if args.only else ["hotpotqa", "multihoprag"]
    embedder = None
    for name in names:
        questions, chunks = prepare(name, config)
        questions_path, chunks_path = benchmark_paths(name)
        write_jsonl(questions_path, questions)
        write_jsonl(chunks_path, chunks)
        print(f"{name}: {len(questions)} questions, {len(chunks)} chunks -> {questions_path.parent}")
        if args.skip_index:
            continue
        embedder = embedder or Embedder.from_config(config)
        vector_config = config["vector_store"]
        store = QdrantStore(
            make_client(vector_config), config["benchmarks"][name]["collection"], embedder, vector_config,
            use_quantization=False,
        )
        seconds = store.build_index(chunks)
        print(f"  indexed {store.collection_name} in {seconds:.1f}s: {store.describe()['points']} points")


if __name__ == "__main__":
    main()
