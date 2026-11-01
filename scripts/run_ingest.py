"""Run corpus ingestion: fetch arXiv papers, then chunk them.

Examples (run from the project root):
    python scripts/run_ingest.py                    # dev corpus: ~500 abstracts
    python scripts/run_ingest.py --max-results 50   # smaller test run
    python scripts/run_ingest.py --scale            # large corpus (long; safe to interrupt and re-run)
    python scripts/run_ingest.py --scale --skip-fulltext

Running the large corpus in the background on Windows PowerShell:
    Start-Process -NoNewWindow -RedirectStandardOutput scale.log -RedirectStandardError scale.err `
        RAGenv\\Scripts\\python.exe -ArgumentList "scripts/run_ingest.py","--scale"
"""

import argparse
import logging
import sys
from pathlib import Path

# Make "src" importable when the script is run as `python scripts/run_ingest.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.ingest.arxiv_fetch import fetch_papers, fetch_scale  # noqa: E402
from src.ingest.chunker import build_chunks_file, load_tokenizer  # noqa: E402
from src.ingest.fulltext import download_fulltexts  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description="Fetch arXiv papers and chunk them.")
    parser.add_argument("--max-results", type=int, default=None, help="Override ingest.max_results (dev mode).")
    parser.add_argument("--scale", action="store_true", help="Build the large corpus for the vector benchmark.")
    parser.add_argument("--skip-fulltext", action="store_true", help="With --scale: abstracts only, no PDF downloads.")
    parser.add_argument("--skip-fetch", action="store_true", help="Only re-chunk existing papers.")
    parser.add_argument("--skip-chunk", action="store_true", help="Only fetch, do not chunk.")
    return parser.parse_args()


def main() -> None:
    """Fetch (and optionally download full text), then chunk."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config()
    raw_dir = PROJECT_ROOT / config["paths"]["raw_dir"]
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]

    if args.scale:
        scale_config = config["scale"]
        papers_path = raw_dir / scale_config["papers_file"]
        chunks_path = processed_dir / scale_config["chunks_file"]
        fulltext_dir = raw_dir / scale_config["fulltext_dir"]
        if not args.skip_fetch:
            fetch_scale(config, papers_path, raw_dir / scale_config["state_file"])
            if not args.skip_fulltext:
                download_fulltexts(config, papers_path, fulltext_dir)
        chunk_fulltext_dir = None if args.skip_fulltext else fulltext_dir
        # Full text is cut smaller than abstracts so MiniLM's 256-token limit does not truncate it.
        fulltext_chunk_size = scale_config["fulltext_chunk_size_tokens"]
        fulltext_overlap = scale_config["fulltext_chunk_overlap_tokens"]
    else:
        ingest_config = config["ingest"]
        papers_path = raw_dir / ingest_config["papers_file"]
        chunks_path = processed_dir / ingest_config["chunks_file"]
        max_results = args.max_results or ingest_config["max_results"]
        if not args.skip_fetch:
            fetch_papers(config, papers_path, max_results)
        chunk_fulltext_dir = None
        fulltext_chunk_size = None
        fulltext_overlap = None

    if args.skip_chunk:
        return

    ingest_config = config["ingest"]
    tokenizer = load_tokenizer(ingest_config["tokenizer"], ingest_config["tokenizer_model"])
    chunk_count = build_chunks_file(
        papers_path,
        chunks_path,
        tokenizer,
        ingest_config["chunk_size_tokens"],
        ingest_config["chunk_overlap_tokens"],
        fulltext_dir=chunk_fulltext_dir,
        fulltext_chunk_size=fulltext_chunk_size,
        fulltext_overlap=fulltext_overlap,
    )
    print(f"Chunks written: {chunk_count} -> {chunks_path}")


if __name__ == "__main__":
    main()
