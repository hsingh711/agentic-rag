"""Download PDFs and extract their text (used only for the large --scale corpus).

Extracted text is cached as one ``<arxiv_id>.txt`` file per paper, so re-running skips
papers that are already done. Papers whose download or extraction fails are listed in
``failed.txt`` and are not retried.
"""

import io
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Set

import requests
from pypdf import PdfReader

from src.ingest.io_utils import PathLike, read_jsonl

logger = logging.getLogger(__name__)

FAILED_FILE_NAME = "failed.txt"


def safe_file_stem(arxiv_id: str) -> str:
    """Make an arXiv id safe for use in a file name (old ids contain a slash)."""
    return arxiv_id.replace("/", "_")


def extract_pdf_text(pdf_bytes: bytes, max_pages: int) -> str:
    """Extract and whitespace-normalise text from the first ``max_pages`` pages of a PDF."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    page_texts = []
    for page in reader.pages[:max_pages]:
        page_texts.append(page.extract_text() or "")
    return re.sub(r"\s+", " ", " ".join(page_texts)).strip()


def load_failed_ids(fulltext_dir: Path) -> Set[str]:
    """Read the ids of papers whose full text already failed once."""
    failed_path = fulltext_dir / FAILED_FILE_NAME
    if not failed_path.is_file():
        return set()
    return {line.strip() for line in failed_path.read_text(encoding="utf-8").splitlines() if line.strip()}


def download_fulltexts(config: Dict[str, Any], papers_path: PathLike, fulltext_dir: PathLike) -> int:
    """Download full text for the first ``scale.fulltext_papers`` papers in ``papers_path``.

    Sleeps ``ingest.request_sleep_seconds`` between downloads to respect arXiv limits.

    Returns:
        The number of papers whose text was newly saved in this call.
    """
    scale_config = config["scale"]
    sleep_seconds = config["ingest"]["request_sleep_seconds"]
    limit = scale_config["fulltext_papers"]
    max_pages = scale_config["fulltext_max_pages"]

    fulltext_dir = Path(fulltext_dir)
    fulltext_dir.mkdir(parents=True, exist_ok=True)
    failed_ids = load_failed_ids(fulltext_dir)

    newly_saved = 0
    for position, paper in enumerate(read_jsonl(papers_path)):
        if position >= limit:
            break
        arxiv_id = paper["arxiv_id"]
        text_path = fulltext_dir / f"{safe_file_stem(arxiv_id)}.txt"
        if text_path.is_file() or arxiv_id in failed_ids:
            continue

        try:
            response = requests.get(paper["pdf_url"], timeout=60)
            response.raise_for_status()
            text = extract_pdf_text(response.content, max_pages)
            if not text:
                raise ValueError("no text extracted")
            text_path.write_text(text, encoding="utf-8")
            newly_saved += 1
        except Exception as error:  # network, parsing and encoding problems all mean "skip this paper"
            logger.warning("Full text failed for %s: %s", arxiv_id, error)
            with open(fulltext_dir / FAILED_FILE_NAME, "a", encoding="utf-8") as failed_file:
                failed_file.write(arxiv_id + "\n")

        if newly_saved % 50 == 0 and newly_saved > 0:
            logger.info("Full text saved for %d papers so far", newly_saved)
        time.sleep(sleep_seconds)

    logger.info("Full text: %d newly saved.", newly_saved)
    return newly_saved
