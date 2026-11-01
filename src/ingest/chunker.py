"""Split paper abstracts (and optional full text) into overlapping token chunks.

Chunks are cut by token count (default 400 tokens, 50 overlap) but the chunk text is
sliced from the ORIGINAL string using character offsets, so no text is altered.

Full text can use a smaller window than abstracts (``fulltext_chunk_size``): the embedder
all-MiniLM-L6-v2 truncates its input at 256 tokens, so a 400-token full-text chunk would be
embedded only in part. Abstracts are about 250 tokens and stay in one chunk either way.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ingest.io_utils import PathLike, read_jsonl, replace_file_atomically

logger = logging.getLogger(__name__)


class WhitespaceTokenizer:
    """Offline fallback tokenizer: one token per whitespace-separated word.

    It mimics the small part of the Hugging Face fast-tokenizer interface that the
    chunker needs (``return_offsets_mapping``), so the two can be swapped freely.
    """

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = True) -> Dict[str, Any]:
        """Return ``{"offset_mapping": [(start, end), ...]}`` for each word in ``text``."""
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return {"offset_mapping": offsets}


def load_tokenizer(kind: str, model_name: str):
    """Build the tokenizer named in the config.

    Args:
        kind: ``"huggingface"`` or ``"whitespace"``.
        model_name: Hugging Face model whose tokenizer to use (ignored for whitespace).
    """
    if kind == "whitespace":
        return WhitespaceTokenizer()
    if kind == "huggingface":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Full-text papers are far longer than the model limit; we only count tokens here.
        tokenizer.model_max_length = 10**9
        return tokenizer
    raise ValueError(f"Unknown tokenizer kind: {kind!r} (use 'huggingface' or 'whitespace')")


def chunk_text(text: str, tokenizer, chunk_size: int, overlap: int) -> List[str]:
    """Split ``text`` into windows of ``chunk_size`` tokens that overlap by ``overlap`` tokens.

    Args:
        text: The text to split.
        tokenizer: Object returning ``offset_mapping`` when called (see WhitespaceTokenizer).
        chunk_size: Maximum number of tokens per chunk.
        overlap: Number of tokens shared by consecutive chunks. Must be < chunk_size.

    Returns:
        A list of chunk strings. Empty text gives an empty list; text that fits in one
        chunk gives a single chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets: List[Tuple[int, int]] = list(encoding["offset_mapping"])
    token_count = len(offsets)
    if token_count == 0:
        return []

    step = chunk_size - overlap
    chunks: List[str] = []
    start_token = 0
    while True:
        end_token = min(start_token + chunk_size, token_count)
        start_char = offsets[start_token][0]
        end_char = offsets[end_token - 1][1]
        chunks.append(text[start_char:end_char].strip())
        if end_token == token_count:
            break
        start_token += step
    return chunks


def chunk_paper(
    paper: Dict[str, Any],
    tokenizer,
    chunk_size: int,
    overlap: int,
    fulltext: Optional[str] = None,
    fulltext_chunk_size: Optional[int] = None,
    fulltext_overlap: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Turn one paper record into chunk records (abstract chunks, then full-text chunks).

    Each chunk has: chunk_id, arxiv_id, title, text, source ("abstract" or "fulltext").
    chunk_id looks like ``2401.00001:abstract:0`` and is stable across runs.

    Args:
        fulltext_chunk_size: Window for the full-text section; defaults to ``chunk_size``.
        fulltext_overlap: Overlap for the full-text section; defaults to ``overlap``.
    """
    chunk_records: List[Dict[str, Any]] = []
    sections = [("abstract", paper.get("abstract", ""), chunk_size, overlap)]
    if fulltext:
        sections.append((
            "fulltext",
            fulltext,
            fulltext_chunk_size if fulltext_chunk_size is not None else chunk_size,
            fulltext_overlap if fulltext_overlap is not None else overlap,
        ))

    for source, text, section_size, section_overlap in sections:
        pieces = chunk_text(text, tokenizer, section_size, section_overlap)
        for index, piece in enumerate(pieces):
            chunk_records.append(
                {
                    "chunk_id": f"{paper['arxiv_id']}:{source}:{index}",
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper.get("title", ""),
                    "text": piece,
                    "source": source,
                }
            )
    return chunk_records


def build_chunks_file(
    papers_path: PathLike,
    chunks_path: PathLike,
    tokenizer,
    chunk_size: int,
    overlap: int,
    fulltext_dir: Optional[PathLike] = None,
    fulltext_chunk_size: Optional[int] = None,
    fulltext_overlap: Optional[int] = None,
) -> int:
    """Chunk every paper in ``papers_path`` and write ``chunks_path``.

    The output is rebuilt from scratch on every call (written to a temporary file and
    swapped in at the end), so re-running never creates duplicates and a crash never
    leaves a half-written chunks file.

    Args:
        fulltext_dir: Optional folder of ``<arxiv_id>.txt`` files; matching papers also
            get full-text chunks.
        fulltext_chunk_size, fulltext_overlap: Window for full-text sections; default to
            ``chunk_size`` / ``overlap``.

    Returns:
        The number of chunks written.
    """
    chunks_path = Path(chunks_path)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = chunks_path.with_suffix(chunks_path.suffix + ".tmp")

    seen_ids = set()
    total_chunks = 0
    with open(temp_path, "w", encoding="utf-8") as output_file:
        for paper in read_jsonl(papers_path):
            if paper["arxiv_id"] in seen_ids:
                continue
            seen_ids.add(paper["arxiv_id"])

            fulltext = None
            if fulltext_dir is not None:
                fulltext_file = Path(fulltext_dir) / f"{paper['arxiv_id'].replace('/', '_')}.txt"
                if fulltext_file.is_file():
                    fulltext = fulltext_file.read_text(encoding="utf-8")

            for chunk in chunk_paper(
                paper, tokenizer, chunk_size, overlap, fulltext, fulltext_chunk_size, fulltext_overlap
            ):
                output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1

    replace_file_atomically(temp_path, chunks_path)
    logger.info("Wrote %d chunks from %d papers to %s", total_chunks, len(seen_ids), chunks_path)
    return total_chunks
