"""Embed and index the large benchmark corpus (section 11) without loading it all into memory.

The scale corpus has hundreds of thousands of chunks, so neither the texts nor the vectors fit
comfortably next to Ollama and Qdrant on a 16 GB laptop. Both steps therefore stream:

1. ``embed_corpus`` reads the chunks JSONL in blocks, encodes each block and writes the vectors
   straight into a float32 ``.npy`` memory map, plus a JSON list of chunk_ids in the same order.
   It resumes: an interrupted run continues at the first row that is still all zeros.
2. ``build_scale_collection`` uploads those vectors to a Qdrant collection in batches, with HNSW
   building switched off during the upload (``indexing_threshold`` 0) and switched on afterwards,
   so the index build can be timed on its own.

The vectors file is also what the FAISS exact baseline in ``src.index.vector_bench`` reads, so
the corpus is embedded exactly once.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from qdrant_client import models

from src.config import PROJECT_ROOT
from src.index.qdrant_store import QdrantStore, chunk_id_to_point_id
from src.ingest.io_utils import read_jsonl

logger = logging.getLogger(__name__)


def scale_paths(config: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    """Return ``(chunks_path, vectors_path, ids_path)`` of the scale corpus."""
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]
    scale_config = config["scale"]
    return (
        processed_dir / scale_config["chunks_file"],
        processed_dir / scale_config["vectors_file"],
        processed_dir / scale_config["vector_ids_file"],
    )


def count_lines(path: Path) -> int:
    """Number of non-empty lines in a text file (chunk count without parsing the JSON)."""
    total = 0
    with open(path, "rb") as handle:
        for line in handle:
            if line.strip():
                total += 1
    return total


def iter_blocks(records: Iterator[Dict[str, Any]], block_size: int) -> Iterator[List[Dict[str, Any]]]:
    """Group an iterator of records into lists of at most ``block_size`` items."""
    block: List[Dict[str, Any]] = []
    for record in records:
        block.append(record)
        if len(block) == block_size:
            yield block
            block = []
    if block:
        yield block


def first_unwritten_row(vectors: np.ndarray) -> int:
    """Index of the first all-zero row of a memory-mapped vectors file.

    Normalised embeddings are never all zero, so an all-zero row means "not written yet".
    Returns ``len(vectors)`` when every row is filled.
    """
    # Scanning 300k x 384 float32 in blocks keeps peak memory small.
    block_size = 20000
    for start in range(0, len(vectors), block_size):
        block = np.asarray(vectors[start : start + block_size])
        row_sums = np.abs(block).sum(axis=1)
        empty = np.nonzero(row_sums == 0)[0]
        if len(empty):
            return start + int(empty[0])
    return len(vectors)


def embed_corpus(
    chunks_path: Path,
    vectors_path: Path,
    ids_path: Path,
    embedder: Any,
    block_size: int = 20000,
    log_every: int = 1,
) -> Dict[str, Any]:
    """Embed every chunk of ``chunks_path`` into a float32 memmap at ``vectors_path``.

    The chunk_ids are written to ``ids_path`` in the same row order, so row *i* of the vectors
    file belongs to ``ids[i]``. An existing vectors file with the right shape is resumed, not
    rebuilt; delete it to force a fresh run.

    Args:
        chunks_path: Chunks JSONL (one record per line, fields chunk_id and text).
        vectors_path: Output ``.npy`` file.
        ids_path: Output JSON file with the chunk_id list.
        embedder: Object with ``dimension`` and ``encode(texts)``.
        block_size: Chunks read, encoded and flushed per step.
        log_every: Log progress every N blocks.

    Returns:
        ``{"chunks", "dimension", "seconds", "resumed_at"}``.
    """
    total = count_lines(chunks_path)
    dimension = embedder.dimension
    exists = vectors_path.is_file()
    if exists:
        existing = np.load(vectors_path, mmap_mode="r")
        if existing.shape != (total, dimension):
            logger.warning(
                "Vectors file %s has shape %s but the corpus needs %s: rebuilding.",
                vectors_path, existing.shape, (total, dimension),
            )
            exists = False
        del existing

    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    if not exists:
        # np.lib.format.open_memmap writes a real .npy header, so np.load can read it later.
        vectors = np.lib.format.open_memmap(vectors_path, mode="w+", dtype=np.float32, shape=(total, dimension))
        start_row = 0
    else:
        vectors = np.load(vectors_path, mmap_mode="r+")
        start_row = first_unwritten_row(vectors)

    logger.info("Embedding %s: %d chunks, starting at row %d", chunks_path.name, total, start_row)
    started = time.perf_counter()
    chunk_ids: List[str] = []
    row = 0
    for block_number, block in enumerate(iter_blocks(read_jsonl(chunks_path), block_size)):
        chunk_ids.extend(record["chunk_id"] for record in block)
        block_end = row + len(block)
        if block_end <= start_row:
            row = block_end
            continue
        block_vectors = embedder.encode([record["text"] for record in block])
        vectors[row:block_end] = block_vectors
        vectors.flush()
        row = block_end
        if block_number % log_every == 0:
            elapsed = time.perf_counter() - started
            done = row - start_row
            rate = done / elapsed if elapsed else 0.0
            remaining = (total - row) / rate if rate else 0.0
            logger.info("  %d/%d chunks (%.0f per second, about %.0f s left)", row, total, rate, remaining)

    ids_path.write_text(json.dumps(chunk_ids), encoding="utf-8")
    elapsed = time.perf_counter() - started
    logger.info("Embedded %d chunks in %.1f s -> %s", total - start_row, elapsed, vectors_path)
    return {"chunks": total, "dimension": dimension, "seconds": elapsed, "resumed_at": start_row}


def wait_until_indexed(store: QdrantStore, timeout_seconds: int, poll_seconds: float = 5.0) -> bool:
    """Block until the collection is green and every vector is in the HNSW index.

    ``QdrantStore.wait_until_indexed`` only waits for the green status and for at most
    ``vector_store.wait_for_index_seconds``; a corpus of this size needs a much longer
    deadline and the stricter check that ``indexed_vectors_count`` has caught up.

    Returns:
        True if the index finished inside the timeout.
    """
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        info = store.client.get_collection(store.collection_name)
        indexed = info.indexed_vectors_count or 0
        points = info.points_count or 0
        if info.status == models.CollectionStatus.GREEN and indexed >= points:
            return True
        logger.info(
            "  %s: status %s, %d/%d vectors indexed", store.collection_name, info.status, indexed, points
        )
        time.sleep(poll_seconds)
    logger.warning("%s did not finish indexing within %d s", store.collection_name, timeout_seconds)
    return False


def build_scale_collection(
    store: QdrantStore,
    chunks_path: Path,
    vectors: np.ndarray,
    batch_size: int,
    final_threshold_kb: int,
    index_timeout_seconds: int = 7200,
    payload_fields: Tuple[str, ...] = ("chunk_id", "arxiv_id", "title", "text", "source"),
) -> Dict[str, Any]:
    """Create a collection, upload the corpus with indexing off, then build the index.

    Args:
        store: A ``QdrantStore`` for the target collection (quantization already decided).
        chunks_path: Chunks JSONL, read in the same order as the ``vectors`` rows.
        vectors: Vectors array or memmap, one row per chunk.
        batch_size: Points per upsert request.
        final_threshold_kb: ``indexing_threshold`` set after the upload, which starts the HNSW build.
        index_timeout_seconds: How long to wait for the index build.
        payload_fields: Chunk fields stored as the point payload.

    Returns:
        ``{"upload_seconds", "index_seconds", "indexed", "points", "indexed_vectors", "status"}``.
    """
    store.create_collection()
    # Switch HNSW off for the upload: building it while points stream in is much slower.
    store.client.update_collection(
        collection_name=store.collection_name,
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )

    upload_started = time.perf_counter()
    row = 0
    for block in iter_blocks(read_jsonl(chunks_path), batch_size):
        points = [
            models.PointStruct(
                id=chunk_id_to_point_id(record["chunk_id"]),
                vector=np.asarray(vectors[row + offset]).tolist(),
                payload={field: record.get(field) for field in payload_fields},
            )
            for offset, record in enumerate(block)
        ]
        store.client.upsert(collection_name=store.collection_name, points=points, wait=False)
        row += len(block)
        if row % (batch_size * 20) == 0:
            logger.info("  uploaded %d points to %s", row, store.collection_name)
    upload_seconds = time.perf_counter() - upload_started

    index_started = time.perf_counter()
    store.client.update_collection(
        collection_name=store.collection_name,
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=final_threshold_kb),
    )
    indexed = wait_until_indexed(store, index_timeout_seconds)
    index_seconds = time.perf_counter() - index_started

    info = store.describe()
    logger.info(
        "%s: %s points, upload %.1f s, index %.1f s, status %s",
        store.collection_name, info["points"], upload_seconds, index_seconds, info["status"],
    )
    return {
        "upload_seconds": upload_seconds,
        "index_seconds": index_seconds,
        "indexed": indexed,
        "points": info["points"],
        "indexed_vectors": info["indexed_vectors"],
        "status": info["status"],
    }


def load_vectors(vectors_path: Path, ids_path: Optional[Path] = None):
    """Memory-map the vectors file (and read the chunk_id list when ``ids_path`` is given)."""
    vectors = np.load(vectors_path, mmap_mode="r")
    if ids_path is None:
        return vectors
    return vectors, json.loads(ids_path.read_text(encoding="utf-8"))
