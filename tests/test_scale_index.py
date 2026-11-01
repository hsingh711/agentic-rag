"""Tests for the streaming embed/index helpers of the scale corpus (no server, no real model)."""

import hashlib
import json

import numpy as np
import pytest

from src.index.scale_index import (
    count_lines, embed_corpus, first_unwritten_row, iter_blocks, load_vectors, scale_paths,
)

DIMENSION = 8


class FakeEmbedder:
    """Deterministic fake embedder: same text -> same unit vector."""

    dimension = DIMENSION

    def __init__(self):
        self.calls = 0

    def encode(self, texts, show_progress=False):
        self.calls += 1
        vectors = []
        for text in texts:
            seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
            vector = np.random.default_rng(seed).normal(size=DIMENSION)
            vectors.append(vector / np.linalg.norm(vector))
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query):
        return self.encode([query])[0]


def write_chunks(path, count):
    """Write ``count`` chunk records to a JSONL file and return their ids."""
    ids = []
    with open(path, "w", encoding="utf-8") as handle:
        for index in range(count):
            record = {"chunk_id": f"p{index}:abstract:0", "text": f"text number {index}", "title": f"T{index}"}
            ids.append(record["chunk_id"])
            handle.write(json.dumps(record) + "\n")
    return ids


def test_count_lines_ignores_blank_lines(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert count_lines(path) == 2


def test_iter_blocks_splits_and_keeps_remainder():
    blocks = list(iter_blocks(iter(range(7)), 3))
    assert [len(block) for block in blocks] == [3, 3, 1]
    assert blocks[-1] == [6]


def test_first_unwritten_row_finds_zero_row():
    vectors = np.ones((5, DIMENSION), dtype=np.float32)
    assert first_unwritten_row(vectors) == 5
    vectors[3] = 0.0
    assert first_unwritten_row(vectors) == 3


def test_embed_corpus_writes_vectors_and_ids(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    ids = write_chunks(chunks_path, 25)
    vectors_path = tmp_path / "vectors.npy"
    ids_path = tmp_path / "ids.json"

    report = embed_corpus(chunks_path, vectors_path, ids_path, FakeEmbedder(), block_size=10)

    assert report["chunks"] == 25
    assert report["resumed_at"] == 0
    vectors, stored_ids = load_vectors(vectors_path, ids_path)
    assert vectors.shape == (25, DIMENSION)
    assert stored_ids == ids
    # Every row is a real unit vector, in the same order as the file.
    assert np.allclose(np.linalg.norm(np.asarray(vectors), axis=1), 1.0, atol=1e-5)
    expected = FakeEmbedder().encode(["text number 7"])[0]
    assert np.allclose(np.asarray(vectors[7]), expected, atol=1e-6)


def test_embed_corpus_resumes_and_skips_finished_blocks(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(chunks_path, 30)
    vectors_path = tmp_path / "vectors.npy"
    ids_path = tmp_path / "ids.json"
    embedder = FakeEmbedder()
    embed_corpus(chunks_path, vectors_path, ids_path, embedder, block_size=10)
    calls_after_full_run = embedder.calls

    # Wipe the last block: the resumed run must encode only that one.
    vectors = np.load(vectors_path, mmap_mode="r+")
    vectors[20:] = 0.0
    vectors.flush()
    del vectors

    second = FakeEmbedder()
    report = embed_corpus(chunks_path, vectors_path, ids_path, second, block_size=10)
    assert report["resumed_at"] == 20
    assert second.calls == 1 < calls_after_full_run
    vectors = np.load(vectors_path)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_embed_corpus_rebuilds_on_shape_mismatch(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(chunks_path, 12)
    vectors_path = tmp_path / "vectors.npy"
    ids_path = tmp_path / "ids.json"
    np.save(vectors_path, np.ones((3, DIMENSION), dtype=np.float32))

    report = embed_corpus(chunks_path, vectors_path, ids_path, FakeEmbedder(), block_size=5)

    assert report["chunks"] == 12
    assert np.load(vectors_path).shape == (12, DIMENSION)


def test_scale_paths_point_into_the_processed_dir():
    from src.config import load_config

    config = load_config()
    chunks_path, vectors_path, ids_path = scale_paths(config)
    assert chunks_path.name == config["scale"]["chunks_file"]
    assert vectors_path.suffix == ".npy"
    assert ids_path.parent == chunks_path.parent


@pytest.mark.parametrize("block_size", [1, 4, 100])
def test_embed_corpus_is_block_size_independent(tmp_path, block_size):
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(chunks_path, 9)
    vectors_path = tmp_path / f"vectors_{block_size}.npy"
    ids_path = tmp_path / f"ids_{block_size}.json"
    embed_corpus(chunks_path, vectors_path, ids_path, FakeEmbedder(), block_size=block_size)
    reference = FakeEmbedder().encode([f"text number {index}" for index in range(9)])
    assert np.allclose(np.load(vectors_path), reference, atol=1e-6)
