"""Tests for the FAISS vs Qdrant benchmark helpers (in-memory Qdrant, fake embedder, tiny corpus)."""

import hashlib

import numpy as np
import pytest

from src.config import PROJECT_ROOT, load_config
from src.index.qdrant_store import QdrantStore, make_client
from src.index.vector_bench import (
    collection_disk_mb, exact_search, format_table, latency_summary, load_bench_queries,
    qdrant_search, recall_at_k,
)

DIMENSION = 16
CORPUS_SIZE = 60


class FakeEmbedder:
    """Deterministic fake embedder: same text -> same unit vector."""

    dimension = DIMENSION

    def encode(self, texts, show_progress=False):
        vectors = []
        for text in texts:
            seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
            vector = np.random.default_rng(seed).normal(size=DIMENSION)
            vectors.append(vector / np.linalg.norm(vector))
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query):
        return self.encode([query])[0]


@pytest.fixture
def corpus():
    """A small corpus of chunks plus their vectors and ids."""
    embedder = FakeEmbedder()
    chunks = [
        {"chunk_id": f"p{i}:abstract:0", "arxiv_id": f"p{i}", "title": f"Title {i}",
         "text": f"text number {i}", "source": "abstract"}
        for i in range(CORPUS_SIZE)
    ]
    vectors = embedder.encode([chunk["text"] for chunk in chunks])
    return chunks, vectors, [chunk["chunk_id"] for chunk in chunks], embedder


@pytest.fixture
def memory_store(corpus):
    """An in-memory Qdrant collection holding the corpus."""
    chunks, vectors, _, embedder = corpus
    vector_config = dict(load_config()["vector_store"], mode="memory")
    store = QdrantStore(make_client(vector_config), "bench_test", embedder, vector_config, use_quantization=False)
    store.build_index(chunks, vectors=vectors)
    return store


def test_latency_summary_percentiles():
    summary = latency_summary([float(value) for value in range(1, 101)])
    assert summary["p50_ms"] == pytest.approx(50.5, abs=1.0)
    assert summary["p90_ms"] == pytest.approx(90.0, abs=1.0)
    assert summary["p99_ms"] == pytest.approx(99.0, abs=1.0)
    assert summary["mean_ms"] == pytest.approx(50.5)


def test_recall_at_k_counts_overlap():
    exact = [["a", "b", "c"], ["d", "e", "f"]]
    approximate = [["a", "b", "x"], ["d", "y", "z"]]
    result = recall_at_k(approximate, exact)
    assert result["recall_at_k"] == pytest.approx((2 / 3 + 1 / 3) / 2)
    assert result["ids_found"] == 3
    assert result["ids_expected"] == 6


def test_recall_at_k_is_one_for_identical_results():
    exact = [["a", "b"], ["c", "d"]]
    assert recall_at_k(exact, exact)["recall_at_k"] == 1.0


def test_exact_search_matches_numpy_argsort(corpus):
    _, vectors, _, embedder = corpus
    queries = embedder.encode(["text number 3", "text number 41"])
    result = exact_search(vectors, queries, top_k=5)

    assert len(result["neighbours"]) == 2
    for row, query_vector in enumerate(queries):
        scores = np.asarray(vectors) @ np.asarray(query_vector)
        expected = list(np.argsort(-scores)[:5])
        assert result["neighbours"][row] == [int(index) for index in expected]
    assert result["n_vectors"] == CORPUS_SIZE
    assert result["latency"]["p50_ms"] >= 0.0


def test_exact_search_puts_the_queried_text_first(corpus):
    _, vectors, chunk_ids, embedder = corpus
    result = exact_search(vectors, embedder.encode(["text number 7"]), top_k=3)
    assert chunk_ids[result["neighbours"][0][0]] == "p7:abstract:0"


def test_qdrant_search_returns_chunk_ids_and_timings(memory_store, corpus):
    _, _, _, embedder = corpus
    queries = embedder.encode(["text number 2", "text number 9"])
    measured = qdrant_search(memory_store, queries, top_k=4)
    assert [len(ids) for ids in measured["chunk_ids"]] == [4, 4]
    assert measured["chunk_ids"][0][0] == "p2:abstract:0"
    assert measured["latency"]["mean_ms"] >= 0.0


def test_exact_and_brute_force_qdrant_agree(memory_store, corpus):
    """Local-mode Qdrant also searches exhaustively, so recall against FAISS must be 1.0."""
    _, vectors, chunk_ids, embedder = corpus
    queries = embedder.encode([f"text number {index}" for index in range(5)])
    exact = exact_search(vectors, queries, top_k=5)
    exact_ids = [[chunk_ids[row] for row in rows] for rows in exact["neighbours"]]
    measured = qdrant_search(memory_store, queries, top_k=5)
    assert recall_at_k(measured["chunk_ids"], exact_ids)["recall_at_k"] == 1.0


def test_collection_disk_mb_handles_missing_and_present(tmp_path):
    assert collection_disk_mb("does_not_exist", storage_dir=tmp_path) is None
    folder = tmp_path / "some_collection"
    folder.mkdir()
    (folder / "segment.bin").write_bytes(b"x" * 2_000_000)
    assert collection_disk_mb("some_collection", storage_dir=tmp_path) == pytest.approx(2.0, abs=0.01)


def test_format_table_has_a_row_per_configuration():
    results = {
        "exact": {"n_vectors": 100, "mean_ms": 1.0, "p50_ms": 1.0, "p90_ms": 1.2, "p99_ms": 1.5},
        "approximate": {
            "fp32@ef64": {"recall_at_k": 0.98, "mean_ms": 2.0, "p50_ms": 2.0, "p90_ms": 2.5, "p99_ms": 3.0, "disk_mb": 123.4},
            "int8@ef64": {"recall_at_k": 0.95, "mean_ms": 1.5, "p50_ms": 1.4, "p90_ms": 1.8, "p99_ms": 2.2, "disk_mb": None},
        },
    }
    table = format_table(results, top_k=10)
    assert "recall@10" in table
    assert "FAISS flat (exact, 100 vectors)" in table
    assert "fp32@ef64" in table and "int8@ef64" in table
    assert "server p50 ms" not in table   # these rows carry no server timings
    assert table.count("\n") == 4  # header, separator, exact row, two approximate rows


def test_format_table_adds_the_server_columns_when_measured():
    results = {
        "exact": {"n_vectors": 100, "mean_ms": 1.0, "p50_ms": 1.0, "p90_ms": 1.2, "p99_ms": 1.5},
        "approximate": {
            "fp32@ef64": {"recall_at_k": 0.98, "mean_ms": 21.0, "p50_ms": 20.0, "p90_ms": 25.0,
                          "p99_ms": 30.0, "disk_mb": 1101.0, "server_p50_ms": 1.4, "server_p90_ms": 2.0},
        },
    }
    table = format_table(results, top_k=10)
    assert "server p50 ms" in table and "server p90 ms" in table
    assert "in-process" in table   # FAISS has no HTTP overhead to report
    assert "| 1.40 |" in table


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "labels" / "query_pool.jsonl").is_file(),
    reason="query pool not built in this checkout",
)
def test_load_bench_queries_is_seeded_and_capped():
    config = load_config()
    first = load_bench_queries(config)
    second = load_bench_queries(config)
    assert first == second
    assert 0 < len(first) <= config["bench"]["n_queries"]
    assert all(query.strip() for query in first)
