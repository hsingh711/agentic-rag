"""Tests for the embedder wrapper and Qdrant store (in-memory Qdrant, no server, no model download)."""

import hashlib

import numpy as np
import pytest

from src.config import load_config
from src.index import embedder as embedder_module
from src.index.qdrant_store import QdrantStore, chunk_id_to_point_id, make_client

DIMENSION = 16


class FakeEmbedder:
    """Deterministic fake: the same text always maps to the same unit vector."""

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


def make_chunks(count):
    return [
        {"chunk_id": f"p{i}:abstract:0", "arxiv_id": f"p{i}", "title": f"Title {i}", "text": f"text number {i}", "source": "abstract"}
        for i in range(count)
    ]


@pytest.fixture
def vector_config():
    config = load_config()["vector_store"]
    config = dict(config, mode="memory")
    return config


def make_store(vector_config, name="test_col", quantize=False):
    return QdrantStore(make_client(vector_config), name, FakeEmbedder(), vector_config, use_quantization=quantize)


def test_point_id_is_stable_uuid():
    assert chunk_id_to_point_id("a:abstract:0") == chunk_id_to_point_id("a:abstract:0")
    assert chunk_id_to_point_id("a:abstract:0") != chunk_id_to_point_id("a:abstract:1")


def test_search_finds_exact_chunk_first(vector_config):
    store = make_store(vector_config)
    store.build_index(make_chunks(30))
    hits = store.search("text number 7", top_k=3)
    assert len(hits) == 3
    assert hits[0]["chunk_id"] == "p7:abstract:0"
    assert hits[0]["title"] == "Title 7"
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-3)


def test_rebuild_does_not_duplicate(vector_config):
    store = make_store(vector_config)
    store.build_index(make_chunks(20))
    store.build_index(make_chunks(20))
    assert store.describe()["points"] == 20


def test_quantized_collection_builds_and_searches(vector_config):
    store = make_store(vector_config, name="int8_col", quantize=True)
    store.build_index(make_chunks(25))
    assert store.search("text number 3", top_k=1)[0]["chunk_id"] == "p3:abstract:0"


def test_mismatched_vectors_raise(vector_config):
    store = make_store(vector_config)
    with pytest.raises(ValueError):
        store.build_index(make_chunks(5), vectors=np.zeros((4, DIMENSION), dtype=np.float32))


def test_unknown_mode_raises(vector_config):
    with pytest.raises(ValueError):
        make_client(dict(vector_config, mode="bogus"))


def test_embedder_wrapper_passes_settings(monkeypatch):
    calls = {}

    class StubModel:
        def __init__(self, name):
            calls["name"] = name

        def get_sentence_embedding_dimension(self):
            return 4

        def encode(self, texts, **kwargs):
            calls["kwargs"] = kwargs
            return np.ones((len(texts), 4), dtype=np.float64)

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", StubModel)
    embedder = embedder_module.Embedder("some/model", batch_size=8, normalize=True)

    vectors = embedder.encode(["a", "b"])
    assert calls["name"] == "some/model"
    assert calls["kwargs"]["batch_size"] == 8
    assert calls["kwargs"]["normalize_embeddings"] is True
    assert vectors.dtype == np.float32 and vectors.shape == (2, 4)
    assert embedder.encode_query("q").shape == (4,)
