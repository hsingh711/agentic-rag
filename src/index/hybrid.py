"""Section 11: hybrid retrieval (dense + BM25) with reciprocal rank fusion and an optional reranker.

``HybridStore`` has the same ``search(query, top_k)`` interface as ``QdrantStore``, so the graph and
the vanilla baseline can use it unchanged. Everything is local and free:
- ``BM25Index``: Okapi BM25 over the chunk texts (pure Python inverted index, no extra package);
- ``reciprocal_rank_fusion``: merges ranked lists by rank, so dense cosine scores and BM25 scores,
  which are on different scales, never need to be compared;
- ``CrossEncoderReranker``: re-scores the best fused candidates with a cross-encoder
  (default cross-encoder/ms-marco-MiniLM-L-6-v2).
"""

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

from src.config import load_config

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOP_WORDS = frozenset(
    "a an the of in on at to for and or is are was were be been by with as from that this it its which who whom "
    "what when where how why do does did has have had not but if than then so such can could would should".split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens without stop words."""
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOP_WORDS]


class BM25Index:
    """Okapi BM25 over a list of chunk dicts (each with ``chunk_id``, ``title`` and ``text``)."""

    def __init__(self, chunks: Sequence[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        """Build the inverted index. The title is indexed together with the text."""
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.postings: Dict[str, List[tuple]] = defaultdict(list)
        self.lengths: List[int] = []
        for position, chunk in enumerate(self.chunks):
            tokens = tokenize(f"{chunk.get('title', '')} {chunk['text']}")
            self.lengths.append(len(tokens))
            for token, count in Counter(tokens).items():
                self.postings[token].append((position, count))
        self.average_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.idf = {
            token: math.log(1 + (len(self.chunks) - len(postings) + 0.5) / (len(postings) + 0.5))
            for token, postings in self.postings.items()
        }

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """The ``top_k`` chunks by BM25 score (best first); chunks sharing no query term are not returned."""
        scores: Dict[int, float] = defaultdict(float)
        for token in set(tokenize(query)):
            idf = self.idf.get(token)
            if idf is None:
                continue
            for position, count in self.postings[token]:
                norm = 1 - self.b + self.b * self.lengths[position] / self.average_length
                scores[position] += idf * count * (self.k1 + 1) / (count + self.k1 * norm)
        best = sorted(scores.items(), key=lambda item: -item[1])[:top_k]
        return [dict(self.chunks[position], score=score) for position, score in best]


def reciprocal_rank_fusion(
    rankings: Sequence[List[Dict[str, Any]]], weights: Optional[Sequence[float]] = None, k: int = 60
) -> List[Dict[str, Any]]:
    """Fuse ranked chunk lists: score(chunk) = sum over lists of weight / (k + rank), rank from 1.

    Returns chunks (first occurrence's payload) sorted by fused score, best first, with ``score`` set.
    """
    weights = weights or [1.0] * len(rankings)
    fused: Dict[str, float] = defaultdict(float)
    payload: Dict[str, Dict[str, Any]] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, chunk in enumerate(ranking, start=1):
            fused[chunk["chunk_id"]] += weight / (k + rank)
            payload.setdefault(chunk["chunk_id"], chunk)
    ordered = sorted(fused, key=lambda chunk_id: -fused[chunk_id])
    return [dict(payload[chunk_id], score=fused[chunk_id]) for chunk_id in ordered]


class CrossEncoderReranker:
    """Re-scores (query, chunk) pairs with a cross-encoder; the model loads on first use."""

    def __init__(self, model_name: str, batch_size: int = 16):
        """Remember the settings."""
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """The ``top_k`` chunks by cross-encoder score (best first)."""
        if not chunks:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=512, device="cpu")
        pairs = [(query, f"{chunk.get('title', '')}. {chunk['text']}") for chunk in chunks]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        ranked = sorted(zip(scores, chunks), key=lambda item: -float(item[0]))
        return [dict(chunk, score=float(score)) for score, chunk in ranked[:top_k]]


class HybridStore:
    """Dense + BM25 retrieval with rank fusion and an optional reranker, behind ``search(query, top_k)``."""

    def __init__(
        self,
        dense_store: Any,
        bm25: BM25Index,
        candidates: int = 50,
        weights: Sequence[float] = (1.0, 1.0),
        rrf_k: int = 60,
        reranker: Optional[CrossEncoderReranker] = None,
        rerank_candidates: int = 30,
    ):
        """Combine a dense store (``search``) with a BM25 index.

        Args:
            candidates: How many hits each retriever contributes to the fusion.
            weights: (dense, bm25) weights in the fusion.
            reranker: If given, the best ``rerank_candidates`` fused chunks are re-scored by it.
        """
        self.dense_store = dense_store
        self.bm25 = bm25
        self.candidates = candidates
        self.weights = list(weights)
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates

    @classmethod
    def from_config(cls, dense_store: Any, chunks: Sequence[Dict[str, Any]], use_reranker: bool, config: Optional[Dict[str, Any]] = None) -> "HybridStore":
        """Build from the ``hybrid`` block of the config."""
        settings = (config or load_config())["hybrid"]
        reranker = CrossEncoderReranker(settings["reranker_model"]) if use_reranker else None
        return cls(
            dense_store, BM25Index(chunks, settings["bm25_k1"], settings["bm25_b"]), settings["candidates"],
            settings["weights"], settings["rrf_k"], reranker, settings["rerank_candidates"],
        )

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fuse dense and BM25 hits; rerank the head of the fused list if a reranker is set."""
        dense_hits = self.dense_store.search(query, self.candidates)
        lexical_hits = self.bm25.search(query, self.candidates)
        fused = reciprocal_rank_fusion([dense_hits, lexical_hits], self.weights, self.rrf_k)
        if self.reranker is not None:
            return self.reranker.rerank(query, fused[: self.rerank_candidates], top_k)
        return fused[:top_k]
