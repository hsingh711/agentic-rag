"""Qdrant collections with an HNSW index and optional int8 scalar quantization.

Use ``mode: server`` in the config for real HNSW and quantization. The qdrant-client
local modes (``memory`` / ``path``) accept the same settings but search by brute force,
so they are only suitable for tests.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient, models

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DISTANCES = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
}


def make_client(vector_store_config: Dict[str, Any]) -> QdrantClient:
    """Create a Qdrant client for the mode named in ``vector_store`` config.

    Server mode prefers gRPC (port ``grpc_port``, 6334 by default). Measured in section 11 on
    the 343k-chunk collection: the same query takes about 2 ms through gRPC and about 28 ms
    through the client's HTTP transport, while the server reports under 1 ms of search time
    either way, so nearly all of the HTTP number is client-side overhead on this machine.
    If the gRPC port cannot be reached, the client falls back to HTTP.
    """
    mode = vector_store_config["mode"]
    if mode == "server":
        url = vector_store_config["url"]
        if vector_store_config.get("prefer_grpc", False):
            client = QdrantClient(
                url=url, grpc_port=vector_store_config.get("grpc_port", 6334), prefer_grpc=True, timeout=120
            )
            try:
                client.get_collections()   # cheap call: fails fast if the gRPC port is closed
                return client
            except Exception as error:     # noqa: BLE001 - any transport failure means "use HTTP"
                logger.warning("gRPC unavailable (%s); falling back to HTTP.", error)
        return QdrantClient(url=url, timeout=120)
    if mode == "path":
        path = PROJECT_ROOT / vector_store_config["path"]
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))
    if mode == "memory":
        return QdrantClient(":memory:")
    raise ValueError(f"Unknown vector_store.mode: {mode!r} (use server, path or memory)")


def chunk_id_to_point_id(chunk_id: str) -> str:
    """Map a chunk_id to a stable UUID (Qdrant ids must be integers or UUIDs).

    The same chunk_id always gives the same UUID, so re-indexing overwrites instead of duplicating.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    """One Qdrant collection plus the embedder used to fill and query it."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedder,
        vector_store_config: Dict[str, Any],
        use_quantization: bool,
    ):
        """Remember the settings; call ``build_index`` to create and fill the collection.

        Args:
            client: A connected QdrantClient.
            collection_name: Name of the collection to create/use.
            embedder: Object with ``dimension``, ``encode(texts)`` and ``encode_query(text)``.
            vector_store_config: The ``vector_store`` section of the config.
            use_quantization: If True, build the collection with int8 scalar quantization.
        """
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder
        self.config = vector_store_config
        self.use_quantization = use_quantization

    def create_collection(self) -> None:
        """Create the collection (deleting any old one with the same name)."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        quantization_config: Optional[models.ScalarQuantization] = None
        if self.use_quantization:
            quantization_config = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=self.config["quantization_quantile"],
                    always_ram=True,
                )
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedder.dimension,
                distance=DISTANCES[self.config["distance"]],
            ),
            hnsw_config=models.HnswConfigDiff(
                m=self.config["hnsw_m"],
                ef_construct=self.config["hnsw_ef_construct"],
            ),
            quantization_config=quantization_config,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=self.config["indexing_threshold_kb"],
            ),
        )

    def build_index(self, chunks: List[Dict[str, Any]], vectors: Optional[np.ndarray] = None) -> float:
        """Create the collection and upload all chunks with their vectors.

        Args:
            chunks: Chunk records (chunk_id, arxiv_id, title, text, source).
            vectors: Optional precomputed embeddings, one row per chunk. If omitted,
                the chunk texts are embedded here. Passing the same vectors to several
                stores avoids embedding twice.

        Returns:
            Seconds spent creating the collection, uploading, and waiting for the index.
        """
        start_time = time.perf_counter()
        if vectors is None:
            vectors = self.embedder.encode([chunk["text"] for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must have the same length")

        self.create_collection()

        batch_size = self.config["upsert_batch_size"]
        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start : batch_start + batch_size]
            points = [
                models.PointStruct(
                    id=chunk_id_to_point_id(chunk["chunk_id"]),
                    vector=vectors[batch_start + offset].tolist(),
                    payload=chunk,
                )
                for offset, chunk in enumerate(batch_chunks)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

        self.wait_until_indexed()
        elapsed = time.perf_counter() - start_time
        logger.info("Built %s with %d points in %.1fs", self.collection_name, len(chunks), elapsed)
        return elapsed

    def wait_until_indexed(self) -> None:
        """Block until the server finished optimizing (HNSW + quantization are built).

        Does nothing in local modes, which have no background optimizer.
        """
        if self.config["mode"] != "server":
            return
        deadline = time.time() + self.config["wait_for_index_seconds"]
        while time.time() < deadline:
            info = self.client.get_collection(self.collection_name)
            if info.status == models.CollectionStatus.GREEN:
                return
            time.sleep(0.5)
        logger.warning("Collection %s was not green after waiting.", self.collection_name)

    def search_by_vector(self, query_vector: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search with an already-embedded query. Returns hits with score and payload fields."""
        search_params = models.SearchParams(hnsw_ef=self.config["hnsw_ef_search"])
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            search_params=search_params,
            with_payload=True,
        )
        hits = []
        for point in response.points:
            hit = dict(point.payload)
            hit["score"] = point.score
            hits.append(hit)
        return hits

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Embed ``query`` and return the ``top_k`` most similar chunks (best first)."""
        return self.search_by_vector(self.embedder.encode_query(query), top_k)

    def describe(self) -> Dict[str, Any]:
        """Return a few facts from the server about how the collection is configured and built."""
        info = self.client.get_collection(self.collection_name)
        return {
            "points": info.points_count,
            "indexed_vectors": info.indexed_vectors_count,
            "status": str(info.status),
            "quantization": info.config.quantization_config,
            "hnsw": info.config.hnsw_config,
        }
