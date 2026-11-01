"""Section 11 vector benchmark: FAISS exact search vs Qdrant HNSW, fp32 vs int8.

The question this answers is the usual production trade-off: how much search quality does an
approximate, quantized index cost, and how much latency does it save? FAISS ``IndexFlatIP`` is
the exact baseline (brute force over every vector, so its top-k IS the ground truth, by
definition recall 1.0). The two Qdrant collections hold the same vectors with an HNSW graph,
one storing float32 vectors and one int8 scalar-quantized, and are measured at several
``hnsw_ef`` settings.

Metrics per configuration:
- ``recall_at_k``: mean share of the exact top-k that the configuration returned;
- ``mean_overlap_ids``: the same thing counted as ids, for sanity;
- latency p50 / p90 / p99 and mean, in milliseconds, one query at a time (no batching), which
  is how the RAG pipeline searches. Two latencies are reported: what the client waits for
  (search plus HTTP round trip, which is what the pipeline actually pays) and the search time
  the Qdrant server reports for the same request. On this Windows machine the loopback round
  trip is about 20 ms and swamps the search itself, so only the server time can separate the
  configurations; FAISS runs in-process and has no comparable overhead;
- the collection's size on disk (Qdrant preallocates about 265 MB of segment files per
  collection, so this number only becomes informative on a corpus of this size).

All vectors are cosine-normalised, so inner product and cosine similarity rank identically and
FAISS ``IndexFlatIP`` is the right exact index.
"""

import json
import logging
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.config import PROJECT_ROOT
from src.index.qdrant_store import QdrantStore
from src.ingest.io_utils import read_jsonl

logger = logging.getLogger(__name__)


def load_bench_queries(config: Dict[str, Any]) -> List[str]:
    """Benchmark query strings: a seeded sample of the arXiv-style questions in the query pool.

    These are real questions over arXiv papers, so query vectors land where the corpus is dense.
    They were written from the dev corpus, which does not matter here: recall is measured against
    exact search over the SAME corpus, and latency does not depend on where the answer is.
    """
    bench_config = config["bench"]
    pool_path = PROJECT_ROOT / config["labels"]["dir"] / config["labels"]["pool_file"]
    sources = set(bench_config["query_sources"])
    queries = [
        record["query"]
        for record in read_jsonl(pool_path)
        if record["source"] in sources and record["query"].strip()
    ]
    rng = random.Random(bench_config["seed"])
    rng.shuffle(queries)
    return queries[: bench_config["n_queries"]]


def latency_summary(timings_ms: Sequence[float]) -> Dict[str, float]:
    """Mean and percentile latencies of a list of per-query millisecond timings."""
    ordered = sorted(timings_ms)
    return {
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": ordered[int(0.50 * (len(ordered) - 1))],
        "p90_ms": ordered[int(0.90 * (len(ordered) - 1))],
        "p99_ms": ordered[int(0.99 * (len(ordered) - 1))],
    }


def exact_search(vectors: np.ndarray, query_vectors: np.ndarray, top_k: int, repeats: int = 1) -> Dict[str, Any]:
    """Exact top-k with FAISS ``IndexFlatIP`` (brute force): the ground truth for recall.

    Args:
        vectors: Corpus vectors, shape (n, dimension), normalised float32. A memmap is copied
            into RAM because FAISS needs a contiguous array.
        query_vectors: Query vectors, shape (q, dimension).
        top_k: Neighbours per query.
        repeats: Timed repeats per query (the fastest run of each query is kept).

    Returns:
        ``{"neighbours": list of row-id lists, "scores": ..., "latency": {...}, "build_seconds": float}``.
    """
    import faiss

    corpus = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    build_started = time.perf_counter()
    index = faiss.IndexFlatIP(corpus.shape[1])
    index.add(corpus)
    build_seconds = time.perf_counter() - build_started

    neighbours: List[List[int]] = []
    scores: List[List[float]] = []
    timings: List[float] = []
    for query_vector in query_vectors:
        single = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        best = None
        for _ in range(max(1, repeats)):
            started = time.perf_counter()
            hit_scores, hit_ids = index.search(single, top_k)
            elapsed = (time.perf_counter() - started) * 1000
            best = elapsed if best is None else min(best, elapsed)
        timings.append(best)
        neighbours.append([int(row) for row in hit_ids[0]])
        scores.append([float(score) for score in hit_scores[0]])

    return {
        "neighbours": neighbours,
        "scores": scores,
        "latency": latency_summary(timings),
        "build_seconds": build_seconds,
        "n_vectors": int(corpus.shape[0]),
    }


def qdrant_search(
    store: QdrantStore, query_vectors: np.ndarray, top_k: int, repeats: int = 1
) -> Dict[str, Any]:
    """Search a Qdrant collection query by query and record chunk_ids plus latencies.

    One warm-up query is run first so the measurement excludes the first HTTP connection.
    """
    store.search_by_vector(query_vectors[0], top_k)
    chunk_ids: List[List[str]] = []
    timings: List[float] = []
    for query_vector in query_vectors:
        best = None
        hits: List[Dict[str, Any]] = []
        for _ in range(max(1, repeats)):
            started = time.perf_counter()
            hits = store.search_by_vector(query_vector, top_k)
            elapsed = (time.perf_counter() - started) * 1000
            best = elapsed if best is None else min(best, elapsed)
        timings.append(best)
        chunk_ids.append([hit["chunk_id"] for hit in hits])
    return {"chunk_ids": chunk_ids, "latency": latency_summary(timings)}


def qdrant_search_rest(
    url: str, collection: str, query_vectors: np.ndarray, top_k: int, hnsw_ef: int, repeats: int = 1
) -> Dict[str, Any]:
    """Search over plain HTTP so the server's own timing can be read from the response.

    The Qdrant REST API returns a ``time`` field (seconds the server spent on the request)
    next to the results. Comparing it with the time the client waited separates the search
    cost from the loopback HTTP overhead, which is the larger number on this machine.

    Returns:
        ``{"chunk_ids", "latency", "server_latency"}``; both latencies are millisecond summaries.
    """
    import requests

    endpoint = f"{url.rstrip('/')}/collections/{collection}/points/query"
    session = requests.Session()

    def one_query(vector: np.ndarray) -> Dict[str, Any]:
        """Run one query; return its ids, the client-side time and the server-side time."""
        body = {
            "query": np.asarray(vector).tolist(),
            "limit": top_k,
            "params": {"hnsw_ef": hnsw_ef},
            "with_payload": ["chunk_id"],
        }
        started = time.perf_counter()
        response = session.post(endpoint, json=body, timeout=60)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        payload = response.json()
        return {
            "ids": [point["payload"]["chunk_id"] for point in payload["result"]["points"]],
            "client_ms": elapsed_ms,
            "server_ms": payload["time"] * 1000,
        }

    one_query(query_vectors[0])   # warm-up: the first connection is not measured
    chunk_ids: List[List[str]] = []
    client_timings: List[float] = []
    server_timings: List[float] = []
    for query_vector in query_vectors:
        attempts = [one_query(query_vector) for _ in range(max(1, repeats))]
        fastest = min(attempts, key=lambda attempt: attempt["client_ms"])
        chunk_ids.append(fastest["ids"])
        client_timings.append(fastest["client_ms"])
        server_timings.append(min(attempt["server_ms"] for attempt in attempts))
    session.close()
    return {
        "chunk_ids": chunk_ids,
        "latency": latency_summary(client_timings),
        "server_latency": latency_summary(server_timings),
    }


def recall_at_k(approximate: Sequence[Sequence[str]], exact: Sequence[Sequence[str]]) -> Dict[str, float]:
    """Mean share of each query's exact top-k that the approximate result also returned.

    Both arguments are lists of chunk_id lists, one per query, in the same query order.
    """
    shares = []
    found = 0
    expected = 0
    for approximate_ids, exact_ids in zip(approximate, exact):
        exact_set = set(exact_ids)
        overlap = sum(1 for chunk_id in approximate_ids if chunk_id in exact_set)
        shares.append(overlap / len(exact_set) if exact_set else 0.0)
        found += overlap
        expected += len(exact_set)
    return {
        "recall_at_k": statistics.fmean(shares) if shares else 0.0,
        "ids_found": found,
        "ids_expected": expected,
    }


def collection_disk_mb(collection_name: str, storage_dir: Optional[Path] = None) -> Optional[float]:
    """Size of a collection's folder in the Qdrant storage directory, in megabytes.

    Returns None when the folder cannot be found (for example with a remote server).
    """
    base = storage_dir or (PROJECT_ROOT / "data" / "qdrant_storage" / "collections")
    folder = base / collection_name
    if not folder.is_dir():
        return None
    total = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
    return total / 1e6


def benchmark(
    vectors: np.ndarray,
    chunk_ids: Sequence[str],
    query_vectors: np.ndarray,
    stores: Dict[str, QdrantStore],
    top_k: int,
    ef_values: Sequence[int],
    repeats: int = 1,
    rest_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Run FAISS exact search and every (collection, ef_search) combination.

    Args:
        vectors: Corpus vectors (row i belongs to ``chunk_ids[i]``).
        chunk_ids: chunk_id per vector row, used to compare FAISS row ids with Qdrant payloads.
        query_vectors: Embedded benchmark queries.
        stores: ``name -> QdrantStore``; each store's config is copied per ef value.
        top_k: Neighbours per query.
        ef_values: ``hnsw_ef`` values to measure.
        repeats: Timed repeats per query.
        rest_url: Qdrant HTTP url. When given, searches go over plain HTTP so the server's own
            search time is recorded too; without it only the client-side latency is known (the
            local in-memory client used by the tests reports no server time).

    Returns:
        ``{"exact": {...}, "approximate": {"<name>@ef<N>": {...}}}``.
    """
    exact = exact_search(vectors, query_vectors, top_k, repeats)
    exact_chunk_ids = [[chunk_ids[row] for row in rows] for rows in exact["neighbours"]]

    results: Dict[str, Any] = {
        "exact": {
            "engine": "faiss_flat_ip",
            "n_vectors": exact["n_vectors"],
            "build_seconds": exact["build_seconds"],
            "recall_at_k": 1.0,
            **exact["latency"],
        },
        "approximate": {},
    }

    for name, store in stores.items():
        for ef_value in ef_values:
            if rest_url:
                measured = qdrant_search_rest(
                    rest_url, store.collection_name, query_vectors, top_k, ef_value, repeats
                )
            else:
                search_config = dict(store.config)
                search_config["hnsw_ef_search"] = ef_value
                tuned = QdrantStore(
                    store.client, store.collection_name, store.embedder, search_config, store.use_quantization
                )
                measured = qdrant_search(tuned, query_vectors, top_k, repeats)
            key = f"{name}@ef{ef_value}"
            row = {
                "collection": store.collection_name,
                "quantization": "int8" if store.use_quantization else "none",
                "hnsw_ef": ef_value,
                "disk_mb": collection_disk_mb(store.collection_name),
                **recall_at_k(measured["chunk_ids"], exact_chunk_ids),
                **measured["latency"],
            }
            if "server_latency" in measured:
                row.update({f"server_{field}": value for field, value in measured["server_latency"].items()})
            results["approximate"][key] = row
            logger.info(
                "%s: recall@%d %.3f, client p50 %.2f ms, server p50 %s",
                key, top_k, row["recall_at_k"], row["p50_ms"],
                f"{row['server_p50_ms']:.2f} ms" if "server_p50_ms" in row else "n/a",
            )
    return results


def format_table(results: Dict[str, Any], top_k: int) -> str:
    """Markdown table of the benchmark results (exact row first).

    The server-time columns only appear when the measurements came from the HTTP path.
    """
    rows = results["approximate"]
    with_server = any("server_p50_ms" in row for row in rows.values())
    server_header = " server p50 ms | server p90 ms |" if with_server else ""
    server_separator = "---|---|" if with_server else ""
    header = (
        f"| configuration | recall@{top_k} | client mean ms | client p50 ms | client p99 ms |"
        f"{server_header} disk MB |\n"
        f"|---|---|---|---|---|{server_separator}---|"
    )
    exact = results["exact"]
    exact_server = " in-process | in-process |" if with_server else ""
    lines = [
        header,
        f"| FAISS flat (exact, {exact['n_vectors']} vectors) | 1.000 | {exact['mean_ms']:.2f} | "
        f"{exact['p50_ms']:.2f} | {exact['p99_ms']:.2f} |{exact_server} - |",
    ]
    for name, row in rows.items():
        disk = "-" if row["disk_mb"] is None else f"{row['disk_mb']:.0f}"
        server_cells = ""
        if with_server:
            server_cells = (
                f" {row['server_p50_ms']:.2f} | {row['server_p90_ms']:.2f} |"
                if "server_p50_ms" in row else " - | - |"
            )
        lines.append(
            f"| {name} | {row['recall_at_k']:.3f} | {row['mean_ms']:.2f} | {row['p50_ms']:.2f} | "
            f"{row['p99_ms']:.2f} |{server_cells} {disk} |"
        )
    return "\n".join(lines)


def save_results(config: Dict[str, Any], results: Dict[str, Any], suffix: str = "") -> Path:
    """Write the benchmark results JSON to the path in ``bench.output``.

    Args:
        suffix: Appended to the file stem, so runs on different corpora do not overwrite
            each other (``vector_benchmark_dev.json`` next to ``vector_benchmark.json``).
    """
    path = PROJECT_ROOT / config["bench"]["output"]
    if suffix:
        path = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path
