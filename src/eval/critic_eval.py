"""Evaluate the sufficiency critic against proxy labels built from section 5 data (no labelling calls).

For every generated question in ``necessity_details.jsonl`` we know the 5 chunks that were
retrieved for it, whether its source chunk was among them, whether the RAG answer was correct
and whether a reference answer appears in the chunk texts. Proxy label:

- sufficient: source chunk retrieved AND RAG answer correct AND answer text in the chunks;
- not_sufficient: source chunk NOT retrieved AND RAG answer wrong AND answer text not in the chunks;
- mixed cases are skipped.

The labels are a proxy (decided by an experiment, not by a person), so the report says so.

These two groups are the easy extremes (exact source chunk present vs. nothing relevant). A harder
set of near-miss negatives is built from "sufficient" rows: the source chunk is removed and the
other 4 retrieved chunks (same topic, retrieved for the same question) are kept, provided none of
them contains a reference answer. The correct verdict for those is not_sufficient.
"""

import random
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.classifier.make_labels import evidence_contains_reference

LABELS = ["sufficient", "not_sufficient"]


def proxy_label(record: Dict[str, Any]) -> Optional[str]:
    """Return the proxy label of one details record, or None for a mixed case."""
    retrieved = record.get("source_chunk_retrieved")
    correct = record.get("rag_score") == 1
    evidence = bool(record.get("evidence_found"))
    if retrieved is True and correct and evidence:
        return "sufficient"
    if retrieved is False and not correct and not evidence:
        return "not_sufficient"
    return None


def build_proxy_rows(records: Iterable[Dict[str, Any]], sources: List[str]) -> List[Dict[str, Any]]:
    """Keep records from ``sources`` that have a clear proxy label."""
    rows = []
    for record in records:
        if record.get("source") not in sources:
            continue
        label = proxy_label(record)
        if label is None:
            continue
        rows.append(
            {
                "query_id": record["query_id"],
                "query": record["query"],
                "source": record["source"],
                "label": label,
                "chunk_ids": record["retrieved_chunk_ids"],
            }
        )
    return rows


def build_near_miss_rows(
    rows: List[Dict[str, Any]],
    source_chunk_by_query: Dict[str, str],
    references_by_query: Dict[str, List[str]],
    chunks_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Turn "sufficient" rows into not_sufficient near misses by removing their source chunk.

    A row is kept only if the remaining chunks contain none of its reference answers.
    """
    near_misses = []
    for row in rows:
        if row["label"] != "sufficient":
            continue
        source_chunk_id = source_chunk_by_query.get(row["query_id"])
        remaining = [chunk_id for chunk_id in row["chunk_ids"] if chunk_id != source_chunk_id]
        if source_chunk_id is None or len(remaining) == len(row["chunk_ids"]):
            continue
        texts = [chunks_by_id[chunk_id]["text"] for chunk_id in remaining if chunk_id in chunks_by_id]
        if evidence_contains_reference(texts, references_by_query.get(row["query_id"], [])):
            continue
        near_misses.append(dict(row, label="not_sufficient", chunk_ids=remaining, kind="near_miss"))
    return near_misses


def sample_balanced(rows: List[Dict[str, Any]], per_class: int, seed: int) -> List[Dict[str, Any]]:
    """Draw up to ``per_class`` rows of each label (seeded), then shuffle them together."""
    generator = random.Random(seed)
    sample = []
    for label in LABELS:
        of_label = [row for row in rows if row["label"] == label]
        generator.shuffle(of_label)
        sample.extend(of_label[:per_class])
    generator.shuffle(sample)
    return sample


def compute_metrics(labels: List[str], predictions: List[str]) -> Dict[str, Any]:
    """Accuracy, per-class precision/recall/F1 and a confusion matrix (rows true, columns predicted)."""
    total = len(labels)
    correct = sum(1 for label, prediction in zip(labels, predictions) if label == prediction)
    metrics: Dict[str, Any] = {"n": total, "accuracy": correct / total if total else 0.0}

    for label in LABELS:
        true_positive = sum(1 for t, p in zip(labels, predictions) if t == label and p == label)
        predicted = sum(1 for p in predictions if p == label)
        actual = sum(1 for t in labels if t == label)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}

    metrics["confusion"] = [
        [sum(1 for t, p in zip(labels, predictions) if t == true_label and p == predicted_label) for predicted_label in LABELS]
        for true_label in LABELS
    ]
    metrics["confusion_order"] = LABELS
    return metrics


def evaluate_critic(
    rows: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Dict[str, Any]],
    judge_fn: Callable[[str, List[Dict[str, Any]]], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run ``judge_fn`` on every row with its original retrieved chunks; return one result per row.

    Stops early (returning what is done) only if ``judge_fn`` raises; the caller handles that.
    """
    results = []
    for row in rows:
        chunks = [chunks_by_id[chunk_id] for chunk_id in row["chunk_ids"] if chunk_id in chunks_by_id]
        verdict = judge_fn(row["query"], chunks)
        results.append(
            {
                **row,
                "prediction": verdict["verdict"],
                "missing": verdict.get("missing", ""),
                "confidence": verdict.get("confidence"),
                "fallback": verdict.get("fallback", False),
                "model": verdict.get("model"),
                "latency_seconds": verdict.get("latency_seconds"),
            }
        )
    return results
