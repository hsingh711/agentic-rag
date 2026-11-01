"""Section 11: retrieval quality of dense, BM25, hybrid and hybrid + reranker (no LLM calls).

Gold evidence per dataset:
- HotpotQA / MultiHop-RAG: the titles of the supporting documents (``gold_titles``);
- arXiv: the chunk the question was written from (``source_chunk_id``).

Metrics at top-k: ``hit`` (any gold found), ``gold_recall`` (share of the gold found, averaged over
questions) and ``all_gold`` (every gold item found; this is what a multi-hop answer needs).
Weights and other choices are made on development questions, never on the evaluation questions.
"""

import statistics
import time
from typing import Any, Callable, Dict, List

from src.config import PROJECT_ROOT
from src.eval.benchmarks import benchmark_paths
from src.eval.harness import load_eval_set
from src.ingest.io_utils import read_jsonl


def gold_items(question: Dict[str, Any], dataset: str) -> List[str]:
    """The gold evidence ids of a question (titles, or the source chunk_id for arXiv)."""
    if dataset == "arxiv":
        return [question["source_chunk_id"]] if question.get("source_chunk_id") else []
    return list(question.get("gold_titles", []))


def found_items(hits: List[Dict[str, Any]], dataset: str) -> set:
    """What the retrieved chunks cover: their titles, or their chunk_ids for arXiv."""
    key = "chunk_id" if dataset == "arxiv" else "title"
    return {hit.get(key) for hit in hits}


def question_metrics(hits: List[Dict[str, Any]], gold: List[str], dataset: str) -> Dict[str, float]:
    """hit / gold_recall / all_gold of one question."""
    covered = found_items(hits, dataset)
    found = sum(item in covered for item in gold)
    return {"hit": float(found > 0), "gold_recall": found / len(gold), "all_gold": float(found == len(gold))}


def load_questions(dataset: str, config: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    """Evaluation (``eval``) or development (``dev``) questions that have gold evidence.

    dev = benchmark questions after the first ``benchmarks.eval_questions``, or (arXiv) generated
    questions that are not in the evaluation set.
    """
    if dataset == "arxiv":
        labels = config["labels"]
        pool = list(read_jsonl(PROJECT_ROOT / labels["dir"] / labels["pool_file"]))
        by_id = {record["query_id"]: record for record in pool}
        evaluation = [dict(question, source_chunk_id=by_id[question["query_id"]].get("source_chunk_id"))
                      for question in load_eval_set("arxiv", config)]
        if split == "eval":
            questions = evaluation
        else:
            taken = {question["query_id"] for question in evaluation}
            questions = [record for record in pool if record["query_id"] not in taken and record.get("source_chunk_id")][:200]
    else:
        everything = list(read_jsonl(benchmark_paths(dataset)[0]))
        limit = config["benchmarks"]["eval_questions"]
        questions = everything[:limit] if split == "eval" else everything[limit:]
    return [question for question in questions if gold_items(question, dataset)]


def evaluate_retriever(
    search: Callable[[str, int], List[Dict[str, Any]]], questions: List[Dict[str, Any]], dataset: str, top_k: int
) -> Dict[str, Any]:
    """Mean metrics and median latency (ms) of a ``search(query, top_k)`` function."""
    rows, timings = [], []
    for question in questions:
        started = time.perf_counter()
        hits = search(question["query"], top_k)
        timings.append((time.perf_counter() - started) * 1000)
        rows.append(question_metrics(hits, gold_items(question, dataset), dataset))
    count = len(rows)
    return {
        "n": count,
        "hit": sum(row["hit"] for row in rows) / count,
        "gold_recall": sum(row["gold_recall"] for row in rows) / count,
        "all_gold": sum(row["all_gold"] for row in rows) / count,
        "median_ms": statistics.median(timings),
    }
