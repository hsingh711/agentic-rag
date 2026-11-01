"""Label queries "retrieve" or "skip" by experiment, not by opinion.

For every query in the pool, the SAME generator used by the pipeline answers twice:
once closed-book and once with the top_k retrieved chunks. Both answers are scored
against the reference answers (exact match / containment / token F1, no LLM judge).

    closed-book correct                 -> skip     (retrieval adds nothing, or even hurts)
    closed-book wrong, RAG correct      -> retrieve (retrieval measurably helps)
    both wrong                          -> dropped  (ambiguous, not used for training)

Results are appended per batch to a details JSONL (the source of truth, also used to
resume) and the CSV is rewritten from it, so a crash or daily limit never loses work.
"""

import csv
import json
import logging
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.prompts import LABEL_ANSWER_SYSTEM, build_label_closed_book_prompt, build_label_rag_prompt
from src.classifier.scoring import contains_answer, score_answer
from src.ingest.io_utils import append_jsonl, read_jsonl
from src.llm import call_llm

logger = logging.getLogger(__name__)

CSV_COLUMNS = ["query", "label", "closed_book_score", "rag_score", "source"]


def decide_label(closed_book_correct: int, rag_correct: int, evidence_found: bool = True) -> str:
    """Apply the labelling rule (see module docstring).

    ``evidence_found`` says whether a reference answer appears in the retrieved chunks.
    A RAG answer that is correct WITHOUT evidence was a lucky guess (the context only
    changed the wording of the prompt), so it is dropped instead of labelled "retrieve".
    """
    if closed_book_correct:
        return "skip"
    if rag_correct and evidence_found:
        return "retrieve"
    return "dropped"


def evidence_contains_reference(context_texts: List[str], references: List[str]) -> bool:
    """True if any reference answer appears (as whole words) in any retrieved chunk."""
    for text in context_texts:
        for reference in references:
            if contains_answer(text, reference):
                return True
    return False


def label_query(query: Dict[str, Any], store, top_k: int, f1_threshold: float) -> Dict[str, Any]:
    """Run the closed-book vs RAG experiment for one query and return a details record."""
    closed_book = call_llm(
        build_label_closed_book_prompt(query["query"]), system=LABEL_ANSWER_SYSTEM, role="generator"
    )
    contexts = store.search(query["query"], top_k)
    with_retrieval = call_llm(
        build_label_rag_prompt(query["query"], contexts), system=LABEL_ANSWER_SYSTEM, role="generator"
    )

    closed_book_correct, closed_book_f1 = score_answer(closed_book.text, query["references"], f1_threshold)
    rag_correct, rag_f1 = score_answer(with_retrieval.text, query["references"], f1_threshold)
    retrieved_ids = [chunk["chunk_id"] for chunk in contexts]
    evidence_found = evidence_contains_reference([chunk["text"] for chunk in contexts], query["references"])

    return {
        "query_id": query["query_id"],
        "query": query["query"],
        "source": query["source"],
        "references": query["references"][:5],
        "closed_book_answer": closed_book.text.strip(),
        "rag_answer": with_retrieval.text.strip(),
        "closed_book_score": closed_book_correct,
        "rag_score": rag_correct,
        "closed_book_f1": round(closed_book_f1, 3),
        "rag_f1": round(rag_f1, 3),
        "evidence_found": evidence_found,
        "label": decide_label(closed_book_correct, rag_correct, evidence_found),
        "retrieved_chunk_ids": retrieved_ids,
        # Diagnostic for generated questions: did retrieval find the chunk the question came from?
        "source_chunk_retrieved": query.get("source_chunk_id") in retrieved_ids if "source_chunk_id" in query else None,
    }


def ordered_pool(pool_path: Path, seed: int) -> List[Dict[str, Any]]:
    """Load the pool in a fixed shuffled order so any prefix (e.g. a pilot) mixes all sources."""
    pool = list(read_jsonl(pool_path))
    pool.sort(key=lambda record: record["query_id"])
    random.Random(seed).shuffle(pool)
    return pool


def write_csv(details_path: Path, csv_path: Path) -> int:
    """Rewrite the training CSV (labelled rows only, no dropped) from the details file."""
    rows = [record for record in read_jsonl(details_path) if record["label"] != "dropped"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = csv_path.with_suffix(".tmp")
    with open(temp_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in rows:
            writer.writerow({column: record[column] for column in CSV_COLUMNS})
    temp_path.replace(csv_path)
    return len(rows)


def run_labelling(
    config: Dict[str, Any],
    store,
    pool_path: Path,
    details_path: Path,
    csv_path: Path,
    limit: Optional[int] = None,
) -> int:
    """Label every pool query not yet in the details file, in batches.

    Args:
        limit: Stop after this many newly processed queries (pilot runs).

    Returns:
        Number of queries processed in this call.
    """
    labels_config = config["labels"]
    top_k = config["retrieval"]["top_k"]
    batch_size = labels_config["batch_size"]
    f1_threshold = labels_config["f1_correct_threshold"]

    done_ids = {record["query_id"] for record in read_jsonl(details_path)}
    remaining = [query for query in ordered_pool(pool_path, labels_config["seed"]) if query["query_id"] not in done_ids]
    if limit is not None:
        remaining = remaining[:limit]
    logger.info("%d queries already labelled, %d to do in this run.", len(done_ids), len(remaining))

    processed = 0
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start : start + batch_size]
        records = []
        try:
            for query in batch:
                records.append(label_query(query, store, top_k, f1_threshold))
        finally:
            # Save whatever finished, even if a call failed half-way through the batch.
            append_jsonl(details_path, records)
            write_csv(details_path, csv_path)
            processed += len(records)
        logger.info("Labelled %d / %d queries in this run.", processed, len(remaining))
    return processed


def rescore(
    pool_path: Path, details_path: Path, csv_path: Path, f1_threshold: float, chunk_text_by_id: Dict[str, str]
) -> Counter:
    """Re-score stored answers with the current rules (no LLM calls) and rewrite outputs.

    Uses the full reference lists from the pool (the details file keeps only 5) and the
    stored retrieved chunk ids to check whether the evidence contains a reference.

    Returns:
        Counter of (old_label, new_label) pairs that changed.
    """
    references_by_id = {query["query_id"]: query["references"] for query in read_jsonl(pool_path)}
    records = list(read_jsonl(details_path))
    changes: Counter = Counter()
    for record in records:
        references = references_by_id[record["query_id"]]
        closed_book_correct, closed_book_f1 = score_answer(record["closed_book_answer"], references, f1_threshold)
        rag_correct, rag_f1 = score_answer(record["rag_answer"], references, f1_threshold)
        context_texts = [chunk_text_by_id[chunk_id] for chunk_id in record["retrieved_chunk_ids"]]
        evidence_found = evidence_contains_reference(context_texts, references)
        new_label = decide_label(closed_book_correct, rag_correct, evidence_found)
        if new_label != record["label"]:
            changes[(record["label"], new_label)] += 1
        record.update(
            closed_book_score=closed_book_correct,
            rag_score=rag_correct,
            closed_book_f1=round(closed_book_f1, 3),
            rag_f1=round(rag_f1, 3),
            evidence_found=evidence_found,
            label=new_label,
        )

    temp_path = details_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as details_file:
        for record in records:
            details_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(details_path)
    write_csv(details_path, csv_path)
    return changes


# ---------------------------------------------------------------------------
# Estimate and report
# ---------------------------------------------------------------------------


def estimate(config: Dict[str, Any], chunks: List[Dict[str, Any]], pool_path: Path, state: Dict[str, Any], details_path: Path) -> str:
    """Describe the calls, time and quota the full run needs (printed before running)."""
    labels_config = config["labels"]
    llm_config = config["llm"]
    model_keys = llm_config["roles"]["label_generation"]
    if isinstance(model_keys, str):
        model_keys = [model_keys]
    generation_models = [llm_config["models"][key] for key in model_keys]
    # The role rotates over its models, so their limits add up.
    combined_rpm = sum(model["rpm"] for model in generation_models)
    combined_tpm = sum(model["tpm"] for model in generation_models)
    daily_budget = sum(int(model["rpd"] * llm_config["daily_stop_fraction"]) for model in generation_models)
    model_names = ", ".join(model["model_id"] for model in generation_models)

    remaining_chunks = [chunk for chunk in chunks if chunk["chunk_id"] not in set(state["chunks_done"])]
    generation_calls = math.ceil(len(remaining_chunks) / labels_config["chunks_per_call"])
    average_chunk_tokens = sum(len(chunk["text"]) for chunk in chunks) / max(len(chunks), 1) / 4
    tokens_per_call = (
        labels_config["chunks_per_call"] * (average_chunk_tokens + 30)
        + 250
        + labels_config["chunks_per_call"] * labels_config["questions_per_chunk"] * 40
        + 300  # allowance for thinking tokens
    )
    fraction = llm_config["throttle_fraction"]
    generation_minutes = max(
        generation_calls / (combined_rpm * fraction),
        generation_calls * tokens_per_call / (combined_tpm * fraction),
    )

    pool_size = sum(1 for _ in read_jsonl(pool_path))
    if not state["static_done"]:
        pool_size += labels_config["arithmetic_count"] + labels_config["triviaqa_count"] + labels_config["nq_count"]
    expected_chunk_questions = len(remaining_chunks) * labels_config["questions_per_chunk"]
    done = sum(1 for _ in read_jsonl(details_path))
    queries_to_label = max(pool_size + expected_chunk_questions - done, 0)
    ollama_calls = 2 * queries_to_label
    ollama_hours = ollama_calls * labels_config["assumed_seconds_per_ollama_call"] / 3600

    return "\n".join(
        [
            "=== Section 5 estimate (before running) ===",
            f"Question generation ({model_names}): {generation_calls} calls for {len(remaining_chunks)} chunks, "
            f"~{tokens_per_call:,.0f} tokens/call, ~{generation_minutes:.0f} min under the 80% RPM/TPM throttle; "
            f"daily budget {daily_budget} requests (uses {generation_calls / max(daily_budget, 1):.1%}).",
            f"Queries to label: ~{queries_to_label} (pool {pool_size} now + ~{expected_chunk_questions} chunk questions "
            f"before filtering - {done} done).",
            f"Generator calls (local Ollama, free, no daily cap): ~{ollama_calls}, "
            f"~{ollama_hours:.1f} h at {labels_config['assumed_seconds_per_ollama_call']} s/call.",
            "Money cost: $0 (free tier and local model). Notional cost: not configured.",
        ]
    )


def report(details_path: Path, sample_count: int = 10, seed: int = 13) -> str:
    """Class balance, dropped share, per-source breakdown and random samples for manual checking."""
    records = list(read_jsonl(details_path))
    if not records:
        return "No labelled queries yet."
    label_counts = Counter(record["label"] for record in records)
    kept = [record for record in records if record["label"] != "dropped"]

    lines = ["=== Label report ===", f"Processed queries: {len(records)}"]
    for label in ["retrieve", "skip", "dropped"]:
        lines.append(f"  {label:9s} {label_counts[label]:5d}  ({label_counts[label] / len(records):.1%})")
    if kept:
        retrieve_share = label_counts["retrieve"] / len(kept)
        lines.append(f"Balance of kept rows: retrieve {retrieve_share:.1%} / skip {1 - retrieve_share:.1%}")

    lines.append("\nPer source (retrieve / skip / dropped):")
    by_source: Dict[str, Counter] = {}
    for record in records:
        by_source.setdefault(record["source"], Counter())[record["label"]] += 1
    for source, counts in sorted(by_source.items()):
        lines.append(f"  {source:12s} {counts['retrieve']:5d} / {counts['skip']:5d} / {counts['dropped']:5d}")

    chunk_records = [record for record in records if record["source_chunk_retrieved"] is not None]
    if chunk_records:
        hit_rate = sum(record["source_chunk_retrieved"] for record in chunk_records) / len(chunk_records)
        lines.append(f"\nGenerated questions whose source chunk was retrieved (top_k): {hit_rate:.1%}")

    lines.append(f"\n{sample_count} random labelled samples for manual checking:")
    rng = random.Random(seed)
    for record in rng.sample(kept, min(sample_count, len(kept))):
        lines.append(
            f"- [{record['label']}] ({record['source']}) {record['query']}\n"
            f"    reference: {record['references'][0]!r} | closed-book: {record['closed_book_answer'][:80]!r} "
            f"| RAG: {record['rag_answer'][:80]!r}"
        )
    return "\n".join(lines)
