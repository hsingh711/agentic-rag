"""Section 10 evaluation harness: closed-book vs vanilla RAG vs the agentic pipeline.

Datasets (reported separately, never averaged): ``arxiv`` (held-out test split of the classifier
labels), ``hotpotqa`` and ``multihoprag`` (first ``benchmarks.eval_questions`` questions, each in
its own collection).

Systems:
- ``closed_book``: the generator answers with no retrieval;
- ``vanilla``: one retrieval, one generation (the section 4 baseline);
- ``vanilla_hybrid`` / ``vanilla_hybrid_rerank``: the same, with dense + BM25 fusion (and a reranker)
  instead of dense retrieval only (section 11);
- ``agentic``: classifier -> reformulate -> retrieve -> critique loop -> grounded generation with the
  inline-citation prompt (section 9 as first built) -> verify;
- ``agentic_v2``: same, but the short-answer generator with citations attached afterwards (the fix
  made after the first evaluation showed the inline-citation prompt hurting the 3B model);
- ``agentic_no_classifier`` / ``agentic_v2_no_classifier``: always retrieve (classifier ablation).

One record per (dataset, system, question) is appended to ``results/eval/<dataset>_<system>.jsonl``;
re-runs skip finished questions, so a daily-limit stop resumes cleanly. Latency is reported as
model seconds (the generator and Gemini calls themselves), rate-limit wait, and other seconds
(classifier, retrieval, NLI). Wall time of a re-run is not meaningful because LLM replies are cached.
"""

import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.agents.prompts import CLOSED_BOOK_SYSTEM, build_closed_book_prompt
from src.baseline.vanilla_rag import vanilla_rag
from src.config import PROJECT_ROOT
from src.eval.answer_scoring import is_null_question, score_record
from src.eval.benchmarks import benchmark_paths, eval_questions
from src.generation.verifier import verify_answer
from src.graph.pipeline import run_pipeline
from src.ingest.io_utils import append_jsonl, read_jsonl
from src.llm import call_llm

DATASETS = ("arxiv", "hotpotqa", "multihoprag")
SYSTEMS = (
    "closed_book", "vanilla", "vanilla_hybrid", "vanilla_hybrid_rerank",
    "agentic", "agentic_v2", "agentic_no_classifier", "agentic_v2_no_classifier",
)
LOCAL_MODELS = ("ollama_qwen",)


def results_dir(config: Dict[str, Any]) -> Path:
    """Folder of the per-question result files."""
    return PROJECT_ROOT / config["eval"]["results_dir"]


def load_eval_set(dataset: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluation questions of a dataset: ``query_id, query, references, question_type, gold_titles``."""
    if dataset != "arxiv":
        return eval_questions(dataset, config)
    labels = config["labels"]
    pool = {record["query"].strip().lower(): record for record in read_jsonl(PROJECT_ROOT / labels["dir"] / labels["pool_file"])}
    test_path = PROJECT_ROOT / config["classifier"]["splits_dir"] / "test.csv"
    questions = []
    with open(test_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            record = pool.get(row["query"].strip().lower())
            if record is None:
                continue
            questions.append({
                "query_id": record["query_id"],
                "query": record["query"],
                "references": record["references"],
                "question_type": record["source"],
                "gold_titles": [],
                "dataset": "arxiv",
            })
    return questions


def chunk_text_lookup(dataset: str, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """chunk_id -> chunk (title, text, ...) for the collection a dataset's answers were retrieved from."""
    if dataset == "arxiv":
        path = PROJECT_ROOT / config["paths"]["processed_dir"] / config["ingest"]["chunks_file"]
    else:
        path = benchmark_paths(dataset)[1]
    return {chunk["chunk_id"]: chunk for chunk in read_jsonl(path)}


def reverify(config: Dict[str, Any], dataset: str, system: str, scorer: Any) -> int:
    """Recompute scores and claim counts of stored answers with the current scorer and verifier (no LLM calls).

    Used after the verifier changed (the evaluation answers themselves stay as generated).
    Returns how many records were rewritten.
    """
    path = results_dir(config) / f"{dataset}_{system}.jsonl"
    if not path.is_file():
        return 0
    chunks = chunk_text_lookup(dataset, config)
    full_references = {question["query_id"]: question["references"] for question in load_eval_set(dataset, config)}
    threshold = config["labels"]["f1_correct_threshold"]
    records = list(read_jsonl(path))
    for record in records:
        references = full_references.get(record["query_id"], record["references"])
        record.update(score_record(record["answer"], references, threshold))
        contexts = [chunks[chunk_id] for chunk_id in record["retrieved_chunk_ids"] if chunk_id in chunks]
        report = verify_answer(record["answer"], contexts, scorer, config["verifier"], question=record["query"])
        record.update(verification_fields(report))
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(path)
    return len(records)


def gold_recall(contexts: List[Dict[str, Any]], gold_titles: List[str]) -> Optional[float]:
    """Share of the gold document titles present in the contexts (None if the question has no gold titles)."""
    if not gold_titles:
        return None
    present = {chunk.get("title") for chunk in contexts}
    return sum(title in present for title in gold_titles) / len(gold_titles)


def verification_fields(report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Claim counts from a verification report (None-safe)."""
    if not report or report.get("status") != "verified":
        return {"n_claims": 0, "n_supported": 0, "n_contradicted": 0}
    claims = report["claims"]
    return {
        "n_claims": len(claims),
        "n_supported": sum(item["label"] == "supported" for item in claims),
        "n_contradicted": sum(item["label"] == "contradicted" for item in claims),
    }


def trace_fields(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Latency, tokens and call counts summed over a graph trace."""
    model_seconds = wait_seconds = other_seconds = 0.0
    tokens_local = tokens_gemini = lite_calls = 0
    for entry in trace:
        if entry.get("model"):
            model_seconds += entry.get("latency_seconds") or 0.0
            wait_seconds += entry.get("wait_seconds") or 0.0
            tokens = entry.get("tokens") or 0
            if entry["model"] in LOCAL_MODELS:
                tokens_local += tokens
            else:
                tokens_gemini += tokens
                lite_calls += 1
        else:
            other_seconds += entry.get("seconds") or 0.0
    return {
        "model_seconds": model_seconds,
        "wait_seconds": wait_seconds,
        "other_seconds": other_seconds,
        "tokens_local": tokens_local,
        "tokens_gemini": tokens_gemini,
        "lite_calls": lite_calls,
    }


def run_closed_book(question: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    """Closed-book system: no retrieval."""
    result = call_llm(build_closed_book_prompt(question["query"]), system=CLOSED_BOOK_SYSTEM, role="generator")
    return {
        "answer": result.text.strip(), "decision": "skip", "contexts": [],
        "model_seconds": result.latency_seconds, "wait_seconds": result.wait_seconds, "other_seconds": 0.0,
        "tokens_local": result.total_tokens, "tokens_gemini": 0, "lite_calls": 0,
    }


def run_vanilla(question: Dict[str, Any], store: Any, top_k: int, scorer: Any, verifier_settings: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    """Vanilla RAG system; the answer is also NLI-verified so hallucination rates are comparable."""
    result = vanilla_rag(question["query"], store, top_k)
    started = time.perf_counter()
    report = verify_answer(result["answer"], result["contexts"], scorer, verifier_settings, question=question["query"])
    return {
        "answer": result["answer"], "decision": "retrieve", "contexts": result["contexts"], "verification": report,
        "model_seconds": result["generation_latency"], "wait_seconds": 0.0,
        "other_seconds": result["retrieval_latency"] + (time.perf_counter() - started),
        "tokens_local": result["tokens"], "tokens_gemini": 0, "lite_calls": 0,
    }


def run_agentic(question: Dict[str, Any], app: Any, **_: Any) -> Dict[str, Any]:
    """Agentic system: run the compiled graph (verification is a graph node)."""
    state = run_pipeline(app, question["query"])
    decision = next(entry["decision"] for entry in state["trace"] if entry["step"] == "classify")
    generate = next(entry for entry in state["trace"] if entry["step"] == "generate")
    return {
        "answer": state["answer"], "decision": decision,
        "contexts": state.get("answer_contexts", []), "verification": state.get("verification"),
        "retries": state.get("retries", 0), "stop_reason": state.get("stop_reason"),
        "insufficient": bool(generate.get("insufficient") or generate.get("refused")),
        "uncited": bool(generate.get("uncited")),
        **trace_fields(state["trace"]),
    }


def build_record(system: str, dataset: str, question: Dict[str, Any], output: Dict[str, Any], f1_threshold: float) -> Dict[str, Any]:
    """Score one system output and flatten it into a result record."""
    scores = score_record(output["answer"], question["references"], f1_threshold)
    contexts = output.pop("contexts", [])
    verification = output.pop("verification", None)
    return {
        "system": system, "dataset": dataset, "query_id": question["query_id"], "query": question["query"],
        "question_type": question["question_type"], "references": question["references"],
        "null_question": is_null_question(question["references"]),
        "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in contexts],
        # Only meaningful when the system retrieved; a skipped question has no contexts to score.
        "gold_recall": gold_recall(contexts, question.get("gold_titles", [])) if contexts else None,
        **scores, **verification_fields(verification), **output,
    }


def evaluate(
    config: Dict[str, Any],
    dataset: str,
    system: str,
    questions: List[Dict[str, Any]],
    runner: Callable[..., Dict[str, Any]],
    runner_args: Dict[str, Any],
    limit: Optional[int] = None,
) -> int:
    """Run ``system`` on ``questions`` and append a record per new question; return how many were new."""
    path = results_dir(config) / f"{dataset}_{system}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {record["query_id"] for record in read_jsonl(path)}
    todo = [question for question in questions if question["query_id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    threshold = config["labels"]["f1_correct_threshold"]
    for question in todo:
        started = time.perf_counter()
        output = runner(question, **runner_args)
        record = build_record(system, dataset, question, output, threshold)
        record["wall_seconds"] = time.perf_counter() - started
        append_jsonl(path, [record])
    return len(todo)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def mean(values: List[float]) -> Optional[float]:
    """Mean of a list, None if empty."""
    return sum(values) / len(values) if values else None


def bootstrap_mean_interval(values: List[float], samples: int = 1000, seed: int = 13) -> Optional[List[float]]:
    """95% percentile bootstrap interval of the mean (None for fewer than 2 values)."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(samples))
    return [means[int(0.025 * samples)], means[int(0.975 * samples) - 1]]


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Metrics of one (dataset, system) result list."""
    answerable = [record for record in records if not record["null_question"]]
    null = [record for record in records if record["null_question"]]
    claims = sum(record["n_claims"] for record in records)
    supported = sum(record["n_supported"] for record in records)
    with_gold = [record["gold_recall"] for record in records if record.get("gold_recall") is not None]
    relaxed = [float(record["relaxed"]) for record in records]
    return {
        "n": len(records),
        "n_null_questions": len(null),
        "strict_accuracy": mean([float(record["strict"]) for record in records]),
        "relaxed_accuracy": mean(relaxed),
        "relaxed_accuracy_ci95": bootstrap_mean_interval(relaxed),
        "answerable_relaxed_accuracy": mean([float(record["relaxed"]) for record in answerable]),
        "null_abstention_rate": mean([float(record["abstained"]) for record in null]),
        "false_abstention_rate": mean([float(record["abstained"]) for record in answerable]),
        "skip_rate": mean([float(record["decision"] == "skip") for record in records]),
        "gold_title_recall": mean(with_gold),
        "claims": claims,
        "unsupported_claim_rate": (1 - supported / claims) if claims else None,
        "contradicted_claim_rate": (sum(record["n_contradicted"] for record in records) / claims) if claims else None,
        "mean_model_seconds": mean([record["model_seconds"] for record in records]),
        "mean_wait_seconds": mean([record["wait_seconds"] for record in records]),
        "mean_other_seconds": mean([record["other_seconds"] for record in records]),
        "mean_tokens_local": mean([record["tokens_local"] for record in records]),
        "mean_tokens_gemini": mean([record["tokens_gemini"] for record in records]),
        "mean_lite_calls": mean([record["lite_calls"] for record in records]),
        "by_question_type": {
            question_type: {
                "n": len(group),
                "relaxed_accuracy": mean([float(record["relaxed"]) for record in group]),
            }
            for question_type in sorted({record["question_type"] for record in records})
            for group in [[record for record in records if record["question_type"] == question_type]]
        },
    }


def paired_difference(records: List[Dict[str, Any]], baseline: List[Dict[str, Any]], samples: int = 2000, seed: int = 13) -> Dict[str, Any]:
    """Relaxed-accuracy difference (records minus baseline) over the questions both answered.

    Paired bootstrap over questions gives the 95% interval; ``wins`` / ``losses`` count questions
    only one of the two systems got right. An interval that contains 0 means the difference is
    within noise at this sample size.
    """
    base_by_id = {record["query_id"]: record for record in baseline}
    pairs = [(record["relaxed"], base_by_id[record["query_id"]]["relaxed"]) for record in records if record["query_id"] in base_by_id]
    if not pairs:
        return {"n": 0}
    differences = [float(first - second) for first, second in pairs]
    rng = random.Random(seed)
    means = sorted(mean([differences[rng.randrange(len(differences))] for _ in differences]) for _ in range(samples))
    return {
        "n": len(pairs),
        "difference": mean(differences),
        "ci95": [means[int(0.025 * samples)], means[int(0.975 * samples) - 1]],
        "wins": sum(1 for first, second in pairs if first and not second),
        "losses": sum(1 for first, second in pairs if second and not first),
    }


def summarize_all(config: Dict[str, Any]) -> Dict[str, Any]:
    """Summaries for every (dataset, system) result file that exists, keyed ``dataset -> system``."""
    summary: Dict[str, Any] = {}
    for dataset in DATASETS:
        for system in SYSTEMS:
            path = results_dir(config) / f"{dataset}_{system}.jsonl"
            if path.is_file():
                records = list(read_jsonl(path))
                if records:
                    summary.setdefault(dataset, {})[system] = summarize(records)
                    if system not in ("vanilla", "closed_book"):
                        vanilla_path = results_dir(config) / f"{dataset}_vanilla.jsonl"
                        if vanilla_path.is_file():
                            summary[dataset][system]["paired_vs_vanilla"] = paired_difference(records, list(read_jsonl(vanilla_path)))
    return summary


def save_summary(config: Dict[str, Any], summary: Dict[str, Any]) -> Path:
    """Write the summary JSON next to the result files."""
    path = results_dir(config) / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
