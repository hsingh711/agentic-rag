"""Run the agentic pipeline (classifier -> reformulate -> retrieve -> critique loop -> generate)
on the demo questions and print a readable trace for each.

Needs the Qdrant server (scripts/start_qdrant.ps1) and Ollama (generator) running.
Calls are cached, so re-running costs no quota.

Run from the project root:
    python scripts/run_graph.py
    python scripts/run_graph.py --question "your question" --no-classifier
    python scripts/run_graph.py --compare-early-stop   # same questions with early stop on and off
    python scripts/run_graph.py --compare-multipart    # two-part questions: one call vs per-part answers
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src.agents.critic import judge  # noqa: E402
from src.agents.reformulator import reformulate  # noqa: E402
from src.classifier.predict import should_retrieve  # noqa: E402
from src.classifier.scoring import score_answer  # noqa: E402
from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.eval.critic_eval import build_proxy_rows, sample_balanced  # noqa: E402
from src.generation.verifier import NLIScorer, verify_answer  # noqa: E402
from src.graph.pipeline import build_graph, run_pipeline  # noqa: E402
from src.ingest.io_utils import read_jsonl  # noqa: E402
from src.index.embedder import Embedder  # noqa: E402
from src.index.qdrant_store import QdrantStore, make_client  # noqa: E402
from src.llm import DailyLimitReached  # noqa: E402


def format_entry(entry: Dict[str, Any]) -> str:
    """One trace entry as one or more readable lines."""
    step = entry["step"]
    if step == "classify":
        return f"  classify     -> {entry['decision']} (confidence {entry['confidence']}, {entry['seconds'] * 1000:.0f} ms)"
    timing = ""
    if entry.get("latency_seconds") is not None:
        timing = f" [{entry.get('model')}, {entry['latency_seconds']:.1f}s{', cached' if entry.get('cached') else ''}]"
    if step == "reformulate":
        lines = [f"  reformulate  -> {entry['mode']} (retry {entry['retry']}){timing}"]
        lines += [f"                  - {query}" for query in entry["queries"]]
        return "\n".join(lines)
    if step == "retrieve":
        return (f"  retrieve     -> {entry['hits']} hits, {entry['new_chunks']} new, "
                f"{entry['total_chunks']} unique so far ({entry['seconds'] * 1000:.0f} ms)")
    if step == "critique":
        line = f"  critique     -> {entry['verdict']} (confidence {entry['confidence']}){timing}"
        if entry.get("stop_reason"):
            line += f"  STOP: {entry['stop_reason']}"
        if entry["missing"]:
            line += f"\n                  missing: {entry['missing']}"
        return line
    if step == "verify":
        if entry["status"] != "verified":
            return f"  verify       -> {entry['status']}"
        lines = [f"  verify       -> {entry['n_supported']}/{entry['n_claims']} claims supported ({entry['seconds']:.1f}s)"]
        lines += [f"                  [{item['label']} {item['entailment']:.2f}] {item['claim'][:100]}" for item in entry["claims"]]
        return "\n".join(lines)
    if entry.get("per_part"):
        lines = [f"  generate     -> per part, {len(entry['chunks_used'])} chunks{timing}"]
        lines += [f"                  {line}" for line in entry["answer"].split("\n")]
        return "\n".join(lines)
    answer = entry["answer"].replace("\n", " ")
    return f"  generate     -> {len(entry['chunks_used'])} chunks{timing}\n                  answer: {answer[:300]}"


def count_lite_calls(trace: List[Dict[str, Any]]) -> int:
    """Reformulator and critic calls in one trace (each is one Gemini request when not cached)."""
    return sum(1 for entry in trace if entry["step"] in ("reformulate", "critique") and entry.get("model"))


def compare_early_stop(store: Any, config: Dict[str, Any]) -> None:
    """Run proxy-labelled section 5 questions with early stop on and off; compare calls and accuracy.

    Retrieval is forced (classifier off) because every sampled question is about the corpus.
    """
    settings = config["graph"]["compare_early_stop"]
    labels = config["labels"]
    details = read_jsonl(PROJECT_ROOT / labels["dir"] / labels["details_file"])
    rows = sample_balanced(build_proxy_rows(details, config["critic"]["eval_sources"]), settings["per_class"], settings["seed"])
    references = {query["query_id"]: query["references"] for query in read_jsonl(PROJECT_ROOT / labels["dir"] / labels["pool_file"])}

    # "off" = no early stop at all; "on" = both rules (the most aggressive setting).
    apps = {}
    for name, enabled in (("off", False), ("on", True)):
        early_stop = dict(config["graph"]["early_stop"], no_new_chunks=enabled, same_missing=enabled)
        apps[name] = build_graph(
            store, classifier=should_retrieve, reformulate_fn=reformulate, judge_fn=judge,
            use_classifier=False, early_stop=early_stop,
        )

    records = []
    for number, row in enumerate(rows, start=1):
        record = {"query": row["query"], "proxy_label": row["label"]}
        for name, app in apps.items():
            state = run_pipeline(app, row["query"])
            correct, _ = score_answer(state["answer"], references[row["query_id"]], labels["f1_correct_threshold"])
            record[name] = {
                "answer": state["answer"],
                "correct": correct,
                "first_mode": state.get("first_mode"),
                "retries": state.get("retries", 0),
                "lite_calls": count_lite_calls(state["trace"]),
                "stop_reason": state.get("stop_reason"),
            }
        records.append(record)
        print(f"{number:2d}. [{row['label']:14s}] calls off {record['off']['lite_calls']} / on {record['on']['lite_calls']} "
              f"({record['on']['stop_reason']}), correct off {record['off']['correct']} / on {record['on']['correct']}  {row['query'][:70]}")

    summary = {}
    for name in ("off", "on"):
        summary[name] = {
            "lite_calls": sum(record[name]["lite_calls"] for record in records),
            "correct": sum(record[name]["correct"] for record in records),
        }
    changed = sum(1 for record in records if record["off"]["answer"] != record["on"]["answer"])
    print(f"\n{len(records)} questions. Lite calls: off {summary['off']['lite_calls']}, on {summary['on']['lite_calls']}. "
          f"Correct: off {summary['off']['correct']}, on {summary['on']['correct']}. Answers changed: {changed}.")

    output_path = PROJECT_ROOT / settings["output"]
    output_path.write_text(json.dumps({"summary": summary, "answers_changed": changed, "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {output_path}")


def join_questions(first: str, second: str) -> str:
    """Join two questions into one two-part question ("A, and b?")."""
    second = second.strip()
    if len(second) > 1 and second[1].islower():
        second = second[0].lower() + second[1:]
    return f"{first.strip().rstrip('?').rstrip()}, and {second}"


def make_question_pairs(config: Dict[str, Any], pairs: int, seed: int) -> List[Dict[str, Any]]:
    """Pair answerable section 5 arXiv questions from different papers into two-part questions."""
    labels = config["labels"]
    details = read_jsonl(PROJECT_ROOT / labels["dir"] / labels["details_file"])
    answerable = [row for row in details if row["source"] == "arxiv_chunk" and row["label"] == "retrieve"]
    references = {query["query_id"]: query["references"] for query in read_jsonl(PROJECT_ROOT / labels["dir"] / labels["pool_file"])}
    random.Random(seed).shuffle(answerable)

    def paper_of(row: Dict[str, Any]) -> str:
        return row["query_id"].split(":")[1]

    result = []
    used_papers = set()
    waiting = None
    for row in answerable:
        if paper_of(row) in used_papers:
            continue
        used_papers.add(paper_of(row))
        if waiting is None:
            waiting = row
            continue
        result.append({
            "question": join_questions(waiting["query"], row["query"]),
            "parts": [waiting["query"], row["query"]],
            "references": [references[waiting["query_id"]], references[row["query_id"]]],
        })
        waiting = None
        if len(result) >= pairs:
            break
    return result


def compare_multipart(store: Any, config: Dict[str, Any]) -> None:
    """Answer two-part questions with one generation call vs per-part generation; score each part."""
    settings = config["graph"]["compare_multipart"]
    threshold = config["labels"]["f1_correct_threshold"]
    questions = make_question_pairs(config, settings["pairs"], settings["seed"])

    apps = {}
    for name, per_part in (("single", False), ("per_part", True)):
        apps[name] = build_graph(
            store, classifier=should_retrieve, reformulate_fn=reformulate, judge_fn=judge,
            use_classifier=False, per_part_generation=per_part,
        )

    records = []
    for number, item in enumerate(questions, start=1):
        record = dict(item)
        for name, app in apps.items():
            state = run_pipeline(app, item["question"])
            parts_correct = [score_answer(state["answer"], refs, threshold)[0] for refs in item["references"]]
            record[name] = {"answer": state["answer"], "parts_correct": parts_correct, "first_mode": state.get("first_mode")}
        records.append(record)
        print(f"{number:2d}. mode {record['per_part']['first_mode']:9s} single {record['single']['parts_correct']} "
              f"per_part {record['per_part']['parts_correct']}  {item['question'][:90]}")

    summary = {}
    for name in apps:
        part_scores = [score for record in records for score in record[name]["parts_correct"]]
        summary[name] = {
            "parts_correct": sum(part_scores),
            "parts_total": len(part_scores),
            "both_correct": sum(1 for record in records if all(record[name]["parts_correct"])),
        }
    decomposed = sum(1 for record in records if record["per_part"]["first_mode"] == "decompose")
    print(f"\n{len(records)} two-part questions, {decomposed} decomposed by the reformulator.")
    for name, values in summary.items():
        print(f"  {name:9s}: parts correct {values['parts_correct']}/{values['parts_total']}, "
              f"both parts correct {values['both_correct']}/{len(records)}")

    output_path = PROJECT_ROOT / settings["output"]
    output_path.write_text(
        json.dumps({"summary": summary, "decomposed": decomposed, "records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {output_path}")


def main() -> None:
    """Build the graph with the real components and run the demo questions."""
    parser = argparse.ArgumentParser(description="Run the agentic RAG graph and print traces.")
    parser.add_argument("--question", action="append", help="Question to run (repeatable). Default: graph.demo_questions.")
    parser.add_argument("--no-classifier", action="store_true", help="Always retrieve (classifier off).")
    parser.add_argument("--no-verify", action="store_true", help="Skip the NLI answer verification step.")
    parser.add_argument("--compare-early-stop", action="store_true", help="Compare early stop on/off on proxy-labelled questions.")
    parser.add_argument("--compare-multipart", action="store_true", help="Compare single vs per-part answers on two-part questions.")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config()
    vector_config = config["vector_store"]
    store = QdrantStore(
        make_client(vector_config), config["graph"]["collection"], Embedder.from_config(config), vector_config,
        use_quantization=False,
    )
    if args.compare_early_stop or args.compare_multipart:
        try:
            if args.compare_early_stop:
                compare_early_stop(store, config)
            if args.compare_multipart:
                compare_multipart(store, config)
        except DailyLimitReached as error:
            print(f"\nStopped: {error}. Re-run tomorrow (finished calls are cached).")
        return

    verify_fn = None
    if config["verifier"]["enabled"] and not args.no_verify:
        scorer = NLIScorer.from_config(config)
        verify_fn = lambda answer, contexts, question=None: verify_answer(answer, contexts, scorer, question=question)  # noqa: E731
    app = build_graph(
        store, classifier=should_retrieve, reformulate_fn=reformulate, judge_fn=judge,
        use_classifier=False if args.no_classifier else None, verify_fn=verify_fn,
    )

    questions = args.question or config["graph"]["demo_questions"]
    records = []
    try:
        for number, question in enumerate(questions, start=1):
            state = run_pipeline(app, question)
            print(f"\n=== Q{number}: {question}")
            for entry in state["trace"]:
                print(format_entry(entry))
            print(f"  total {state['total_seconds']:.1f}s, retries {state.get('retries', 0)}, "
                  f"lite calls {count_lite_calls(state['trace'])}")
            records.append({
                "question": question,
                "answer": state["answer"],
                "verification": state.get("verification"),
                "retries": state.get("retries", 0),
                "total_seconds": state["total_seconds"],
                "trace": state["trace"],
            })
    except DailyLimitReached as error:
        print(f"\nStopped: {error}. Results so far are saved; re-run tomorrow (finished calls are cached).")

    output_path = PROJECT_ROOT / config["generation"]["demo_output"]
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
