"""Measure the sufficiency critic against proxy labels from section 5 (see src/eval/critic_eval.py).

Costs one critic call per sampled question (``critic.eval_per_class`` x 2); calls are cached,
so re-running is free. Does not need Qdrant or Ollama: the chunks retrieved in section 5 are reused.

Run from the project root:
    python scripts/eval_critic.py
    python scripts/eval_critic.py --model flash_lite_35   # pin the critic to one model
"""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src.agents.critic import judge  # noqa: E402
from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.eval.critic_eval import (  # noqa: E402
    build_near_miss_rows,
    build_proxy_rows,
    compute_metrics,
    evaluate_critic,
    sample_balanced,
)
from src.ingest.io_utils import read_jsonl  # noqa: E402
from src import llm  # noqa: E402
from src.llm import DailyLimitReached, LLMClient  # noqa: E402


def main() -> None:
    """Sample proxy-labelled questions, run the critic, print and save the metrics."""
    parser = argparse.ArgumentParser(description="Evaluate the sufficiency critic on proxy labels.")
    parser.add_argument("--model", default=None, help="Model key to use for the critic instead of llm.roles.critic.")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config()
    settings = config["critic"]
    if args.model:
        # Pin the critic role to one model for this run only (config file unchanged).
        pinned = copy.deepcopy(config)
        pinned["llm"]["roles"]["critic"] = args.model
        llm._default_client = LLMClient(pinned)

    details_path = PROJECT_ROOT / config["labels"]["dir"] / config["labels"]["details_file"]
    chunks_path = PROJECT_ROOT / config["paths"]["processed_dir"] / config["ingest"]["chunks_file"]
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in read_jsonl(chunks_path)}

    rows = build_proxy_rows(read_jsonl(details_path), settings["eval_sources"])
    sample = sample_balanced(rows, settings["eval_per_class"], settings["eval_seed"])
    print(f"Proxy rows: {len(rows)}; sampled {len(sample)}; this costs up to {len(sample)} critic calls (cached calls are free).")

    # Near-miss negatives: "sufficient" rows not in the main sample, with the source chunk removed.
    pool_path = PROJECT_ROOT / config["labels"]["dir"] / config["labels"]["pool_file"]
    pool = list(read_jsonl(pool_path))
    source_chunk_by_query = {query["query_id"]: query.get("source_chunk_id") for query in pool}
    references_by_query = {query["query_id"]: query["references"] for query in pool}
    sampled_ids = {row["query_id"] for row in sample}
    unused = [row for row in rows if row["query_id"] not in sampled_ids]
    near_misses = build_near_miss_rows(unused, source_chunk_by_query, references_by_query, chunks_by_id)
    near_misses = sample_balanced(near_misses, settings["eval_near_miss_count"], settings["eval_seed"])
    print(f"Near-miss negatives sampled: {len(near_misses)} (one more critic call each).")

    try:
        results = evaluate_critic(sample, chunks_by_id, judge)
        near_miss_results = evaluate_critic(near_misses, chunks_by_id, judge)
    except DailyLimitReached as error:
        print(f"Stopped: {error}. Re-run tomorrow; finished calls are cached.")
        return

    metrics = compute_metrics([row["label"] for row in results], [row["prediction"] for row in results])
    metrics["fallbacks"] = sum(1 for row in results if row["fallback"])
    metrics["median_latency_seconds"] = sorted(row["latency_seconds"] or 0.0 for row in results)[len(results) // 2]
    metrics["per_source"] = {}
    for source in sorted({row["source"] for row in results}):
        of_source = [row for row in results if row["source"] == source]
        hits = sum(1 for row in of_source if row["label"] == row["prediction"])
        metrics["per_source"][source] = {"n": len(of_source), "accuracy": hits / len(of_source)}

    print(f"\nAccuracy {metrics['accuracy']:.3f} on {metrics['n']} questions (proxy labels)")
    for label in ("sufficient", "not_sufficient"):
        values = metrics[label]
        print(f"  {label:15s} precision {values['precision']:.3f} recall {values['recall']:.3f} f1 {values['f1']:.3f} (n={values['support']})")
    print(f"  confusion [true suff, true not] x [pred suff, pred not]: {metrics['confusion']}")
    print(f"  per source: {metrics['per_source']}; fallbacks {metrics['fallbacks']}; median latency {metrics['median_latency_seconds']:.2f}s")

    caught = sum(1 for row in near_miss_results if row["prediction"] == "not_sufficient")
    metrics["near_miss"] = {"n": len(near_miss_results), "caught": caught, "accuracy": caught / len(near_miss_results) if near_miss_results else 0.0}
    print(f"\nNear misses (source chunk removed, same-topic chunks kept): {caught}/{len(near_miss_results)} judged not_sufficient "
          f"({metrics['near_miss']['accuracy']:.3f})")

    errors = [row for row in results + near_miss_results if row["label"] != row["prediction"]]
    print(f"\nDisagreements ({len(errors)}), first 6:")
    for row in errors[:6]:
        print(f"  [{row['label']} -> {row['prediction']}] {row['query']}\n      missing: {row['missing']}")

    metrics["models"] = sorted({row["model"] for row in results + near_miss_results if row["model"]})
    output_path = PROJECT_ROOT / settings["eval_output"]
    if args.model:
        output_path = output_path.with_name(f"{output_path.stem}_{args.model}{output_path.suffix}")
    output_path.write_text(json.dumps({"metrics": metrics, "results": results, "near_miss_results": near_miss_results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
