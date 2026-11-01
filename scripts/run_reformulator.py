"""Run the query reformulator on sample questions and save the results for review.

Samples: real HotpotQA (distractor, validation) multi-hop questions, a few vague queries
and the retry cases from ``reformulator`` in configs/config.yaml. Every call goes through
the cached LLM wrapper, so re-running costs no quota.

Run from the project root:
    python scripts/run_reformulator.py
    python scripts/run_reformulator.py --model gemma_26b --hotpotqa-count 2   # test one model
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src import llm  # noqa: E402
from src.agents.reformulator import reformulate  # noqa: E402
from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.llm import DailyLimitReached, LLMClient  # noqa: E402


def load_hotpotqa_questions(count: int, seed: int) -> List[Dict[str, Any]]:
    """Stream ``count`` shuffled HotpotQA distractor validation questions (no full download).

    Half are "comparison" questions and half "bridge" questions, so both multi-hop types are seen.
    """
    from datasets import load_dataset

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=1000)
    wanted = {"comparison": count // 2, "bridge": count - count // 2}
    questions = []
    for row in dataset:
        if wanted.get(row["type"], 0) <= 0:
            continue
        wanted[row["type"]] -= 1
        questions.append({"question": row["question"], "answer": row["answer"], "type": row["type"]})
        if len(questions) >= count:
            break
    return questions


def use_model_for_reformulator(model_key: str) -> None:
    """Point the reformulator role at one model for this run only (config file unchanged)."""
    config = copy.deepcopy(load_config())
    config["llm"]["roles"]["reformulator"] = model_key
    llm._default_client = LLMClient(config)


def print_result(label: str, result: Dict[str, Any]) -> None:
    """Print one reformulation in a readable block."""
    print(f"\n[{label}] mode={result['mode']} model={result['model']} "
          f"latency={result['latency_seconds']:.1f}s wait={result['wait_seconds']:.1f}s cached={result['cached']}")
    for query in result["queries"]:
        print(f"   -> {query}")
    print(f"   reason: {result['reason']}")


def main() -> None:
    """Reformulate the sample questions, print them and save them to the demo output file."""
    parser = argparse.ArgumentParser(description="Run the query reformulator on sample questions.")
    parser.add_argument("--model", default=None, help="Model key to use instead of llm.roles.reformulator.")
    parser.add_argument("--hotpotqa-count", type=int, default=None, help="Override reformulator.demo_hotpotqa_count.")
    parser.add_argument("--no-extras", action="store_true", help="Skip the vague and retry examples.")
    args = parser.parse_args()

    # Windows consoles default to cp1252; HotpotQA has names like "Przemysław".
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(PROJECT_ROOT / ".env")
    settings = load_config()["reformulator"]
    if args.model:
        use_model_for_reformulator(args.model)

    count = args.hotpotqa_count if args.hotpotqa_count is not None else settings["demo_hotpotqa_count"]
    samples = [dict(item, kind="hotpotqa") for item in load_hotpotqa_questions(count, settings["demo_seed"])]
    if not args.no_extras:
        samples += [{"question": query, "kind": "vague"} for query in settings["demo_vague_queries"]]
        samples += [dict(case, question=case["query"], kind="retry") for case in settings["demo_retry_cases"]]

    records = []
    try:
        for sample in samples:
            print(f"\nQ ({sample['kind']}{', ' + sample['type'] if 'type' in sample else ''}): {sample['question']}")
            result = reformulate(
                sample["question"], feedback=sample.get("feedback"), previous_queries=sample.get("previous_queries")
            )
            print_result(sample["kind"], result)
            records.append({"input": sample, "output": result})
    except DailyLimitReached as error:
        print(f"\nStopped: {error}. Results so far are saved; re-run tomorrow (finished calls are cached).")

    output_path = PROJECT_ROOT / settings["demo_output"]
    if args.model:
        output_path = output_path.with_name(f"{output_path.stem}_{args.model}{output_path.suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    modes = [record["output"]["mode"] for record in records]
    print(f"\n{len(records)} samples, modes: " + ", ".join(f"{mode}={modes.count(mode)}" for mode in sorted(set(modes))))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
