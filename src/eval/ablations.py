"""Section 11 ablations: turn one part of the agentic pipeline off at a time and measure the cost.

Every configuration is the full ``agentic_v2`` system (classifier -> reformulate -> retrieve ->
critique loop -> short cited generation -> NLI verification) with exactly one knob changed, so a
difference in accuracy can be attributed to that knob. They all run on the same arXiv evaluation
questions and are compared against the ``full`` configuration with a paired bootstrap, because a
difference of a few points on 100 questions is usually noise.

Results go to their own folder (``ablations.results_dir``) in the same per-question JSONL format as
the section 10 harness, so a stopped run resumes and the LLM cache makes repeated calls free.

The knobs:

======================  ===========================================================
``full``                the current default; the row everything else is compared to
``no_classifier``       always retrieve (block 1 off)
``no_reformulation``    search the question as typed, no LLM rewrite (block 2 off)
``no_decompose``        rewrite only: a multi-hop question is never split
``no_critic``           evidence is accepted without judging it (blocks 3-4 off)
``retries_0``           judge once, never retry (block 4 off, block 3 still runs)
``retries_1``           at most one retry instead of two
``contexts_3``          show the critic and the generator 3 chunks instead of 6
``contexts_10``         show 10 chunks (more evidence, but the 3B generator has 4096 tokens)
``top_k_10``            retrieve 10 chunks per query instead of 5
``int8``                retrieve from the int8-quantized collection instead of fp32
======================  ===========================================================
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.config import PROJECT_ROOT
from src.eval.harness import evaluate, paired_difference, results_dir, summarize
from src.ingest.io_utils import read_jsonl

logger = logging.getLogger(__name__)

BASELINE = "full"

# name -> (description, build_graph keyword overrides). "collection" and "reformulate" are
# handled by the caller because they need the store and the reformulator function.
ABLATIONS: Dict[str, Dict[str, Any]] = {
    "full": {"description": "current default (agentic_v2)", "overrides": {}},
    "no_classifier": {"description": "block 1 off: always retrieve", "overrides": {"use_classifier": False}},
    "no_reformulation": {"description": "block 2 off: search the question as typed", "overrides": {"reformulate": "identity"}},
    "no_decompose": {"description": "block 2 partial: rewrite only, never split", "overrides": {"reformulate": "no_decompose"}},
    "no_critic": {"description": "blocks 3-4 off: evidence accepted unjudged", "overrides": {"judge": "accept_all", "max_retries": 0}},
    "retries_0": {"description": "block 4 off: judge once, no retry", "overrides": {"max_retries": 0}},
    "retries_1": {"description": "at most one retry", "overrides": {"max_retries": 1}},
    "contexts_3": {"description": "3 chunks shown instead of 6", "overrides": {"max_contexts": 3}},
    "contexts_10": {"description": "10 chunks shown instead of 6", "overrides": {"max_contexts": 10}},
    "top_k_10": {"description": "retrieve 10 chunks per query instead of 5", "overrides": {"top_k": 10}},
    "int8": {"description": "retrieve from the int8-quantized collection", "overrides": {"collection": "int8"}},
}


def identity_reformulate(
    query: str, feedback: Optional[str] = None, previous_queries: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Reformulation ablation: return the question unchanged, with no LLM call.

    Has the same signature and result shape as ``src.agents.reformulator.reformulate``.
    """
    return {
        "queries": [query],
        "reason": "ablation: reformulation off",
        "mode": "identity",
        "model": None,
        "latency_seconds": 0.0,
        "wait_seconds": 0.0,
        "tokens": 0,
        "cached": False,
    }


def no_decompose_reformulate(reformulate_fn: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Wrap a reformulator so a decomposition is collapsed to its first query.

    The LLM call still happens (and is cached), so this isolates "was splitting the question
    worth it?" from "was rewriting the question worth it?".
    """

    def wrapped(query: str, feedback: Optional[str] = None, previous_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        result = dict(reformulate_fn(query, feedback=feedback, previous_queries=previous_queries))
        if result["mode"] == "decompose":
            result["queries"] = result["queries"][:1]
            result["mode"] = "rewrite"
            result["reason"] = "ablation: decomposition off; " + result.get("reason", "")
        return result

    return wrapped


def accept_all_judge(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Critic ablation: always answer "sufficient" without an LLM call.

    Same result shape as ``src.agents.critic.judge``. Combined with ``max_retries: 0`` this
    removes blocks 3 and 4 completely: one retrieval, then generation.
    """
    return {
        "verdict": "sufficient",
        "missing": "",
        "confidence": None,
        "fallback": False,
        "model": None,
        "latency_seconds": 0.0,
        "wait_seconds": 0.0,
        "tokens": 0,
        "cached": False,
    }


def ablation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of the config whose ``eval.results_dir`` points at the ablation folder.

    The section 10 harness functions write to ``eval.results_dir``; redirecting it keeps the
    ablation records out of the main evaluation results.
    """
    copied = dict(config)
    copied["eval"] = dict(config["eval"])
    copied["eval"]["results_dir"] = config["ablations"]["results_dir"]
    return copied


def run_ablation(
    config: Dict[str, Any],
    dataset: str,
    name: str,
    questions: List[Dict[str, Any]],
    runner: Callable[..., Dict[str, Any]],
    runner_args: Dict[str, Any],
    limit: Optional[int] = None,
) -> int:
    """Run one ablation on ``questions`` and append the new records; returns how many were new."""
    return evaluate(ablation_config(config), dataset, name, questions, runner, runner_args, limit)


def summarize_ablations(config: Dict[str, Any], dataset: str, names: List[str]) -> Dict[str, Any]:
    """Summarise every ablation that has results, with a paired difference against ``full``.

    Returns ``{"<name>": {metrics..., "paired_vs_full": {...}, "description": str}}``.
    """
    folder = results_dir(ablation_config(config))
    records_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for name in names:
        path = folder / f"{dataset}_{name}.jsonl"
        if path.is_file():
            records = list(read_jsonl(path))
            if records:
                records_by_name[name] = records

    baseline_records = records_by_name.get(BASELINE, [])
    summary: Dict[str, Any] = {}
    for name, records in records_by_name.items():
        values = summarize(records)
        values["description"] = ABLATIONS[name]["description"]
        if baseline_records and name != BASELINE:
            values["paired_vs_full"] = paired_difference(records, baseline_records)
        summary[name] = values
    return summary


def save_ablation_summary(config: Dict[str, Any], summary: Dict[str, Any]) -> Path:
    """Write the ablation summary JSON into the ablation results folder."""
    path = results_dir(ablation_config(config)) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def format_table(summary: Dict[str, Any], order: List[str]) -> str:
    """Markdown table of the ablation summary, one row per configuration."""
    lines = [
        "| configuration | n | relaxed acc | vs full [95% CI] | skip | false abst | model s | lite calls | what changed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in order:
        values = summary.get(name)
        if not values:
            continue
        paired = values.get("paired_vs_full")
        if paired and paired.get("n"):
            difference = f"{paired['difference']:+.3f} [{paired['ci95'][0]:+.3f}, {paired['ci95'][1]:+.3f}]"
        else:
            difference = "-" if name != BASELINE else "(baseline)"
        skip = values["skip_rate"]
        false_abstention = values["false_abstention_rate"]
        lines.append(
            f"| {name} | {values['n']} | {values['relaxed_accuracy']:.3f} | {difference} | "
            f"{'-' if skip is None else f'{skip:.2f}'} | "
            f"{'-' if false_abstention is None else f'{false_abstention:.2f}'} | "
            f"{values['mean_model_seconds']:.1f} | {values['mean_lite_calls']:.2f} | {values['description']} |"
        )
    return "\n".join(lines)


def save_table(config: Dict[str, Any], table: str) -> Path:
    """Write the markdown table next to the ablation results."""
    path = PROJECT_ROOT / config["ablations"]["table_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table + "\n", encoding="utf-8")
    return path
