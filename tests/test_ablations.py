"""Tests for the section 11 ablations: the knob stand-ins, the results folder and the table."""

import json

import pytest

from src.config import load_config
from src.eval.ablations import (
    ABLATIONS, BASELINE, ablation_config, accept_all_judge, format_table, identity_reformulate,
    no_decompose_reformulate, run_ablation, save_ablation_summary, summarize_ablations,
)


def test_every_ablation_changes_exactly_one_thing():
    assert ABLATIONS[BASELINE]["overrides"] == {}
    for name, entry in ABLATIONS.items():
        assert entry["description"], f"{name} has no description"
        if name == BASELINE:
            continue
        # "no_critic" is the one pair: dropping the critic also makes retries meaningless.
        expected = 2 if name == "no_critic" else 1
        assert len(entry["overrides"]) == expected, f"{name} changes {len(entry['overrides'])} knobs"


def test_identity_reformulate_makes_no_call_and_keeps_the_query():
    result = identity_reformulate("What is RAG?")
    assert result["queries"] == ["What is RAG?"]
    assert result["mode"] == "identity"
    assert result["model"] is None
    assert result["tokens"] == 0


def test_identity_reformulate_ignores_feedback():
    result = identity_reformulate("q", feedback="missing the year", previous_queries=["q"])
    assert result["queries"] == ["q"]


def test_no_decompose_collapses_a_decomposition():
    def fake_reformulate(query, feedback=None, previous_queries=None):
        return {"queries": ["part one", "part two"], "mode": "decompose", "reason": "two hops",
                "model": "flash_lite_31", "latency_seconds": 0.5, "wait_seconds": 0.0, "tokens": 10, "cached": False}

    wrapped = no_decompose_reformulate(fake_reformulate)
    result = wrapped("q")
    assert result["queries"] == ["part one"]
    assert result["mode"] == "rewrite"
    assert "decomposition off" in result["reason"]
    # The call still happened, so its cost is still reported.
    assert result["model"] == "flash_lite_31"


def test_no_decompose_leaves_a_rewrite_alone():
    def fake_reformulate(query, feedback=None, previous_queries=None):
        return {"queries": ["clear query"], "mode": "rewrite", "reason": "", "model": "flash_lite_31",
                "latency_seconds": 0.1, "wait_seconds": 0.0, "tokens": 5, "cached": True}

    result = no_decompose_reformulate(fake_reformulate)("q")
    assert result["queries"] == ["clear query"]
    assert result["mode"] == "rewrite"


def test_accept_all_judge_is_free_and_always_sufficient():
    result = accept_all_judge("q", [{"chunk_id": "c1", "title": "t", "text": "x"}])
    assert result["verdict"] == "sufficient"
    assert result["model"] is None
    assert result["tokens"] == 0
    assert result["missing"] == ""


def test_ablation_config_redirects_the_results_dir():
    config = load_config()
    redirected = ablation_config(config)
    assert redirected["eval"]["results_dir"] == config["ablations"]["results_dir"]
    # The original config must not be modified.
    assert config["eval"]["results_dir"] != redirected["eval"]["results_dir"]


def make_config(tmp_path):
    """A config whose ablation results land in a temporary folder."""
    config = load_config()
    config = dict(config)
    config["ablations"] = dict(config["ablations"], results_dir=str(tmp_path / "ablations"))
    return config


def make_questions(count):
    return [
        {"query_id": f"q{i}", "query": f"question {i}", "references": [f"answer {i}"],
         "question_type": "arxiv_chunk", "gold_titles": [], "dataset": "arxiv"}
        for i in range(count)
    ]


def fake_runner(question, correct, **_):
    """Stand-in system runner: answers with the reference answer when ``correct``."""
    return {
        "answer": question["references"][0] if correct else "something else",
        "decision": "retrieve", "contexts": [], "verification": None,
        "model_seconds": 1.0, "wait_seconds": 0.0, "other_seconds": 0.1,
        "tokens_local": 100, "tokens_gemini": 10, "lite_calls": 2,
    }


def test_run_ablation_writes_records_and_resumes(tmp_path):
    config = make_config(tmp_path)
    questions = make_questions(5)
    new = run_ablation(config, "arxiv", BASELINE, questions, fake_runner, {"correct": True})
    assert new == 5
    # A second run finds every question finished.
    assert run_ablation(config, "arxiv", BASELINE, questions, fake_runner, {"correct": True}) == 0

    path = tmp_path / "ablations" / f"arxiv_{BASELINE}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 5
    assert all(record["system"] == BASELINE for record in records)


def test_run_ablation_respects_the_limit(tmp_path):
    config = make_config(tmp_path)
    assert run_ablation(config, "arxiv", BASELINE, make_questions(10), fake_runner, {"correct": True}, limit=3) == 3


def test_summarize_ablations_pairs_against_full(tmp_path):
    config = make_config(tmp_path)
    questions = make_questions(10)
    run_ablation(config, "arxiv", BASELINE, questions, fake_runner, {"correct": True})
    run_ablation(config, "arxiv", "no_critic", questions, fake_runner, {"correct": False})

    summary = summarize_ablations(config, "arxiv", list(ABLATIONS))
    assert set(summary) == {BASELINE, "no_critic"}
    assert summary[BASELINE]["relaxed_accuracy"] == 1.0
    assert summary["no_critic"]["relaxed_accuracy"] == 0.0
    paired = summary["no_critic"]["paired_vs_full"]
    assert paired["n"] == 10
    assert paired["difference"] == pytest.approx(-1.0)
    assert paired["losses"] == 10
    assert "paired_vs_full" not in summary[BASELINE]
    assert summary["no_critic"]["description"] == ABLATIONS["no_critic"]["description"]


def test_summarize_ablations_without_results_is_empty(tmp_path):
    assert summarize_ablations(make_config(tmp_path), "arxiv", list(ABLATIONS)) == {}


def test_save_and_format(tmp_path):
    config = make_config(tmp_path)
    questions = make_questions(4)
    run_ablation(config, "arxiv", BASELINE, questions, fake_runner, {"correct": True})
    run_ablation(config, "arxiv", "int8", questions, fake_runner, {"correct": False})
    summary = summarize_ablations(config, "arxiv", list(ABLATIONS))

    saved = save_ablation_summary(config, summary)
    assert json.loads(saved.read_text(encoding="utf-8"))["full"]["n"] == 4

    table = format_table(summary, list(ABLATIONS))
    assert "(baseline)" in table
    assert "| full |" in table and "| int8 |" in table
    # Only configurations with results get a row (plus two header lines).
    assert len(table.splitlines()) == 4
