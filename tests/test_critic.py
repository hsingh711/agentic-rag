"""Tests for the sufficiency critic (section 8a). All LLM calls are mocked."""

import json
from types import SimpleNamespace

import pytest

from src.agents import critic
from src.agents.critic import CriticReply, judge
from src.agents.prompts import build_critic_prompt
from src.llm import DailyLimitReached, LLMError, LLMJSONError
from tests.test_llm import make_client

CHUNKS = [
    {"chunk_id": "2610.01427:abstract:0", "title": "SHAMS", "text": "SHAMS contains 1,300 utterances."},
    {"chunk_id": "2610.00554:abstract:0", "title": "Brain", "text": "We use Proper Orthogonal Decomposition."},
]


def patch_call(monkeypatch, reply=None, error=None):
    """Replace call_llm; return the list that records each call."""
    calls = []

    def fake_call(prompt, system=None, role=None, response_model=None, json_mode=False):
        calls.append({"prompt": prompt, "role": role, "response_model": response_model})
        if error is not None:
            raise error
        return SimpleNamespace(parsed=reply, model_key="fake", latency_seconds=0.3, wait_seconds=0.0, total_tokens=90, cached=False)

    monkeypatch.setattr(critic, "call_llm", fake_call)
    return calls


def test_sufficient_verdict(monkeypatch):
    calls = patch_call(monkeypatch, CriticReply(verdict="sufficient", missing="ignored", confidence=0.9))
    result = judge("How many utterances are in SHAMS?", CHUNKS)
    assert result["verdict"] == "sufficient"
    assert result["missing"] == ""  # missing text is only kept for not_sufficient
    assert result["confidence"] == 0.9 and not result["fallback"]
    assert calls[0]["role"] == "critic" and calls[0]["response_model"] is CriticReply


def test_not_sufficient_keeps_missing(monkeypatch):
    patch_call(monkeypatch, CriticReply(verdict="not_sufficient", missing="the recall@5 of MatRAG", confidence=0.8))
    result = judge("q", CHUNKS)
    assert result["verdict"] == "not_sufficient" and result["missing"] == "the recall@5 of MatRAG"


def test_not_sufficient_without_missing_gets_default(monkeypatch):
    patch_call(monkeypatch, CriticReply(verdict="not_sufficient", missing="  ", confidence=0.8))
    assert judge("which model?", CHUNKS)["missing"] == "more specific evidence for: which model?"


def test_no_chunks_is_not_sufficient_without_a_call(monkeypatch):
    calls = patch_call(monkeypatch, None)
    result = judge("q", [])
    assert result["verdict"] == "not_sufficient" and calls == []


def test_llm_failure_falls_back_to_sufficient(monkeypatch):
    patch_call(monkeypatch, error=LLMJSONError("bad"))
    result = judge("q", CHUNKS)
    assert result["verdict"] == "sufficient" and result["fallback"] and result["confidence"] == 0.0

    patch_call(monkeypatch, error=LLMError("HTTP 503"))
    assert judge("q", CHUNKS)["fallback"]


def test_daily_limit_is_not_hidden(monkeypatch):
    patch_call(monkeypatch, error=DailyLimitReached("limit"))
    with pytest.raises(DailyLimitReached):
        judge("q", CHUNKS)


def test_prompt_lists_chunk_ids_and_format():
    prompt = build_critic_prompt("How many utterances?", CHUNKS)
    assert "[2610.01427:abstract:0] SHAMS" in prompt
    assert '"verdict"' in prompt and "How many utterances?" in prompt


# -- through the real wrapper (fake provider) --------------------------------------


def use_wrapper(monkeypatch, tmp_path, script):
    client, provider, _ = make_client(tmp_path, script, roles={"critic": "m1"})
    monkeypatch.setattr(critic, "call_llm", client.call)
    return provider


def test_valid_json_through_wrapper(monkeypatch, tmp_path):
    reply = json.dumps({"verdict": "not_sufficient", "missing": "dates", "confidence": 0.7})
    provider = use_wrapper(monkeypatch, tmp_path, [reply])
    assert judge("q", CHUNKS)["missing"] == "dates"
    assert len(provider.calls) == 1


def test_malformed_json_retried_once_then_fallback(monkeypatch, tmp_path):
    bad_verdict = json.dumps({"verdict": "maybe", "confidence": 0.5})
    out_of_range = json.dumps({"verdict": "sufficient", "confidence": 7})
    provider = use_wrapper(monkeypatch, tmp_path, [bad_verdict, out_of_range])
    result = judge("q", CHUNKS)
    assert result["fallback"] and len(provider.calls) == 2


def test_malformed_then_valid_through_wrapper(monkeypatch, tmp_path):
    good = json.dumps({"verdict": "sufficient", "missing": "", "confidence": 0.95})
    provider = use_wrapper(monkeypatch, tmp_path, ["The passages look fine to me.", good])
    result = judge("q", CHUNKS)
    assert result["verdict"] == "sufficient" and not result["fallback"]
    assert len(provider.calls) == 2


# -- critic evaluation against section 5 proxy labels ------------------------------------

from src.eval.critic_eval import build_proxy_rows, compute_metrics, evaluate_critic, proxy_label, sample_balanced  # noqa: E402


def details(retrieved, score, evidence, source="arxiv_chunk", query_id="q"):
    return {
        "query_id": query_id, "query": "question", "source": source, "rag_score": score,
        "source_chunk_retrieved": retrieved, "evidence_found": evidence, "retrieved_chunk_ids": ["c1", "c2"],
    }


def test_proxy_label_rule():
    assert proxy_label(details(True, 1, True)) == "sufficient"
    assert proxy_label(details(False, 0, False)) == "not_sufficient"
    assert proxy_label(details(True, 0, True)) is None      # evidence there but answer wrong
    assert proxy_label(details(False, 1, True)) is None     # answer found in another chunk
    assert proxy_label(details(True, 1, False)) is None


def test_build_proxy_rows_filters_sources_and_mixed():
    records = [details(True, 1, True, query_id="a"), details(False, 0, False, "triviaqa", "b"), details(True, 0, True, query_id="c")]
    rows = build_proxy_rows(records, ["arxiv_chunk"])
    assert [row["query_id"] for row in rows] == ["a"]


def test_sample_balanced_is_seeded_and_balanced():
    rows = [{"label": "sufficient", "id": i} for i in range(10)] + [{"label": "not_sufficient", "id": i} for i in range(3)]
    first = sample_balanced(rows, per_class=4, seed=1)
    assert sample_balanced(rows, per_class=4, seed=1) == first
    assert sum(row["label"] == "sufficient" for row in first) == 4
    assert sum(row["label"] == "not_sufficient" for row in first) == 3


def test_compute_metrics():
    labels = ["sufficient", "sufficient", "not_sufficient", "not_sufficient"]
    predictions = ["sufficient", "not_sufficient", "not_sufficient", "not_sufficient"]
    metrics = compute_metrics(labels, predictions)
    assert metrics["accuracy"] == 0.75
    assert metrics["not_sufficient"]["recall"] == 1.0 and metrics["not_sufficient"]["precision"] == pytest.approx(2 / 3)
    assert metrics["sufficient"]["recall"] == 0.5
    assert metrics["confusion"] == [[1, 1], [0, 2]]


def test_evaluate_critic_passes_the_original_chunks():
    seen = []

    def fake_judge(query, chunks):
        seen.append([chunk["chunk_id"] for chunk in chunks])
        return {"verdict": "sufficient"}

    rows = [{"query_id": "a", "query": "q", "source": "arxiv_chunk", "label": "sufficient", "chunk_ids": ["c1", "missing", "c2"]}]
    results = evaluate_critic(rows, {"c1": {"chunk_id": "c1"}, "c2": {"chunk_id": "c2"}}, fake_judge)
    assert seen == [["c1", "c2"]] and results[0]["prediction"] == "sufficient"


def test_build_near_miss_rows_removes_source_and_checks_answer():
    from src.eval.critic_eval import build_near_miss_rows

    rows = [
        {"query_id": "a", "query": "q", "source": "arxiv_chunk", "label": "sufficient", "chunk_ids": ["src_a", "x", "y"]},
        {"query_id": "b", "query": "q", "source": "arxiv_chunk", "label": "sufficient", "chunk_ids": ["src_b", "z"]},
        {"query_id": "c", "query": "q", "source": "arxiv_chunk", "label": "not_sufficient", "chunk_ids": ["x"]},
    ]
    chunks = {
        "x": {"text": "unrelated"}, "y": {"text": "also unrelated"},
        "z": {"text": "this one mentions 1,300 utterances"},
    }
    near = build_near_miss_rows(rows, {"a": "src_a", "b": "src_b"}, {"a": ["POD"], "b": ["1,300"]}, chunks)
    assert [(row["query_id"], row["label"], row["chunk_ids"]) for row in near] == [("a", "not_sufficient", ["x", "y"])]
