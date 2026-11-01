"""Tests for the query reformulation agent (section 7). All LLM calls are mocked."""

import json
from types import SimpleNamespace

import pytest

from src.agents import reformulator
from src.agents.prompts import build_reformulate_prompt, build_retry_prompt
from src.agents.reformulator import (
    ReformulationReply,
    RetryReply,
    clean_queries,
    fallback_queries,
    reformulate,
)
from src.llm import DailyLimitReached, LLMError, LLMJSONError
from tests.test_llm import make_client


def fake_llm_result(parsed):
    """Shape of an LLMResult with only the fields the reformulator reads."""
    return SimpleNamespace(
        parsed=parsed, model_key="fake", latency_seconds=0.4, wait_seconds=0.1, total_tokens=50, cached=False
    )


def patch_call(monkeypatch, reply=None, error=None):
    """Replace call_llm; return the list that records each call's arguments."""
    calls = []

    def fake_call(prompt, system=None, role=None, response_model=None, json_mode=False):
        calls.append({"prompt": prompt, "system": system, "role": role, "response_model": response_model})
        if error is not None:
            raise error
        return fake_llm_result(reply)

    monkeypatch.setattr(reformulator, "call_llm", fake_call)
    return calls


# -- clean_queries / fallback -----------------------------------------------------


def test_clean_queries_strips_dedupes_and_caps():
    queries = ["  Who directed  Inception? ", "who directed inception?", "", "   ", "a", "b", "c"]
    assert clean_queries(queries, max_queries=3) == ["Who directed Inception?", "a", "b"]


def test_clean_queries_drops_repeats_of_previous_searches():
    cleaned = clean_queries(["old query", "New angle"], max_queries=3, previous_queries=["Old  Query"])
    assert cleaned == ["New angle"]


def test_clean_queries_keeps_repeats_if_nothing_new():
    assert clean_queries(["old query"], max_queries=3, previous_queries=["old query"]) == ["old query"]


def test_fallback_queries():
    assert fallback_queries("q", None) == ["q"]
    assert fallback_queries("q", "missing recall numbers") == ["q missing recall numbers"]


# -- modes ---------------------------------------------------------------------------


def test_rewrite_mode(monkeypatch):
    reply = ReformulationReply(mode="rewrite", queries=["effect of int8 quantization on vector search recall"], reason="vague")
    calls = patch_call(monkeypatch, reply)
    result = reformulate("is quantization bad for search", max_queries=3)

    assert result["mode"] == "rewrite"
    assert result["queries"] == ["effect of int8 quantization on vector search recall"]
    assert result["reason"] == "vague"
    assert result["model"] == "fake" and result["tokens"] == 50
    assert calls[0]["role"] == "reformulator"
    assert calls[0]["response_model"] is ReformulationReply
    assert "Question: is quantization bad for search" in calls[0]["prompt"]


def test_decompose_mode(monkeypatch):
    reply = ReformulationReply(
        mode="decompose",
        queries=["Who directed the film Inception?", "What is the nationality of the director of Inception?"],
        reason="bridge entity",
    )
    patch_call(monkeypatch, reply)
    result = reformulate("What nationality is the director of Inception?", max_queries=3)
    assert result["mode"] == "decompose"
    assert len(result["queries"]) == 2


def test_decompose_with_one_query_is_reported_as_rewrite(monkeypatch):
    patch_call(monkeypatch, ReformulationReply(mode="decompose", queries=["only one"], reason=""))
    assert reformulate("q", max_queries=3)["mode"] == "rewrite"


def test_decompose_is_capped_at_max_queries(monkeypatch):
    patch_call(monkeypatch, ReformulationReply(mode="decompose", queries=["a", "b", "c", "d"], reason=""))
    assert reformulate("q", max_queries=2)["queries"] == ["a", "b"]


def test_retry_mode_uses_feedback_and_previous_queries(monkeypatch):
    reply = RetryReply(queries=["recall@5 gain of MatRAG on HotpotQA", "best retrieval method recall HotpotQA"], reason="")
    calls = patch_call(monkeypatch, reply)
    result = reformulate(
        "Which method has the best recall?",
        feedback="no recall numbers",
        previous_queries=["best retrieval method recall HotpotQA"],
        max_queries=3,
    )
    assert result["mode"] == "retry"
    assert result["queries"] == ["recall@5 gain of MatRAG on HotpotQA"]  # the repeat was dropped
    assert calls[0]["response_model"] is RetryReply
    assert "no recall numbers" in calls[0]["prompt"]
    assert "- best retrieval method recall HotpotQA" in calls[0]["prompt"]


def test_max_queries_defaults_to_config(monkeypatch):
    calls = patch_call(monkeypatch, ReformulationReply(queries=["x"]))
    reformulate("q")
    assert "2 to 3 sub-questions" in calls[0]["prompt"]


# -- failures ------------------------------------------------------------------------


def test_json_error_falls_back_to_original_query(monkeypatch):
    patch_call(monkeypatch, error=LLMJSONError("bad twice"))
    result = reformulate("original question", max_queries=3)
    assert result["mode"] == "fallback"
    assert result["queries"] == ["original question"]
    assert "bad twice" in result["reason"]


def test_provider_error_in_retry_mode_falls_back_with_feedback(monkeypatch):
    patch_call(monkeypatch, error=LLMError("HTTP 500 after 3 retries"))
    result = reformulate("q", feedback="missing dates", max_queries=3)
    assert result["mode"] == "fallback"
    assert result["queries"] == ["q missing dates"]


def test_daily_limit_is_not_hidden(monkeypatch):
    patch_call(monkeypatch, error=DailyLimitReached("limit"))
    with pytest.raises(DailyLimitReached):
        reformulate("q", max_queries=3)


def test_empty_queries_fall_back(monkeypatch):
    patch_call(monkeypatch, ReformulationReply(queries=["   "], reason=""))
    result = reformulate("q", max_queries=3)
    assert result["mode"] == "fallback" and result["queries"] == ["q"]


# -- through the real wrapper (fake provider): JSON parsing and the one retry ---------


def use_wrapper_with_script(monkeypatch, tmp_path, script):
    """Route reformulate() through a real LLMClient whose provider returns scripted replies."""
    client, provider, _ = make_client(tmp_path, script, roles={"reformulator": "m1"})
    monkeypatch.setattr(reformulator, "call_llm", client.call)
    return provider


def test_valid_json_is_parsed_through_wrapper(monkeypatch, tmp_path):
    reply = json.dumps({"mode": "decompose", "queries": ["sub 1", "sub 2"], "reason": "two hops"})
    provider = use_wrapper_with_script(monkeypatch, tmp_path, ["```json\n" + reply + "\n```"])
    result = reformulate("q", max_queries=3)
    assert result["mode"] == "decompose" and result["queries"] == ["sub 1", "sub 2"]
    assert len(provider.calls) == 1


def test_malformed_json_is_retried_once_through_wrapper(monkeypatch, tmp_path):
    good = json.dumps({"mode": "rewrite", "queries": ["clear query"], "reason": "r"})
    provider = use_wrapper_with_script(monkeypatch, tmp_path, ["Sure! Here are queries: sub 1, sub 2", good])
    result = reformulate("q", max_queries=3)
    assert result["queries"] == ["clear query"]
    assert len(provider.calls) == 2
    assert "not valid JSON" in provider.calls[1]["prompt"]


def test_schema_violation_twice_falls_back_through_wrapper(monkeypatch, tmp_path):
    provider = use_wrapper_with_script(monkeypatch, tmp_path, ['{"queries": []}', '{"mode": "other", "queries": ["x"]}'])
    result = reformulate("original", max_queries=3)
    assert result["mode"] == "fallback" and result["queries"] == ["original"]
    assert len(provider.calls) == 2  # exactly one retry, then give up


# -- prompts -------------------------------------------------------------------------


def test_prompts_contain_question_and_json_format():
    first = build_reformulate_prompt("Who is older, A or B?", 3)
    assert "Who is older, A or B?" in first and '"mode"' in first and "2 to 3" in first
    retry = build_retry_prompt("q", "missing birth dates", [], 3)
    assert "missing birth dates" in retry and "(none recorded)" in retry and '"queries"' in retry
