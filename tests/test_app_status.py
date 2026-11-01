"""Tests for the demo app's service checks and trace helpers (no network, no Streamlit)."""

import pytest

from app import status as status_module
from app.status import (
    ServiceStatus,
    blocking,
    check_all,
    check_gemini_key,
    check_ollama,
    check_qdrant,
    claim_icon,
    count_lite_calls,
    retrieval_rounds,
    step_of,
    summarise,
    verdict_sequence,
)


@pytest.fixture
def config():
    """Only the parts of the config the checks read."""
    return {
        "vector_store": {"url": "http://127.0.0.1:6333"},
        "graph": {"collection": "papers_fp32"},
        "llm": {
            "models": {
                "ollama_qwen": {
                    "base_url": "http://127.0.0.1:11434",
                    "model_id": "qwen2.5:3b-instruct",
                }
            }
        },
    }


class FakeResponse:
    """A requests-like response that returns a fixed payload."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        """Successful responses never raise."""

    def json(self):
        """Return the fixed payload."""
        return self._payload


def fake_get(payload):
    """Build a `requests.get` stand-in that always returns `payload`."""

    def _get(url, timeout=None):
        return FakeResponse(payload)

    return _get


def failing_get(error):
    """Build a `requests.get` stand-in that always raises `error`."""

    def _get(url, timeout=None):
        raise error

    return _get


@pytest.fixture
def trace():
    """A trace with one retry round: two critiques, two retrievals."""
    return [
        {"step": "classify", "decision": "retrieve", "confidence": 0.91, "seconds": 0.08},
        {"step": "reformulate", "mode": "decompose", "retry": 0, "queries": ["a", "b"], "model": "flash_lite_31"},
        {"step": "retrieve", "hits": 10, "new_chunks": 10, "total_chunks": 10, "seconds": 0.01},
        {"step": "critique", "verdict": "not_sufficient", "confidence": 0.7, "missing": "no numbers", "model": "flash_lite_31"},
        {"step": "reformulate", "mode": "retry", "retry": 1, "queries": ["c"], "model": "flash_lite_31"},
        {"step": "retrieve", "hits": 5, "new_chunks": 2, "total_chunks": 12, "seconds": 0.01},
        {"step": "critique", "verdict": "sufficient", "confidence": 0.9, "missing": "", "model": "flash_lite_31"},
        {"step": "generate", "answer": "An answer.", "chunks_used": ["x:abstract:0"], "model": "ollama_qwen"},
        {"step": "verify", "status": "verified", "n_claims": 2, "n_supported": 2, "claims": [], "seconds": 1.2},
    ]


def test_qdrant_check_passes_when_the_collection_is_there(config, monkeypatch):
    """A running server holding the pipeline's collection is usable."""
    payload = {"result": {"collections": [{"name": "papers_fp32"}, {"name": "papers_int8"}]}}
    monkeypatch.setattr(status_module.requests, "get", fake_get(payload))
    result = check_qdrant(config)
    assert result.ok is True
    assert "papers_fp32" in result.detail


def test_qdrant_check_explains_a_missing_collection(config, monkeypatch):
    """A running server without the collection needs the index built, not restarting."""
    monkeypatch.setattr(status_module.requests, "get", fake_get({"result": {"collections": []}}))
    result = check_qdrant(config)
    assert result.ok is False
    assert "missing" in result.detail
    assert "build_index" in result.remedy


def test_qdrant_check_reports_a_dead_server_with_a_remedy(config, monkeypatch):
    """A connection failure is reported as a status, never raised."""
    monkeypatch.setattr(status_module.requests, "get", failing_get(ConnectionError("refused")))
    result = check_qdrant(config)
    assert result.ok is False
    assert "start_qdrant" in result.remedy


def test_ollama_check_accepts_a_tagged_model(config, monkeypatch):
    """The pulled model may carry a tag suffix; matching is on the model name."""
    payload = {"models": [{"name": "qwen2.5:3b-instruct"}]}
    monkeypatch.setattr(status_module.requests, "get", fake_get(payload))
    assert check_ollama(config).ok is True


def test_ollama_check_fails_when_the_model_is_not_pulled(config, monkeypatch):
    """A running Ollama without the generator cannot answer anything."""
    monkeypatch.setattr(status_module.requests, "get", fake_get({"models": [{"name": "llama3:8b"}]}))
    result = check_ollama(config)
    assert result.ok is False
    assert "ollama pull" in result.remedy


def test_gemini_key_check_reads_the_environment():
    """A present key passes without being printed in full."""
    present = check_gemini_key({"GOOGLE_API_KEY": "abcd1234secret"})
    assert present.ok is True
    assert "secret" not in present.detail
    assert present.detail.endswith("cret)")

    missing = check_gemini_key({})
    assert missing.ok is False
    assert ".env" in missing.remedy


def test_blocking_lists_only_the_failures():
    """The page blocks on failures and ignores the checks that passed."""
    statuses = [
        ServiceStatus("A", True, "fine"),
        ServiceStatus("B", False, "broken"),
        ServiceStatus("C", False, "also broken"),
    ]
    assert [status.name for status in blocking(statuses)] == ["B", "C"]


def test_check_all_runs_every_check(config, monkeypatch):
    """All three dependencies are reported, in pipeline order."""
    monkeypatch.setattr(status_module.requests, "get", failing_get(ConnectionError("refused")))
    names = [status.name for status in check_all(config)]
    assert names == ["Qdrant", "Ollama", "Gemini key"]


def test_lite_calls_count_only_remote_steps(trace):
    """Retrieval and generation are local, so they never count against the Gemini quota."""
    assert count_lite_calls(trace) == 4  # two reformulations, two critiques


def test_lite_calls_ignore_steps_without_a_model():
    """A step that was served without an LLM call is not counted."""
    assert count_lite_calls([{"step": "reformulate", "queries": []}]) == 0


def test_retrieval_rounds_and_verdicts(trace):
    """The retry loop is summarised by its searches and its verdicts in order."""
    assert retrieval_rounds(trace) == 2
    assert verdict_sequence(trace) == ["not_sufficient", "sufficient"]


def test_step_of_returns_none_for_a_step_that_never_ran(trace):
    """On the skip path there is no retrieval step at all."""
    assert step_of(trace, "classify")["decision"] == "retrieve"
    assert step_of(trace, "nonexistent") is None


def test_summarise_reduces_a_state_to_the_shown_numbers(trace):
    """The metric row is built from the trace, not from a separate bookkeeping path."""
    state = {
        "trace": trace,
        "retries": 1,
        "total_seconds": 12.5,
        "contexts": [{"chunk_id": "a"}, {"chunk_id": "b"}],
        "stop_reason": "sufficient",
    }
    summary = summarise(state)
    assert summary["decision"] == "retrieve"
    assert summary["retrieval_rounds"] == 2
    assert summary["retries"] == 1
    assert summary["lite_calls"] == 4
    assert summary["chunks"] == 2
    assert summary["verdicts"] == ["not_sufficient", "sufficient"]


def test_summarise_handles_the_skip_path():
    """With the classifier skipping, there is no retrieval and no remote call."""
    state = {
        "trace": [{"step": "classify", "decision": "skip", "confidence": 0.1, "seconds": 0.08},
                  {"step": "generate", "answer": "Paris.", "chunks_used": [], "model": "ollama_qwen"}],
        "retries": 0,
        "total_seconds": 0.6,
        "contexts": [],
    }
    summary = summarise(state)
    assert summary["decision"] == "skip"
    assert summary["retrieval_rounds"] == 0
    assert summary["lite_calls"] == 0
    assert summary["chunks"] == 0


def test_claim_icon_is_distinct_per_label():
    """Each verification label gets its own icon, so colour is never the only channel."""
    icons = {claim_icon(label) for label in ("supported", "unsupported", "contradicted")}
    assert len(icons) == 3
    assert claim_icon("something else") == "•"
