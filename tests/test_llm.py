"""Tests for the LLM wrapper. No network, no real models, no real waiting."""

import copy
import csv
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from src import llm
from src.config import load_config
from src.llm import (
    ConfigError,
    DailyCounter,
    DailyLimitReached,
    LLMClient,
    LLMError,
    LLMJSONError,
    OllamaProvider,
    GeminiProvider,
    ProviderReply,
    RateLimiter,
    RetryableError,
    parse_json_text,
)


class FakeClock:
    """A clock whose ``sleep`` just moves time forward."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class ScriptedProvider:
    """Returns (or raises) the next scripted item on each call and records the calls."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate(self, model_config, prompt, system, json_mode, temperature, timeout):
        self.calls.append({"prompt": prompt, "system": system, "json_mode": json_mode})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return ProviderReply(text=item, input_tokens=10, output_tokens=5)


class Answer(BaseModel):
    verdict: str
    confidence: float


def gemini_model(rpm=15, tpm=250000, rpd=500):
    return {
        "provider": "fake",
        "model_id": "fake-model",
        "rpm": rpm,
        "tpm": tpm,
        "rpd": rpd,
        "supports_system_instruction": True,
        "supports_json_mode": True,
        "notional_usd_per_1m_input": 1.0,
        "notional_usd_per_1m_output": 2.0,
    }


def make_config(tmp_path, models=None, roles=None):
    config = copy.deepcopy(load_config())
    llm_config = config["llm"]
    llm_config["cache_dir"] = str(tmp_path / "cache")
    llm_config["usage_log_path"] = str(tmp_path / "usage.csv")
    llm_config["daily_counter_path"] = str(tmp_path / "counter.json")
    llm_config["models"] = models or {"m1": gemini_model()}
    llm_config["roles"] = roles or {"generator": "m1"}
    return config


def make_client(tmp_path, script, models=None, roles=None, today="2026-01-01"):
    clock = FakeClock()
    provider = ScriptedProvider(script)
    # tmp_path is absolute, so PROJECT_ROOT / absolute path resolves to tmp_path.
    client = LLMClient(
        make_config(tmp_path, models, roles),
        providers={"fake": provider},
        clock=clock.time,
        sleep=clock.sleep,
        today_fn=lambda: today,
    )
    return client, provider, clock


# -- cache --------------------------------------------------------------------


def test_identical_call_is_served_from_cache(tmp_path):
    client, provider, _ = make_client(tmp_path, ["hello"])
    first = client.call("question")
    second = client.call("question")
    assert first.text == second.text == "hello"
    assert not first.cached and second.cached
    assert len(provider.calls) == 1
    assert second.input_tokens == 10  # tokens of the original call are kept for reports


def test_different_prompt_or_system_is_not_cached(tmp_path):
    client, provider, _ = make_client(tmp_path, ["a", "b", "c"])
    client.call("q1")
    client.call("q2")
    client.call("q1", system="be brief")
    assert len(provider.calls) == 3


# -- rate limiter ---------------------------------------------------------------


def test_request_limit_is_80_percent_and_waits_for_window():
    clock = FakeClock()
    limiter = RateLimiter(rpm=5, tpm=250000, fraction=0.8, clock=clock.time, sleep=clock.sleep)
    for _ in range(4):  # 80% of 5 = 4 requests per minute
        _, waited = limiter.acquire(10)
        assert waited == 0
    _, waited = limiter.acquire(10)
    assert waited == pytest.approx(60.05, abs=0.1)


def test_token_limit_is_80_percent():
    clock = FakeClock()
    limiter = RateLimiter(rpm=30, tpm=16000, fraction=0.8, clock=clock.time, sleep=clock.sleep)  # 12,800 tokens
    limiter.acquire(10000)
    _, waited = limiter.acquire(5000)  # 15,000 > 12,800 -> must wait for the first call to expire
    assert waited > 55


def test_actual_tokens_replace_estimate():
    clock = FakeClock()
    limiter = RateLimiter(rpm=30, tpm=16000, fraction=0.8, clock=clock.time, sleep=clock.sleep)
    entry, _ = limiter.acquire(10000)
    RateLimiter.record_actual_tokens(entry, 100)
    _, waited = limiter.acquire(10000)
    assert waited == 0


def test_oversized_single_call_is_admitted_when_window_is_empty():
    clock = FakeClock()
    limiter = RateLimiter(rpm=30, tpm=1000, fraction=0.8, clock=clock.time, sleep=clock.sleep)
    _, waited = limiter.acquire(5000)
    assert waited == 0


# -- daily limit ----------------------------------------------------------------


def test_daily_counter_persists_and_resets(tmp_path):
    path = tmp_path / "c.json"
    day = {"value": "2026-01-01"}
    counter = DailyCounter(path, "America/Los_Angeles", today_fn=lambda: day["value"])
    counter.increment("m1")
    counter.increment("m1")
    assert DailyCounter(path, "America/Los_Angeles", today_fn=lambda: day["value"]).count("m1") == 2
    day["value"] = "2026-01-02"
    assert counter.count("m1") == 0


def test_stops_at_90_percent_of_daily_limit(tmp_path):
    models = {"m1": gemini_model(rpd=10, rpm=1000)}  # 90% of 10 = 9 requests
    client, provider, _ = make_client(tmp_path, [f"r{i}" for i in range(20)], models=models)
    for number in range(9):
        client.call(f"question {number}")
    with pytest.raises(DailyLimitReached):
        client.call("question 9")
    assert len(provider.calls) == 9


def test_daily_counter_survives_new_client(tmp_path):
    models = {"m1": gemini_model(rpd=10, rpm=1000)}
    first, _, _ = make_client(tmp_path, ["a"] * 20, models=models)
    for number in range(9):
        first.call(f"q{number}")
    second, _, _ = make_client(tmp_path, ["a"] * 20, models=models)
    with pytest.raises(DailyLimitReached):
        second.call("another new question")


def test_role_list_uses_model_with_most_quota_left(tmp_path):
    models = {"a": gemini_model(rpd=100), "b": gemini_model(rpd=100)}
    models["b"]["model_id"] = "fake-b"
    client, _, _ = make_client(tmp_path, ["x"] * 10, models=models, roles={"generator": ["a", "b"]})
    client.settings["model_selection"] = "most_quota_left"
    client.counter.increment("a")
    client.counter.increment("a")
    assert client.choose_model("generator") == "b"
    client.counter.increment("b")
    client.counter.increment("b")
    client.counter.increment("b")
    assert client.choose_model("generator") == "a"


def test_role_list_skips_exhausted_model(tmp_path):
    models = {"a": gemini_model(rpd=10), "b": gemini_model(rpd=1000)}
    models["b"]["model_id"] = "fake-b"
    client, _, _ = make_client(tmp_path, ["x"], models=models, roles={"generator": ["a", "b"]})
    for _ in range(9):
        client.counter.increment("a")
    assert client.choose_model("generator") == "b"


def test_ordered_selection_uses_first_model_until_its_limit(tmp_path):
    models = {"a": gemini_model(rpd=10), "b": gemini_model(rpd=1000)}
    models["b"]["model_id"] = "fake-b"
    client, _, _ = make_client(tmp_path, ["x"], models=models, roles={"generator": ["a", "b"]})
    assert client.settings["model_selection"] == "ordered"  # the project default
    for _ in range(5):
        client.counter.increment("a")
    assert client.choose_model("generator") == "a"  # a has used more, but it comes first
    for _ in range(4):
        client.counter.increment("a")  # 9 of 10 = its 90% stop
    assert client.choose_model("generator") == "b"
    for _ in range(900):
        client.counter.increment("b")
    with pytest.raises(DailyLimitReached):
        client.choose_model("generator")


def test_cached_reply_on_second_model_wins_and_costs_nothing(tmp_path):
    models = {"a": gemini_model(rpd=100), "b": gemini_model(rpd=100)}
    models["b"]["model_id"] = "fake-b"
    client, provider, _ = make_client(tmp_path, ["from b", "from a"], models=models, roles={"generator": ["a", "b"]})
    for _ in range(50):
        client.counter.increment("a")
    client.settings["model_selection"] = "most_quota_left"
    first = client.call("question")  # b has more quota left -> b answers
    assert first.model_key == "b"

    client.settings["model_selection"] = "ordered"  # now a would be chosen for a NEW call
    again = client.call("question")
    assert again.model_key == "b" and again.cached and again.text == "from b"
    assert len(provider.calls) == 1
    assert client.call("other question").model_key == "a"


def test_unknown_selection_strategy_is_refused(tmp_path):
    client, _, _ = make_client(tmp_path, ["x"])
    client.settings["model_selection"] = "random"
    with pytest.raises(ConfigError):
        client.call("hi")


def test_null_model_id_is_refused(tmp_path):
    models = {"m1": dict(gemini_model(), model_id=None)}
    client, _, _ = make_client(tmp_path, ["x"], models=models)
    with pytest.raises(ConfigError):
        client.call("hi")


def test_unknown_role_is_refused(tmp_path):
    client, _, _ = make_client(tmp_path, ["x"])
    with pytest.raises(ConfigError):
        client.call("hi", role="nope")


# -- retries ----------------------------------------------------------------------


def test_429_is_retried_with_exponential_backoff(tmp_path):
    client, provider, clock = make_client(tmp_path, [RetryableError("429"), RetryableError("429"), "ok"])
    result = client.call("hi")
    assert result.text == "ok"
    assert result.retries == 2
    assert clock.sleeps[:2] == [5, 10]
    assert len(provider.calls) == 3


def test_gives_up_after_max_retries(tmp_path):
    errors = [RetryableError("429")] * 4  # first try + 3 retries
    client, provider, _ = make_client(tmp_path, errors)
    with pytest.raises(LLMError):
        client.call("hi")
    assert len(provider.calls) == 4


# -- JSON -------------------------------------------------------------------------------


def test_json_reply_is_validated_with_pydantic(tmp_path):
    client, _, _ = make_client(tmp_path, ['{"verdict": "sufficient", "confidence": 0.9}'])
    result = client.call("judge this", response_model=Answer)
    assert isinstance(result.parsed, Answer)
    assert result.parsed.verdict == "sufficient"


def test_malformed_json_is_retried_once_with_nudge(tmp_path):
    client, provider, _ = make_client(tmp_path, ["not json at all", '{"verdict": "ok", "confidence": 0.5}'])
    result = client.call("judge this", response_model=Answer)
    assert result.parsed.verdict == "ok"
    assert len(provider.calls) == 2
    assert "not valid JSON" in provider.calls[1]["prompt"]


def test_schema_violation_counts_as_malformed(tmp_path):
    client, provider, _ = make_client(tmp_path, ['{"verdict": "ok"}', '{"verdict": "ok", "confidence": 1}'])
    result = client.call("judge this", response_model=Answer)
    assert result.parsed.confidence == 1
    assert len(provider.calls) == 2


def test_json_failing_twice_raises(tmp_path):
    client, provider, _ = make_client(tmp_path, ["nope", "still nope"])
    with pytest.raises(LLMJSONError):
        client.call("judge this", response_model=Answer)
    assert len(provider.calls) == 2  # exactly one retry, never more


def test_json_mode_without_model_returns_dict(tmp_path):
    client, _, _ = make_client(tmp_path, ['{"a": 1}'])
    assert client.call("x", json_mode=True).parsed == {"a": 1}


def test_parse_json_text_handles_fences_and_extra_text():
    assert parse_json_text('```json\n{"a": 1}\n```', None) == {"a": 1}
    assert parse_json_text('Sure! Here it is: {"a": 2} Hope that helps', None) == {"a": 2}
    assert parse_json_text("no json", None) is None


# -- usage log ----------------------------------------------------------------------------


def test_usage_log_rows_and_notional_cost(tmp_path):
    client, _, _ = make_client(tmp_path, ["answer"])
    client.call("q", role="generator")
    client.call("q", role="generator")  # cached
    with open(tmp_path / "usage.csv", newline="", encoding="utf-8") as log_file:
        rows = list(csv.DictReader(log_file))
    assert [row["cached"] for row in rows] == ["False", "True"]
    assert rows[0]["role"] == "generator"
    assert rows[0]["input_tokens"] == "10" and rows[0]["output_tokens"] == "5"
    assert float(rows[0]["notional_cost_usd"]) == pytest.approx(10 / 1e6 * 1.0 + 5 / 1e6 * 2.0)


def test_cost_is_blank_without_prices(tmp_path):
    model = gemini_model()
    model["notional_usd_per_1m_input"] = None
    client, _, _ = make_client(tmp_path, ["answer"], models={"m1": model})
    client.call("q")
    with open(tmp_path / "usage.csv", newline="", encoding="utf-8") as log_file:
        assert list(csv.DictReader(log_file))[0]["notional_cost_usd"] == ""


# -- providers ------------------------------------------------------------------------------


def test_ollama_request_and_token_counts(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(url=url, payload=json)
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"message": {"content": "Paris"}, "prompt_eval_count": 12, "eval_count": 3},
        )

    monkeypatch.setattr(llm.requests, "post", fake_post)
    model_config = {"model_id": "qwen", "base_url": "http://x:1", "num_ctx": 4096}
    reply = OllamaProvider().generate(model_config, "capital?", "be brief", True, 0.0, 30)

    assert (reply.text, reply.input_tokens, reply.output_tokens) == ("Paris", 12, 3)
    assert sent["url"] == "http://x:1/api/chat"
    assert sent["payload"]["format"] == "json"
    assert sent["payload"]["options"]["num_ctx"] == 4096
    assert "num_predict" not in sent["payload"]["options"]  # no cap configured here
    assert sent["payload"]["messages"][0] == {"role": "system", "content": "be brief"}

    OllamaProvider().generate(dict(model_config, max_output_tokens=64), "capital?", None, False, 0.0, 30)
    assert sent["payload"]["options"]["num_predict"] == 64


def test_ollama_429_and_connection_errors_are_retryable(monkeypatch):
    model_config = {"model_id": "qwen", "base_url": "http://x:1", "num_ctx": 4096}
    monkeypatch.setattr(llm.requests, "post", lambda url, json, timeout: SimpleNamespace(status_code=429, text=""))
    with pytest.raises(RetryableError):
        OllamaProvider().generate(model_config, "p", None, False, 0.0, 30)

    def raise_connection_error(url, json, timeout):
        raise llm.requests.ConnectionError("down")

    monkeypatch.setattr(llm.requests, "post", raise_connection_error)
    with pytest.raises(RetryableError):
        OllamaProvider().generate(model_config, "p", None, False, 0.0, 30)


class FakeGeminiClient:
    def __init__(self):
        self.captured = {}
        self.models = self

    def generate_content(self, model, contents, config):
        self.captured = {"model": model, "contents": contents, "config": config}
        usage = SimpleNamespace(prompt_token_count=20, candidates_token_count=7, thoughts_token_count=3)
        return SimpleNamespace(text="reply", usage_metadata=usage)


def test_gemma_gets_system_text_in_prompt_and_no_json_mime(monkeypatch):
    provider = GeminiProvider()
    fake_client = FakeGeminiClient()
    monkeypatch.setattr(provider, "_get_client", lambda: fake_client)
    model_config = {"model_id": "gemma-x", "supports_system_instruction": False, "supports_json_mode": False}

    reply = provider.generate(model_config, "the question", "the system", True, 0.0, 30)

    assert fake_client.captured["contents"] == "the system\n\nthe question"
    assert fake_client.captured["config"].system_instruction is None
    assert fake_client.captured["config"].response_mime_type is None
    assert reply.output_tokens == 10  # 7 output + 3 thinking tokens


def test_flash_model_gets_system_field_and_json_mime(monkeypatch):
    provider = GeminiProvider()
    fake_client = FakeGeminiClient()
    monkeypatch.setattr(provider, "_get_client", lambda: fake_client)
    model_config = {"model_id": "flash-x", "supports_system_instruction": True, "supports_json_mode": True}

    provider.generate(model_config, "the question", "the system", True, 0.0, 30)

    assert fake_client.captured["contents"] == "the question"
    assert fake_client.captured["config"].system_instruction == "the system"
    assert fake_client.captured["config"].response_mime_type == "application/json"


def test_library_internal_retries_are_disabled(monkeypatch):
    provider = GeminiProvider()
    fake_client = FakeGeminiClient()
    monkeypatch.setattr(provider, "_get_client", lambda: fake_client)
    model_config = {"model_id": "flash-x", "supports_system_instruction": True, "supports_json_mode": True}

    provider.generate(model_config, "q", None, False, 0.0, 30)

    assert fake_client.captured["config"].http_options.retry_options.attempts == 1


def test_real_config_roles_all_reference_known_models():
    llm_config = load_config()["llm"]
    for role, candidates in llm_config["roles"].items():
        candidates = [candidates] if isinstance(candidates, str) else candidates
        for model_key in candidates:
            assert model_key in llm_config["models"], (role, model_key)
