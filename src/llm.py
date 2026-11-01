"""One thin wrapper for every LLM call in the project (free tiers only).

Main entry point::

    result = call_llm(prompt, system=None, json_mode=False, role="generator")

The ``role`` (generator, critic, judge, ...) is mapped to one or more models in
``configs/config.yaml``. The wrapper then takes care of:

  * caching every response on disk (an identical call is never sent twice),
  * throttling to a fraction of each model's requests/tokens per minute,
  * a persistent per-model daily request counter (resets at midnight Pacific Time)
    that stops at a fraction of the daily limit by raising ``DailyLimitReached``,
  * retrying HTTP 429 / timeouts with exponential backoff (bounded),
  * validating JSON replies with pydantic and retrying once if the reply is malformed,
  * logging tokens and latency of every call to a CSV file.

The code is single-threaded by design (rate limiter and counter are not thread-safe).
"""

import csv
import hashlib
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Type, Union
from zoneinfo import ZoneInfo

import httpx
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from src.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

RATE_WINDOW_SECONDS = 60.0

USAGE_LOG_COLUMNS = [
    "timestamp",
    "role",
    "model_key",
    "provider",
    "model_id",
    "input_tokens",
    "output_tokens",
    "latency_seconds",
    "wait_seconds",
    "retries",
    "cached",
    "notional_cost_usd",
]

JSON_RETRY_NUDGE = (
    "\n\nYour previous reply was not valid JSON. Reply again with ONLY the JSON object, "
    "no explanation and no code fences."
)


# ---------------------------------------------------------------------------
# Errors and result types
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base class for all errors raised by this module."""


class ConfigError(LLMError):
    """The configuration is incomplete (for example a model_id is still null)."""


class RetryableError(LLMError):
    """A temporary failure (HTTP 429, 5xx, timeout) that may succeed if tried again."""


class DailyLimitReached(LLMError):
    """Every model allowed for this role has used its daily request budget."""


class LLMJSONError(LLMError):
    """The model returned malformed JSON twice in a row."""


@dataclass
class ProviderReply:
    """Raw answer from a provider."""

    text: str
    input_tokens: int
    output_tokens: int


@dataclass
class LLMResult:
    """What ``call_llm`` returns.

    ``latency_seconds`` is the time of the model call itself (waiting for rate limits
    is reported separately in ``wait_seconds``). For cached results it is the latency
    measured when the call was originally made, so evaluations can still report it.
    """

    text: str
    parsed: Any
    model_key: str
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    wait_seconds: float
    retries: int
    cached: bool

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens."""
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


class ResponseCache:
    """Stores one JSON file per distinct call, so reruns and resumes never repeat a call."""

    def __init__(self, cache_dir: Path):
        """Create the cache folder if needed."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(model_id: str, system: Optional[str], prompt: str, json_mode: bool, temperature: float) -> str:
        """Hash everything that can change the answer into a short file name."""
        payload = json.dumps([model_id, system or "", prompt, json_mode, temperature], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Return the cached entry or None. A corrupt file counts as a miss."""
        path = self.cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Ignoring corrupt cache file %s", path)
            return None

    def put(self, key: str, entry: Dict[str, Any]) -> None:
        """Write an entry atomically (temp file, then rename)."""
        path = self.cache_dir / f"{key}.json"
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        os.replace(str(temp_path), str(path))


# ---------------------------------------------------------------------------
# Rate limiting and daily quota
# ---------------------------------------------------------------------------


class RateLimiter:
    """Sliding 60-second window limiter for requests and tokens of one model."""

    def __init__(
        self,
        rpm: int,
        tpm: int,
        fraction: float,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """Allow at most ``fraction`` of ``rpm`` requests and ``tpm`` tokens per minute."""
        self.max_requests = max(1, int(rpm * fraction))
        self.max_tokens = tpm * fraction
        self.clock = clock
        self.sleep = sleep
        self.entries: Deque[List[float]] = deque()  # each entry is [timestamp, tokens]

    def _drop_old_entries(self, now: float) -> None:
        """Forget calls that left the 60-second window."""
        while self.entries and now - self.entries[0][0] >= RATE_WINDOW_SECONDS:
            self.entries.popleft()

    def acquire(self, estimated_tokens: int) -> Tuple[List[float], float]:
        """Block until a call is allowed, then reserve a slot.

        Returns:
            ``(entry, waited_seconds)``. Pass the entry to ``record_actual_tokens``
            after the call so the window uses real token counts.
        """
        waited = 0.0
        while True:
            now = self.clock()
            self._drop_old_entries(now)
            used_tokens = sum(entry[1] for entry in self.entries)
            requests_ok = len(self.entries) < self.max_requests
            # An empty window always admits one call, even if it alone exceeds the token budget.
            tokens_ok = not self.entries or used_tokens + estimated_tokens <= self.max_tokens
            if requests_ok and tokens_ok:
                break
            pause = self.entries[0][0] + RATE_WINDOW_SECONDS - now + 0.05
            logger.info("Rate limit: waiting %.1fs", pause)
            self.sleep(pause)
            waited += pause
        entry = [self.clock(), float(estimated_tokens)]
        self.entries.append(entry)
        return entry, waited

    @staticmethod
    def record_actual_tokens(entry: List[float], actual_tokens: int) -> None:
        """Replace the estimate in ``entry`` with the real token count."""
        if actual_tokens > 0:
            entry[1] = float(actual_tokens)


class DailyCounter:
    """Persistent per-model request counter that resets at midnight in a given timezone."""

    def __init__(self, path: Path, timezone_name: str, today_fn: Optional[Callable[[], str]] = None):
        """Remember where the counter file lives and which timezone defines "a day"."""
        self.path = Path(path)
        self.timezone = ZoneInfo(timezone_name)
        self.today_fn = today_fn or (lambda: datetime.now(self.timezone).strftime("%Y-%m-%d"))

    def _load_counts(self) -> Dict[str, int]:
        """Read today's counts; a missing, corrupt or old file means zero everywhere."""
        if not self.path.is_file():
            return {}
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if state.get("date") != self.today_fn():
            return {}
        return dict(state.get("counts", {}))

    def count(self, model_key: str) -> int:
        """Requests sent to ``model_key`` today."""
        return int(self._load_counts().get(model_key, 0))

    def increment(self, model_key: str) -> int:
        """Add one request for ``model_key`` and save. Returns the new count."""
        counts = self._load_counts()
        counts[model_key] = int(counts.get(model_key, 0)) + 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(json.dumps({"date": self.today_fn(), "counts": counts}, indent=2), encoding="utf-8")
        os.replace(str(temp_path), str(self.path))
        return counts[model_key]


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class OllamaProvider:
    """Talks to a local Ollama server."""

    def generate(
        self,
        model_config: Dict[str, Any],
        prompt: str,
        system: Optional[str],
        json_mode: bool,
        temperature: float,
        timeout: float,
    ) -> ProviderReply:
        """Send one chat request to Ollama and return the reply with token counts."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "model": model_config["model_id"],
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": model_config["num_ctx"]},
        }
        if model_config.get("max_output_tokens"):
            payload["options"]["num_predict"] = model_config["max_output_tokens"]
        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(model_config["base_url"] + "/api/chat", json=payload, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as error:
            raise RetryableError(f"Ollama request failed: {error}") from error
        if response.status_code == 429 or response.status_code >= 500:
            raise RetryableError(f"Ollama returned HTTP {response.status_code}")
        if response.status_code != 200:
            raise LLMError(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")

        data = response.json()
        input_tokens = int(data.get("prompt_eval_count", 0))
        if input_tokens > 0.9 * model_config["num_ctx"]:
            logger.warning(
                "Prompt used %d of %d context tokens; Ollama may truncate it.", input_tokens, model_config["num_ctx"]
            )
        return ProviderReply(
            text=data["message"]["content"],
            input_tokens=input_tokens,
            output_tokens=int(data.get("eval_count", 0)),
        )


class GeminiProvider:
    """Talks to the Gemini / Gemma API through the google-genai package (free tier key)."""

    def __init__(self):
        """The API client is created on first use so tests and Ollama-only runs need no key."""
        self._client = None

    def _get_client(self):
        """Create the API client from GOOGLE_API_KEY (loaded from .env)."""
        if self._client is None:
            from google import genai

            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ConfigError("GOOGLE_API_KEY is not set. Add it to .env.")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def generate(
        self,
        model_config: Dict[str, Any],
        prompt: str,
        system: Optional[str],
        json_mode: bool,
        temperature: float,
        timeout: float,
    ) -> ProviderReply:
        """Send one request to the Gemini API and return the reply with token counts."""
        from google.genai import errors, types

        contents = prompt
        system_instruction = system
        if system and not model_config["supports_system_instruction"]:
            # Gemma models reject a separate system field, so the system text goes into the prompt.
            contents = f"{system}\n\n{prompt}"
            system_instruction = None

        mime_type = "application/json" if json_mode and model_config["supports_json_mode"] else None
        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            response_mime_type=mime_type,
            # attempts=1 switches off google-genai's own hidden retry loop (up to 5 tries per
            # call by default). Our wrapper must be the only one retrying, otherwise the daily
            # request counter undercounts what is really sent.
            http_options=types.HttpOptions(
                timeout=int(timeout * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

        try:
            response = self._get_client().models.generate_content(
                model=model_config["model_id"], contents=contents, config=generation_config
            )
        except errors.APIError as error:
            if error.code == 429 or (error.code is not None and error.code >= 500):
                raise RetryableError(f"Gemini API error {error.code}") from error
            raise LLMError(f"Gemini API error {error.code}: {error}") from error
        except (httpx.TimeoutException, httpx.TransportError) as error:
            raise RetryableError(f"Gemini request failed: {error}") from error

        usage = response.usage_metadata
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        # Thinking tokens are billed and throttled like output tokens, so they are counted here.
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) + int(
            getattr(usage, "thoughts_token_count", 0) or 0
        )
        return ProviderReply(text=response.text or "", input_tokens=input_tokens, output_tokens=output_tokens)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def parse_json_text(text: str, response_model: Optional[Type[BaseModel]]) -> Any:
    """Turn a model reply into a dict (or a validated pydantic object).

    Tolerates code fences and extra text around the JSON object.

    Returns:
        The parsed value, or None if the reply is not valid JSON or fails validation.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()

    data = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            try:
                data = json.loads(cleaned[first_brace : last_brace + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None

    if response_model is None:
        return data
    try:
        return response_model.model_validate(data)
    except ValidationError:
        return None


def estimate_tokens(prompt: str, system: Optional[str], expected_output_tokens: int) -> int:
    """Rough token estimate (4 characters per token) used only for throttling before a call."""
    return (len(prompt) + len(system or "")) // 4 + expected_output_tokens


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


class LLMClient:
    """Routes calls by role, applies cache, throttle, daily limit and retries, and logs usage."""

    def __init__(
        self,
        config: Dict[str, Any],
        providers: Optional[Dict[str, Any]] = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        today_fn: Optional[Callable[[], str]] = None,
    ):
        """Set up cache, counter, limiters and providers from the ``llm`` config section.

        ``providers``, ``clock``, ``sleep`` and ``today_fn`` can be replaced in tests.
        """
        self.settings = config["llm"]
        self.clock = clock
        self.sleep = sleep
        self.providers = providers or {"ollama": OllamaProvider(), "gemini": GeminiProvider()}
        self.cache = ResponseCache(PROJECT_ROOT / self.settings["cache_dir"])
        self.counter = DailyCounter(
            PROJECT_ROOT / self.settings["daily_counter_path"], self.settings["daily_reset_timezone"], today_fn
        )
        self.usage_log_path = PROJECT_ROOT / self.settings["usage_log_path"]
        self.limiters: Dict[str, RateLimiter] = {}

    # -- model selection ----------------------------------------------------

    def _model_config(self, model_key: str) -> Dict[str, Any]:
        """Look up a model and make sure it is usable."""
        models = self.settings["models"]
        if model_key not in models:
            raise ConfigError(f"Unknown model '{model_key}' in llm.roles")
        model_config = models[model_key]
        if not model_config.get("model_id"):
            raise ConfigError(
                f"model_id for '{model_key}' is not set in configs/config.yaml "
                "(copy the exact string from AI Studio 'Get code')."
            )
        return model_config

    def _daily_limit(self, model_config: Dict[str, Any]) -> Optional[int]:
        """Number of requests per day we allow ourselves for this model (None = unlimited)."""
        if not model_config.get("rpd"):
            return None
        return int(model_config["rpd"] * self.settings["daily_stop_fraction"])

    def choose_model(
        self, role: str, prompt: Optional[str] = None, system: Optional[str] = None, json_mode: bool = False
    ) -> str:
        """Pick the model that serves ``role`` for this call.

        1. Cache first: if ``prompt`` is given and any candidate already has a cached reply for
           exactly this call, that model is used (in list order). Re-runs are then free and
           always return the same answer, whichever model answered first.
        2. Otherwise ``llm.model_selection`` decides:
           - ``ordered`` (default): the first model in the role's list that still has quota.
             Results stay reproducible because one model serves the role until its daily stop.
           - ``most_quota_left``: the model with the smallest used fraction of its daily budget.

        Raises:
            DailyLimitReached: if every candidate has used its daily budget.
        """
        roles = self.settings["roles"]
        if role not in roles:
            raise ConfigError(f"Unknown role '{role}'. Known roles: {sorted(roles)}")
        candidates = roles[role]
        if isinstance(candidates, str):
            candidates = [candidates]

        if prompt is not None:
            for model_key in candidates:
                model_config = self._model_config(model_key)
                cache_key = ResponseCache.make_key(
                    model_config["model_id"], system, prompt, json_mode, self.settings["temperature"]
                )
                if self.cache.get(cache_key) is not None:
                    return model_key

        strategy = self.settings.get("model_selection", "ordered")
        if strategy not in ("ordered", "most_quota_left"):
            raise ConfigError(f"Unknown llm.model_selection '{strategy}' (use ordered or most_quota_left)")

        if strategy == "ordered":
            for model_key in candidates:
                model_config = self._model_config(model_key)
                limit = self._daily_limit(model_config)
                if limit is None or self.counter.count(model_key) < limit:
                    return model_key
            raise DailyLimitReached(f"All models for role '{role}' reached their daily request budget: {candidates}")

        best_key = None
        best_fraction = None
        for model_key in candidates:
            model_config = self._model_config(model_key)
            limit = self._daily_limit(model_config)
            used = self.counter.count(model_key) if limit is not None else 0
            if limit is not None and used >= limit:
                continue
            fraction = used / limit if limit else 0.0
            if best_fraction is None or fraction < best_fraction:
                best_key, best_fraction = model_key, fraction
        if best_key is None:
            raise DailyLimitReached(f"All models for role '{role}' reached their daily request budget: {candidates}")
        return best_key

    # -- single model call --------------------------------------------------

    def _limiter_for(self, model_key: str, model_config: Dict[str, Any]) -> Optional[RateLimiter]:
        """Return the rate limiter of a model (None for models without limits, like Ollama)."""
        if not model_config.get("rpm"):
            return None
        if model_key not in self.limiters:
            self.limiters[model_key] = RateLimiter(
                model_config["rpm"],
                model_config["tpm"],
                self.settings["throttle_fraction"],
                clock=self.clock,
                sleep=self.sleep,
            )
        return self.limiters[model_key]

    def _notional_cost(self, model_config: Dict[str, Any], input_tokens: int, output_tokens: int) -> Union[float, str]:
        """Paid-tier price of the call for reports (never charged); blank if no price is configured."""
        price_in = model_config.get("notional_usd_per_1m_input")
        price_out = model_config.get("notional_usd_per_1m_output")
        if price_in is None or price_out is None:
            return ""
        return round(input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out, 8)

    def _log_usage(self, role: str, model_key: str, model_config: Dict[str, Any], result: LLMResult) -> None:
        """Append one row to the usage CSV (header written on first use)."""
        self.usage_log_path.parent.mkdir(parents=True, exist_ok=True)
        is_new_file = not self.usage_log_path.is_file()
        with open(self.usage_log_path, "a", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=USAGE_LOG_COLUMNS)
            if is_new_file:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "role": role,
                    "model_key": model_key,
                    "provider": model_config["provider"],
                    "model_id": model_config["model_id"],
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "latency_seconds": round(result.latency_seconds, 4),
                    "wait_seconds": round(result.wait_seconds, 4),
                    "retries": result.retries,
                    "cached": result.cached,
                    "notional_cost_usd": self._notional_cost(model_config, result.input_tokens, result.output_tokens),
                }
            )

    def _call_model(
        self, model_key: str, role: str, prompt: str, system: Optional[str], json_mode: bool
    ) -> LLMResult:
        """Make one logical call to one model: cache, throttle, count, send, retry, log."""
        model_config = self._model_config(model_key)
        temperature = self.settings["temperature"]
        cache_key = ResponseCache.make_key(model_config["model_id"], system, prompt, json_mode, temperature)

        cached_entry = self.cache.get(cache_key)
        if cached_entry is not None:
            result = LLMResult(
                text=cached_entry["text"],
                parsed=None,
                model_key=model_key,
                model_id=model_config["model_id"],
                input_tokens=cached_entry["input_tokens"],
                output_tokens=cached_entry["output_tokens"],
                latency_seconds=cached_entry["latency_seconds"],
                wait_seconds=0.0,
                retries=0,
                cached=True,
            )
            self._log_usage(role, model_key, model_config, result)
            return result

        daily_limit = self._daily_limit(model_config)
        if daily_limit is not None and self.counter.count(model_key) >= daily_limit:
            raise DailyLimitReached(f"Model '{model_key}' reached its daily request budget ({daily_limit}).")

        provider = self.providers[model_config["provider"]]
        limiter = self._limiter_for(model_key, model_config)
        estimated = estimate_tokens(prompt, system, self.settings["expected_output_tokens"])

        retries = 0
        total_wait = 0.0
        while True:
            window_entry = None
            if limiter is not None:
                window_entry, waited = limiter.acquire(estimated)
                total_wait += waited
            if daily_limit is not None:
                self.counter.increment(model_key)

            started = time.perf_counter()
            try:
                reply = provider.generate(
                    model_config, prompt, system, json_mode, temperature, self.settings["timeout_seconds"]
                )
                break
            except RetryableError as error:
                if retries >= self.settings["max_retries"]:
                    raise LLMError(f"'{model_key}' failed after {retries} retries: {error}") from error
                pause = self.settings["backoff_base_seconds"] * (2**retries)
                logger.warning("'%s' failed (%s); retry %d in %.0fs", model_key, error, retries + 1, pause)
                self.sleep(pause)
                total_wait += pause
                retries += 1
        latency = time.perf_counter() - started

        if window_entry is not None:
            RateLimiter.record_actual_tokens(window_entry, reply.input_tokens + reply.output_tokens)

        self.cache.put(
            cache_key,
            {
                "text": reply.text,
                "input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens,
                "latency_seconds": latency,
                "model_key": model_key,
            },
        )
        result = LLMResult(
            text=reply.text,
            parsed=None,
            model_key=model_key,
            model_id=model_config["model_id"],
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            latency_seconds=latency,
            wait_seconds=total_wait,
            retries=retries,
            cached=False,
        )
        self._log_usage(role, model_key, model_config, result)
        return result

    # -- public call --------------------------------------------------------

    def call(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_mode: bool = False,
        role: str = "generator",
        response_model: Optional[Type[BaseModel]] = None,
    ) -> LLMResult:
        """Send a prompt to the model(s) assigned to ``role``.

        Args:
            prompt: The user prompt.
            system: Optional system instruction.
            json_mode: Ask for a JSON reply and parse it into ``result.parsed``.
            role: Key of ``llm.roles`` in the config.
            response_model: Optional pydantic class; implies ``json_mode`` and validates the reply.

        Returns:
            An ``LLMResult``. With JSON mode, ``parsed`` holds the dict or pydantic object.

        Raises:
            DailyLimitReached: daily budget used up for this role.
            LLMJSONError: the reply was invalid JSON twice (original call plus one retry).
            LLMError: other failures after bounded retries.
        """
        if response_model is not None:
            json_mode = True

        model_key = self.choose_model(role, prompt, system, json_mode)
        result = self._call_model(model_key, role, prompt, system, json_mode)
        if not json_mode:
            return result

        parsed = parse_json_text(result.text, response_model)
        if parsed is not None:
            result.parsed = parsed
            return result

        logger.warning("Malformed JSON from '%s'; retrying once.", model_key)
        retry_key = self.choose_model(role, prompt + JSON_RETRY_NUDGE, system, json_mode)
        retry_result = self._call_model(retry_key, role, prompt + JSON_RETRY_NUDGE, system, json_mode)
        parsed = parse_json_text(retry_result.text, response_model)
        if parsed is None:
            raise LLMJSONError(f"Model '{retry_key}' returned invalid JSON twice. Last reply: {retry_result.text[:200]}")
        retry_result.parsed = parsed
        return retry_result


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    """Return the shared client, creating it (and loading .env) on first use."""
    global _default_client
    if _default_client is None:
        load_dotenv(PROJECT_ROOT / ".env")
        _default_client = LLMClient(load_config())
    return _default_client


def call_llm(
    prompt: str,
    system: Optional[str] = None,
    json_mode: bool = False,
    role: str = "generator",
    response_model: Optional[Type[BaseModel]] = None,
) -> LLMResult:
    """Call the LLM assigned to ``role``. See ``LLMClient.call`` for details."""
    return get_client().call(prompt, system=system, json_mode=json_mode, role=role, response_model=response_model)
