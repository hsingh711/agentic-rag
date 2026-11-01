"""Block 2: query reformulation agent.

``reformulate(query)`` asks the ``reformulator`` LLM role to either rewrite a vague question
into one clear search query or decompose a multi-hop question into sub-questions.
``reformulate(query, feedback=...)`` is the retry mode used by the self-critique loop
(section 8): it writes new queries that target what the critic said was missing.

Replies are JSON validated with pydantic. The LLM wrapper already retries malformed JSON
once; if the second reply is also bad (or the model keeps failing), this module falls back
to a deterministic query so the pipeline can continue. A daily-limit stop is never hidden.
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.agents.prompts import REFORMULATOR_SYSTEM, build_reformulate_prompt, build_retry_prompt
from src.config import load_config
from src.llm import ConfigError, DailyLimitReached, LLMError, call_llm

logger = logging.getLogger(__name__)

ROLE = "reformulator"


class ReformulationReply(BaseModel):
    """First-pass reply: rewrite into one query, or decompose into sub-questions."""

    mode: Literal["rewrite", "decompose"] = "rewrite"
    queries: List[str] = Field(min_length=1)
    reason: str = ""


class RetryReply(BaseModel):
    """Retry reply: new queries aimed at the critic's missing information."""

    queries: List[str] = Field(min_length=1)
    reason: str = ""


def clean_queries(
    queries: List[str], max_queries: int, previous_queries: Optional[List[str]] = None
) -> List[str]:
    """Strip, drop empty and duplicate queries, drop repeats of earlier searches, cap the count.

    Duplicates are compared case-insensitively. If every query repeats an earlier search,
    the repeats are kept (an unchanged query is better than none).
    """
    seen = set()
    cleaned = []
    for query in queries:
        text = " ".join(query.split())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)

    if previous_queries:
        previous_keys = {" ".join(previous.split()).lower() for previous in previous_queries}
        new_only = [query for query in cleaned if query.lower() not in previous_keys]
        if new_only:
            cleaned = new_only

    return cleaned[:max_queries]


def fallback_queries(query: str, feedback: Optional[str]) -> List[str]:
    """Deterministic queries used when the LLM fails: the question, plus the feedback on retry."""
    if feedback:
        return [f"{query} {feedback}".strip()]
    return [query]


def _result(queries: List[str], reason: str, mode: str, llm_result: Any = None) -> Dict[str, Any]:
    """Build the return dict, with model and timing details when an LLM call was made."""
    result = {
        "queries": queries,
        "reason": reason,
        "mode": mode,
        "model": None,
        "latency_seconds": 0.0,
        "wait_seconds": 0.0,
        "tokens": 0,
        "cached": False,
    }
    if llm_result is not None:
        result["model"] = llm_result.model_key
        result["latency_seconds"] = llm_result.latency_seconds
        result["wait_seconds"] = llm_result.wait_seconds
        result["tokens"] = llm_result.total_tokens
        result["cached"] = llm_result.cached
    return result


def reformulate(
    query: str,
    feedback: Optional[str] = None,
    previous_queries: Optional[List[str]] = None,
    max_queries: Optional[int] = None,
) -> Dict[str, Any]:
    """Rewrite, decompose or retry a query.

    Args:
        query: The user's question.
        feedback: The critic's "missing information" text. When given, retry mode is used.
        previous_queries: Searches already run (retry mode avoids repeating them).
        max_queries: Upper bound on returned queries; defaults to ``reformulator.max_queries``.

    Returns:
        ``{"queries": [...], "reason": str, "mode": "rewrite"|"decompose"|"retry"|"fallback",
        "model", "latency_seconds", "wait_seconds", "tokens", "cached"}``.

    Raises:
        DailyLimitReached: the reformulator's daily budget is used up (callers stop and resume).
        ConfigError: the role or model is misconfigured.
    """
    if max_queries is None:
        max_queries = load_config()["reformulator"]["max_queries"]

    if feedback:
        prompt = build_retry_prompt(query, feedback, previous_queries or [], max_queries)
        response_model = RetryReply
    else:
        prompt = build_reformulate_prompt(query, max_queries)
        response_model = ReformulationReply

    try:
        llm_result = call_llm(prompt, system=REFORMULATOR_SYSTEM, role=ROLE, response_model=response_model)
    except (DailyLimitReached, ConfigError):
        raise
    except LLMError as error:
        # Includes LLMJSONError (bad JSON twice) and repeated provider failures.
        logger.warning("Reformulator failed (%s); using the fallback query.", error)
        return _result(fallback_queries(query, feedback), f"fallback after LLM error: {error}", "fallback")

    reply = llm_result.parsed
    queries = clean_queries(reply.queries, max_queries, previous_queries if feedback else None)
    if not queries:
        return _result(fallback_queries(query, feedback), "fallback: the model returned only empty queries", "fallback", llm_result)

    if feedback:
        mode = "retry"
    elif reply.mode == "decompose" and len(queries) > 1:
        mode = "decompose"
    else:
        # A "decomposition" into a single query is just a rewrite.
        mode = "rewrite"
    return _result(queries, reply.reason, mode, llm_result)
