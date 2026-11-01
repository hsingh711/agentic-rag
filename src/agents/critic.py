"""Block 3: retrieval sufficiency critic.

``judge(query, chunks)`` asks the ``critic`` LLM role whether the retrieved chunks are enough
to answer the question. The reply is JSON validated with pydantic:
``{verdict: sufficient|not_sufficient, missing: str, confidence: float}``.
The ``missing`` text is what the reformulator's retry mode searches for next.

If the critic cannot produce a valid reply (bad JSON twice, or repeated provider errors) the
verdict defaults to ``sufficient`` with ``fallback=True``: another retry round would cost more
quota and most likely fail the same way, and the grounded generator (section 9) still refuses
to answer when the evidence does not support an answer. A daily-limit stop is never hidden.
"""

import logging
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from src.agents.prompts import CRITIC_SYSTEM, build_critic_prompt
from src.llm import ConfigError, DailyLimitReached, LLMError, call_llm

logger = logging.getLogger(__name__)

ROLE = "critic"


class CriticReply(BaseModel):
    """The critic's validated JSON reply."""

    verdict: Literal["sufficient", "not_sufficient"]
    missing: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


def _result(verdict: str, missing: str, confidence: float, fallback: bool, llm_result: Any = None) -> Dict[str, Any]:
    """Build the return dict, with model and timing details when an LLM call was made."""
    result = {
        "verdict": verdict,
        "missing": missing,
        "confidence": confidence,
        "fallback": fallback,
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


def judge(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Decide whether ``chunks`` contain enough evidence to answer ``query``.

    Args:
        query: The user's question.
        chunks: Retrieved chunks, each with ``chunk_id``, ``title`` and ``text``.

    Returns:
        ``{"verdict", "missing", "confidence", "fallback", "model", "latency_seconds",
        "wait_seconds", "tokens", "cached"}``.

    Raises:
        DailyLimitReached: the critic's daily budget is used up (callers stop and resume).
        ConfigError: the role or model is misconfigured.
    """
    if not chunks:
        # Nothing to judge: no call needed.
        return _result("not_sufficient", f"any passage relevant to: {query}", 1.0, fallback=False)

    prompt = build_critic_prompt(query, chunks)
    try:
        llm_result = call_llm(prompt, system=CRITIC_SYSTEM, role=ROLE, response_model=CriticReply)
    except (DailyLimitReached, ConfigError):
        raise
    except LLMError as error:
        logger.warning("Critic failed (%s); treating the evidence as sufficient.", error)
        return _result("sufficient", "", 0.0, fallback=True)

    reply = llm_result.parsed
    missing = reply.missing.strip() if reply.verdict == "not_sufficient" else ""
    if reply.verdict == "not_sufficient" and not missing:
        missing = f"more specific evidence for: {query}"
    return _result(reply.verdict, missing, reply.confidence, fallback=False, llm_result=llm_result)
