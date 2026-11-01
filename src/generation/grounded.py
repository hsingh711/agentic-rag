"""Block 5: grounded generation.

``grounded_generate(query, contexts)`` answers only from the approved chunks and cites them by
chunk_id. With no contexts (the skip path) it falls back to a plain closed-book answer, because
there is no evidence to ground on.

The model's output is checked after the call:
- citations that do not name a provided chunk_id are removed and reported in ``invalid_citations``;
- a reply that says "insufficient evidence" is normalised to the configured text;
- an answer with no valid citation is flagged ``uncited`` (the verifier then checks it anyway).
"""

import re
from typing import Any, Dict, List, Tuple

from src.agents.prompts import (
    CLOSED_BOOK_SYSTEM,
    GROUNDED_SYSTEM,
    INSUFFICIENT_EVIDENCE,
    SHORT_GROUNDED_SYSTEM,
    build_closed_book_prompt,
    build_grounded_prompt,
    build_short_grounded_prompt,
)
from src.classifier.scoring import contains_answer, normalize_answer
from src.config import load_config
from src.llm import call_llm

CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def insufficient_text() -> str:
    """The configured refusal text (``generation.insufficient_text``)."""
    return load_config()["generation"].get("insufficient_text", INSUFFICIENT_EVIDENCE)


def is_insufficient(answer: str) -> bool:
    """True if the answer says the evidence is insufficient."""
    return "insufficient evidence" in answer.lower()


def clean_citations(answer: str, allowed_ids: List[str]) -> Tuple[str, List[str], List[str]]:
    """Keep only citations that name a provided chunk.

    A bracket group such as ``[id1, id2]`` is rewritten as ``[id1] [id2]``. Groups with no valid
    id (for example ``[1]`` or a made-up id) are removed from the text.

    Returns:
        The cleaned answer, the valid cited ids in order of first use, and the invalid ids found.
    """
    allowed = set(allowed_ids)
    valid: List[str] = []
    invalid: List[str] = []

    def rewrite(match: "re.Match[str]") -> str:
        kept = []
        for raw in re.split(r"[,;]", match.group(1)):
            chunk_id = raw.strip()
            if not chunk_id:
                continue
            if chunk_id in allowed:
                kept.append(chunk_id)
                if chunk_id not in valid:
                    valid.append(chunk_id)
            else:
                invalid.append(chunk_id)
        return " ".join(f"[{chunk_id}]" for chunk_id in kept)

    cleaned = CITATION_PATTERN.sub(rewrite, answer)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned).strip()
    return cleaned, valid, invalid


def grounded_generate(query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Answer ``query`` from ``contexts`` with citations; closed-book if there are no contexts.

    Returns:
        ``{"answer", "citations", "invalid_citations", "insufficient", "uncited", "grounded",
        "model", "latency_seconds", "wait_seconds", "tokens", "cached"}``.
    """
    if contexts:
        result = call_llm(build_grounded_prompt(query, contexts), system=GROUNDED_SYSTEM, role="generator")
        answer, citations, invalid = clean_citations(result.text.strip(), [chunk["chunk_id"] for chunk in contexts])
        insufficient = is_insufficient(answer) or not answer
        if insufficient:
            answer, citations = insufficient_text(), []
    else:
        result = call_llm(build_closed_book_prompt(query), system=CLOSED_BOOK_SYSTEM, role="generator")
        answer, citations, invalid, insufficient = result.text.strip(), [], [], False
    return {
        "answer": answer,
        "citations": citations,
        "invalid_citations": invalid,
        "insufficient": insufficient,
        "uncited": bool(contexts) and not insufficient and not citations,
        "grounded": bool(contexts),
        "model": result.model_key,
        "latency_seconds": result.latency_seconds,
        "wait_seconds": result.wait_seconds,
        "tokens": result.total_tokens,
        "cached": result.cached,
    }


SHORT_ANSWER_MAX_WORDS = 12


def citing_chunks(answer: str, contexts: List[Dict[str, Any]]) -> List[str]:
    """chunk_ids that support a short answer.

    A short answer (a name, number or phrase) is cited to every chunk that contains it as whole
    words. A longer answer is cited to the one chunk sharing the most of its words, if that is at
    least half of them. Nothing is cited when no chunk qualifies.
    """
    core = answer.strip().rstrip(".")
    if not core:
        return []
    if len(core.split()) <= SHORT_ANSWER_MAX_WORDS:
        return [chunk["chunk_id"] for chunk in contexts if contains_answer(chunk["text"], core)]
    answer_words = set(normalize_answer(core).split())
    best_id, best_share = None, 0.0
    for chunk in contexts:
        share = len(answer_words & set(normalize_answer(chunk["text"]).split())) / len(answer_words)
        if share > best_share:
            best_id, best_share = chunk["chunk_id"], share
    return [best_id] if best_id is not None and best_share >= 0.5 else []


def short_cited_generate(query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Block 5, short-answer style: the model writes a short answer, citations are attached afterwards.

    ``answer`` is the plain answer (no citation text, so scoring and verification see only the
    answer); ``citations`` lists the chunk_ids that contain it. With no contexts it answers
    closed-book, like ``grounded_generate``.
    """
    if not contexts:
        return grounded_generate(query, contexts)
    result = call_llm(build_short_grounded_prompt(query, contexts), system=SHORT_GROUNDED_SYSTEM, role="generator")
    answer = re.sub(r"\s+", " ", CITATION_PATTERN.sub("", result.text)).strip()
    insufficient = is_insufficient(answer) or not answer
    if insufficient:
        answer = insufficient_text()
    citations = [] if insufficient else citing_chunks(answer, contexts)
    return {
        "answer": answer,
        "citations": citations,
        "invalid_citations": [],
        "insufficient": insufficient,
        "uncited": not insufficient and not citations,
        "grounded": True,
        "model": result.model_key,
        "latency_seconds": result.latency_seconds,
        "wait_seconds": result.wait_seconds,
        "tokens": result.total_tokens,
        "cached": result.cached,
    }


def format_with_citations(answer: str, citations: List[str]) -> str:
    """The answer followed by ``[chunk_id]`` markers, for display (scoring uses the plain answer)."""
    return answer + "".join(f" [{chunk_id}]" for chunk_id in citations)
