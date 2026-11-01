"""Block 6: answer verification.

``verify_answer`` splits an answer into claims and checks each claim against the contexts with an
NLI model (cross-encoder/nli-deberta-v3-base). NLI models are trained on single sentences and
score poorly against a whole 400-token chunk, so each claim is checked against the few
sentence windows of the contexts that share the most words with it (``windows_per_claim``).
A short extractive answer to a sub-question ("Q -> POD") is supported when its text appears in a
context (method ``string``), because a question plus a fragment is not a statement NLI can judge.
A claim is:
- ``supported``: some context entails it with probability >= ``verifier.supported_threshold``;
- ``contradicted``: not supported, and some context contradicts it (>= ``contradicted_threshold``);
- ``unsupported``: otherwise.

Optionally (``verifier.use_llm_fallback``) unsupported claims are re-checked by the
``verifier_fallback`` LLM role. That role has a tiny daily quota, so a daily-limit stop or a bad
reply simply keeps the NLI label.

The NLI model is behind a small scorer object (``score_pairs``), so tests use a fake scorer.
"""

import logging
import re
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple

from pydantic import BaseModel

from src.agents.prompts import (
    PART_NOT_FOUND,
    VERIFIER_FALLBACK_SYSTEM,
    build_verifier_fallback_prompt,
)
from src.config import load_config
from src.classifier.scoring import contains_answer
from src.eval.answer_scoring import abstained
from src.generation.grounded import CITATION_PATTERN, is_insufficient
from src.llm import ConfigError, DailyLimitReached, LLMError, call_llm

logger = logging.getLogger(__name__)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
NUMBER_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*")


class FallbackReply(BaseModel):
    """The fallback judge's validated JSON reply."""

    label: Literal["supported", "unsupported", "contradicted"]


class PairScorer(Protocol):
    """Anything that turns (premise, hypothesis) pairs into NLI probabilities."""

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """Return ``{"entailment", "neutral", "contradiction"}`` for each pair."""


class NLIScorer:
    """cross-encoder/nli-deberta-v3-base (or any NLI sequence classifier) run with transformers."""

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512, batch_size: int = 8):
        """Remember the settings; the model loads on first use."""
        self.model_name = model_name
        self.device_name = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._labels: Dict[int, str] = {}
        self._device = None

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "NLIScorer":
        """Build from the ``verifier`` block of the config."""
        settings = (config or load_config())["verifier"]
        return cls(settings["nli_model_name"], settings["device"], settings["max_length"], settings["batch_size"])

    def _load(self) -> None:
        """Load tokenizer and model once."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = self.device_name
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self._device).eval()
        # Read the label order from the model; it differs between NLI models.
        self._labels = {index: name.lower() for index, name in self._model.config.id2label.items()}
        missing = {"entailment", "neutral", "contradiction"} - set(self._labels.values())
        if missing:
            raise ConfigError(f"NLI model {self.model_name} lacks labels {sorted(missing)}: {self._labels}")

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """Probabilities of entailment / neutral / contradiction for each (premise, hypothesis)."""
        import torch

        if self._model is None:
            self._load()
        scores: List[Dict[str, float]] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = self._tokenizer(
                [premise for premise, _ in batch],
                [hypothesis for _, hypothesis in batch],
                truncation="only_first",
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                probabilities = torch.softmax(self._model(**encoded).logits, dim=-1).cpu().tolist()
            for row in probabilities:
                scores.append({self._labels[index]: float(value) for index, value in enumerate(row)})
        return scores


SHORT_ANSWER_WORDS = 6
# A yes/no span appears in almost any text, so finding it there proves nothing.
TRIVIAL_SPANS = {"yes", "no", "true", "false"}


def split_claims(answer: str, max_claims: Optional[int] = None, question: Optional[str] = None) -> List[Dict[str, Optional[str]]]:
    """Split an answer into checkable claims.

    Citations and list numbers are removed. A per-part line ``"1) question -> answer"`` becomes
    ``{"claim": "question answer", "span": "answer"}``; other claims have ``span`` None.
    When ``question`` is given, a one-line answer of at most ``SHORT_ANSWER_WORDS`` words (a bare
    fact such as "1,300") is treated the same way: the fragment alone is not a statement an NLI
    model can judge, but "question + fragment" is.
    Refusals ("Insufficient evidence", "not found") are not claims.
    """
    cleaned = CITATION_PATTERN.sub("", answer).strip()
    if question and cleaned and "\n" not in cleaned and "->" not in cleaned and len(cleaned.split()) <= SHORT_ANSWER_WORDS:
        if is_insufficient(cleaned) or abstained(cleaned) or PART_NOT_FOUND in cleaned.lower():
            return []
        if cleaned.rstrip(".").strip().lower() in TRIVIAL_SPANS:
            # "Yes" / "No" to a comparison question is not a statement NLI can verify.
            return []
        return [{"claim": f"{question.strip()} {cleaned}", "span": cleaned.rstrip(".")}]
    claims: List[Dict[str, Optional[str]]] = []
    for line in answer.splitlines():
        line = CITATION_PATTERN.sub("", line)
        line = re.sub(r"\s+([.,;:!?])", r"\1", line)
        line = NUMBER_PREFIX.sub("", line).strip()
        if not line:
            continue
        if "->" in line:
            question, _, part_answer = line.partition("->")
            if PART_NOT_FOUND in part_answer.lower():
                continue
            claims.append({"claim": f"{question.strip()} {part_answer.strip()}", "span": part_answer.strip()})
            continue
        claims.extend({"claim": sentence.strip(), "span": None} for sentence in SENTENCE_SPLIT.split(line) if sentence.strip())
    claims = [item for item in claims if not is_insufficient(item["claim"]) and not abstained(item["claim"])]
    return claims[:max_claims] if max_claims else claims


def sentence_windows(text: str, window_sentences: int) -> List[str]:
    """Overlapping windows of ``window_sentences`` consecutive sentences (stride 1)."""
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT.split(text) if sentence.strip()]
    if len(sentences) <= window_sentences:
        return [" ".join(sentences)] if sentences else []
    return [" ".join(sentences[start : start + window_sentences]) for start in range(len(sentences) - window_sentences + 1)]


def best_windows(claim: str, contexts: List[Dict[str, Any]], window_sentences: int, top_n: int) -> List[Tuple[str, str]]:
    """The ``top_n`` (chunk_id, window) pairs that share the most words with the claim."""
    claim_words = set(re.findall(r"\w+", claim.lower()))
    scored = []
    for chunk in contexts:
        for window in sentence_windows(chunk["text"], window_sentences):
            overlap = len(claim_words & set(re.findall(r"\w+", window.lower())))
            scored.append((overlap, chunk["chunk_id"], window))
    scored.sort(key=lambda item: -item[0])
    return [(chunk_id, window) for _, chunk_id, window in scored[:top_n]]


def classify_claim(entailment: float, contradiction: float, supported_threshold: float, contradicted_threshold: float) -> str:
    """Label from the best entailment and best contradiction probability over all contexts."""
    if entailment >= supported_threshold:
        return "supported"
    if contradiction >= contradicted_threshold:
        return "contradicted"
    return "unsupported"


def llm_check_claim(claim: str, contexts: List[Dict[str, Any]]) -> Optional[str]:
    """Ask the ``verifier_fallback`` role about one claim; None if it cannot answer."""
    try:
        result = call_llm(
            build_verifier_fallback_prompt(claim, contexts),
            system=VERIFIER_FALLBACK_SYSTEM,
            role="verifier_fallback",
            response_model=FallbackReply,
        )
    except (DailyLimitReached, ConfigError, LLMError) as error:
        logger.warning("Verifier fallback unavailable (%s); keeping the NLI label.", error)
        return None
    return result.parsed.label


def verify_answer(
    answer: str,
    contexts: List[Dict[str, Any]],
    scorer: PairScorer,
    settings: Optional[Dict[str, Any]] = None,
    fallback_fn=llm_check_claim,
    question: Optional[str] = None,
) -> Dict[str, Any]:
    """Check every claim of ``answer`` against ``contexts``.

    Args:
        answer: The generated answer (citations allowed).
        contexts: Chunks the answer was generated from (``chunk_id`` and ``text``).
        scorer: NLI scorer (``NLIScorer`` or a fake).
        settings: The ``verifier`` config block (default: loaded from the config).
        fallback_fn: ``(claim, contexts) -> label or None``, used when ``use_llm_fallback`` is on.
        question: The user question; lets bare short answers be checked as "question + answer".

    Returns:
        ``{"status", "claims": [{claim, label, entailment, contradiction, chunk_id, method}],
        "n_claims", "n_supported", "supported_ratio"}``. ``status`` is ``verified``,
        ``no_claims`` (a refusal or empty answer) or ``no_contexts`` (nothing to check against).
    """
    settings = settings or load_config()["verifier"]
    claims = split_claims(answer, settings["max_claims"], question)
    if not contexts:
        return {"status": "no_contexts", "claims": [], "n_claims": 0, "n_supported": 0, "supported_ratio": None}
    if not claims:
        return {"status": "no_claims", "claims": [], "n_claims": 0, "n_supported": 0, "supported_ratio": None}

    window_sentences = settings.get("window_sentences", 2)
    windows_per_claim = settings.get("windows_per_claim", 4)

    # Score every (window, claim) pair in one batch, then read the scores back per claim.
    claim_windows = [best_windows(item["claim"], contexts, window_sentences, windows_per_claim) for item in claims]
    pairs = [(window, item["claim"]) for item, windows in zip(claims, claim_windows) for _, window in windows]
    scores = scorer.score_pairs(pairs)

    results = []
    offset = 0
    for item, windows in zip(claims, claim_windows):
        claim_scores = scores[offset : offset + len(windows)]
        offset += len(windows)
        entailment, contradiction, chunk_id = 0.0, 0.0, contexts[0]["chunk_id"]
        if claim_scores:
            best = max(range(len(windows)), key=lambda index: claim_scores[index]["entailment"])
            entailment, chunk_id = claim_scores[best]["entailment"], windows[best][0]
            contradiction = max(score["contradiction"] for score in claim_scores)
        label = classify_claim(entailment, contradiction, settings["supported_threshold"], settings["contradicted_threshold"])
        method = "nli"
        if item["span"] and item["span"].strip().lower().rstrip(".") not in TRIVIAL_SPANS:
            found_in = [chunk["chunk_id"] for chunk in contexts if contains_answer(chunk["text"], item["span"])]
            if found_in:
                label, method, chunk_id, entailment = "supported", "string", found_in[0], 1.0
        if label != "supported" and settings.get("use_llm_fallback"):
            fallback_label = fallback_fn(item["claim"], contexts)
            if fallback_label is not None:
                label, method = fallback_label, "llm"
        results.append({
            "claim": item["claim"],
            "label": label,
            "entailment": round(entailment, 3),
            "contradiction": round(contradiction, 3),
            "chunk_id": chunk_id,
            "method": method,
        })

    supported = sum(1 for item in results if item["label"] == "supported")
    return {
        "status": "verified",
        "claims": results,
        "n_claims": len(results),
        "n_supported": supported,
        "supported_ratio": supported / len(results),
    }
