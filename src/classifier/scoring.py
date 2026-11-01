"""Answer scoring against reference answers (no LLM calls).

Uses the usual SQuAD-style normalisation (lowercase, no punctuation, no articles).
Because the generator often answers with a phrase or sentence around the fact, an answer
also counts as correct when it CONTAINS a reference answer as whole words.
"""

import re
import string
from collections import Counter
from typing import List, Tuple

ARTICLES = re.compile(r"\b(a|an|the)\b")
DIGIT_SEPARATOR = re.compile(r"(?<=\d)[.,](?=\d)")


def normalize_answer(text: str) -> str:
    """Lowercase, turn punctuation into spaces, drop articles, collapse whitespace.

    Punctuation becomes a space (not deleted) so "8-expert" -> "8 expert" and still
    shares the token "8" with the answer "8". Decimal points and thousands separators
    inside numbers are removed first, so "0.843" stays one token ("0843") instead of
    splitting into "0" and "843" (which made "0.65" look half-correct).
    """
    text = text.lower()
    text = DIGIT_SEPARATOR.sub("", text)
    text = "".join(" " if character in string.punctuation else character for character in text)
    text = ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> bool:
    """True if both strings are equal after normalisation."""
    return normalize_answer(prediction) == normalize_answer(reference)


def token_f1(prediction: str, reference: str) -> float:
    """Word-overlap F1 between prediction and reference after normalisation."""
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_answer(prediction: str, reference: str) -> bool:
    """True if the normalised reference appears as whole words inside the normalised prediction."""
    normalized_reference = normalize_answer(reference)
    if not normalized_reference:
        return False
    return f" {normalized_reference} " in f" {normalize_answer(prediction)} "


def score_answer(prediction: str, references: List[str], f1_threshold: float) -> Tuple[int, float]:
    """Score one answer against all acceptable references.

    Returns:
        ``(correct, best_f1)`` where ``correct`` is 1 if any reference is matched exactly,
        contained in the answer, or reaches ``f1_threshold`` F1; otherwise 0.
    """
    best_f1 = 0.0
    correct = 0
    for reference in references:
        f1 = token_f1(prediction, reference)
        best_f1 = max(best_f1, f1)
        if exact_match(prediction, reference) or contains_answer(prediction, reference) or f1 >= f1_threshold:
            correct = 1
    return correct, best_f1
