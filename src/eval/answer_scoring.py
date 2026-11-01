"""Answer scoring for the evaluation harness (section 10).

Two scores are always reported side by side:
- strict: the section 5 rule (exact match / whole-word containment / token F1 >= threshold);
- relaxed: also accepts terse or abbreviated answers (reference contained in the answer, answer
  contained in the reference, most of the reference's words present, or an acronym of the
  reference). No LLM is used, so it is free and reproducible; RAGAS covers the meaning-based side.

Questions with no answer in the corpus (MultiHop-RAG ``null_query``) are scored on abstaining.
"""

import re
from typing import List

from src.classifier.scoring import contains_answer, normalize_answer, score_answer

ABSTAIN_PATTERN = re.compile(
    r"insufficient (evidence|information)|cannot find|can't find|could not find|not found|(cannot|can't|could not|not) be found|"
    r"no (relevant )?information|does not (contain|provide|mention|specify)|not (mentioned|provided|available|specified)|"
    r"unable to",
    re.IGNORECASE,
)
NULL_REFERENCE_PATTERN = re.compile(r"insufficient information", re.IGNORECASE)

# Words that carry no answer content when judging word coverage.
STOP_WORDS = {"a", "an", "the", "of", "in", "on", "and", "to", "is", "was", "for", "by", "with"}


def abstained(answer: str) -> bool:
    """True if the answer says the evidence is missing instead of answering."""
    return bool(ABSTAIN_PATTERN.search(answer))


def is_null_question(references: List[str]) -> bool:
    """True if the gold answer is "Insufficient information" (the corpus cannot answer it)."""
    return any(NULL_REFERENCE_PATTERN.search(reference) for reference in references)


def content_words(text: str) -> List[str]:
    """Normalised words of ``text`` without stop words."""
    return [word for word in normalize_answer(text).split() if word not in STOP_WORDS]


def acronym_match(answer: str, reference: str) -> bool:
    """True if one side is the initials of the other ("MRL" vs "Matryoshka Representation Learning")."""
    for short, long in ((answer, reference), (reference, answer)):
        short_token = normalize_answer(short).replace(" ", "")
        initials = "".join(word[0] for word in content_words(long))
        if len(short_token) >= 2 and short_token == initials:
            return True
    return False


def relaxed_match(answer: str, reference: str, word_coverage: float = 0.8) -> bool:
    """Relaxed comparison of one answer with one reference (see module docstring)."""
    if contains_answer(answer, reference):
        return True
    normalized_answer = normalize_answer(answer)
    normalized_reference = normalize_answer(reference)
    if len(normalized_answer) >= 3 and normalized_answer in normalized_reference:
        return True
    reference_words = content_words(reference)
    answer_words = set(content_words(answer))
    if reference_words and sum(word in answer_words for word in reference_words) / len(reference_words) >= word_coverage:
        return True
    return acronym_match(answer, reference)


def score_record(answer: str, references: List[str], f1_threshold: float) -> dict:
    """Strict and relaxed correctness (0/1) of one answer, plus whether it abstained."""
    did_abstain = abstained(answer)
    if is_null_question(references):
        correct = int(did_abstain)
        return {"strict": correct, "relaxed": correct, "abstained": did_abstain}
    if did_abstain:
        return {"strict": 0, "relaxed": 0, "abstained": True}
    strict, _ = score_answer(answer, references, f1_threshold)
    relaxed = int(bool(strict) or any(relaxed_match(answer, reference) for reference in references))
    return {"strict": int(strict), "relaxed": relaxed, "abstained": False}
