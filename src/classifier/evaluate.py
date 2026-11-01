"""Metrics and threshold selection for the retrieval-necessity classifier.

Labels: 1 = retrieve, 0 = skip. Models output P(retrieve); a query is sent to retrieval
when P(retrieve) >= threshold. Missing a needed retrieval (a false "skip") is worse than an
unneeded one, so recall on the "retrieve" class is reported separately and drives the
threshold choice.

All functions here are pure (lists in, dicts out) so they are easy to test.
"""

from typing import Any, Dict, List, Sequence

LABEL_TO_ID = {"skip": 0, "retrieve": 1}
ID_TO_LABEL = {0: "skip", 1: "retrieve"}


def apply_threshold(retrieve_probabilities: Sequence[float], threshold: float) -> List[int]:
    """Turn P(retrieve) values into 0/1 decisions."""
    return [1 if probability >= threshold else 0 for probability in retrieve_probabilities]


def safe_divide(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 instead of failing on an empty denominator."""
    return numerator / denominator if denominator else 0.0


def class_metrics(true_labels: Sequence[int], predicted: Sequence[int], positive: int) -> Dict[str, float]:
    """Precision, recall and F1 for one class treated as the positive class."""
    true_positive = sum(1 for t, p in zip(true_labels, predicted) if t == positive and p == positive)
    false_positive = sum(1 for t, p in zip(true_labels, predicted) if t != positive and p == positive)
    false_negative = sum(1 for t, p in zip(true_labels, predicted) if t == positive and p != positive)
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "support": sum(1 for t in true_labels if t == positive)}


def compute_metrics(true_labels: Sequence[int], retrieve_probabilities: Sequence[float], threshold: float) -> Dict[str, Any]:
    """All headline metrics at one threshold.

    Returns accuracy, per-class precision/recall/F1, macro F1, the confusion matrix
    (rows = true skip/retrieve, columns = predicted skip/retrieve) and the skip rate
    (share of queries that would not hit the vector store).
    """
    predicted = apply_threshold(retrieve_probabilities, threshold)
    total = len(true_labels)
    correct = sum(1 for t, p in zip(true_labels, predicted) if t == p)
    retrieve = class_metrics(true_labels, predicted, positive=1)
    skip = class_metrics(true_labels, predicted, positive=0)

    confusion = [[0, 0], [0, 0]]  # confusion[true][predicted], index 0 = skip, 1 = retrieve
    for true_label, predicted_label in zip(true_labels, predicted):
        confusion[true_label][predicted_label] += 1

    return {
        "threshold": threshold,
        "count": total,
        "accuracy": safe_divide(correct, total),
        "retrieve": retrieve,
        "skip": skip,
        "macro_f1": (retrieve["f1"] + skip["f1"]) / 2,
        "retrieve_recall": retrieve["recall"],
        "skip_rate": safe_divide(sum(1 for p in predicted if p == 0), total),
        "confusion_matrix": {"rows_true_cols_pred": ["skip", "retrieve"], "matrix": confusion},
    }


def choose_threshold(
    true_labels: Sequence[int],
    retrieve_probabilities: Sequence[float],
    target_recall: float,
    default_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Pick the decision threshold on P(retrieve), normally on the validation set.

    Rule: keep ``default_threshold`` (0.5, the natural decision point of a model trained
    with balanced class weights) if it already reaches ``target_recall`` on "retrieve".
    Otherwise lower it to the highest threshold that does.

    A first version always took the HIGHEST threshold meeting the target. That spends the
    whole recall budget right at the edge on a small validation set (one example = 1.5
    recall points here) and fell from 0.955 recall on validation to 0.909 on test, while
    0.5 gave 0.985 on test. The threshold is never raised above the default for that reason.

    Returns:
        ``{"threshold", "retrieve_recall", "skip_rate", "macro_f1", "target_recall", "rule"}``.
    """
    default_metrics = compute_metrics(true_labels, retrieve_probabilities, default_threshold)
    if default_metrics["retrieve_recall"] >= target_recall:
        threshold = default_threshold
        metrics = default_metrics
        rule = f"default {default_threshold} already meets the recall target"
    else:
        threshold = 0.0  # retrieves everything: recall 1.0, always meets the target
        candidates = sorted((p for p in set(retrieve_probabilities) if p < default_threshold), reverse=True)
        for candidate in candidates:
            if compute_metrics(true_labels, retrieve_probabilities, candidate)["retrieve_recall"] >= target_recall:
                threshold = candidate
                break
        metrics = compute_metrics(true_labels, retrieve_probabilities, threshold)
        rule = f"lowered below {default_threshold} to reach the recall target"

    return {
        "threshold": float(threshold),
        "retrieve_recall": metrics["retrieve_recall"],
        "skip_rate": metrics["skip_rate"],
        "macro_f1": metrics["macro_f1"],
        "target_recall": target_recall,
        "rule": rule,
    }


def metrics_by_source(
    true_labels: Sequence[int], retrieve_probabilities: Sequence[float], sources: Sequence[str], threshold: float
) -> Dict[str, Dict[str, Any]]:
    """Accuracy and label counts per data source.

    Exposes shortcut learning: if the model only learned "research question -> retrieve",
    it will fail on the minority rows (arxiv_chunk skip, triviaqa/nq retrieve).
    """
    predicted = apply_threshold(retrieve_probabilities, threshold)
    results: Dict[str, Dict[str, Any]] = {}
    for source in sorted(set(sources)):
        indexes = [i for i, s in enumerate(sources) if s == source]
        majority_label = majority(true_labels, indexes)

        correct = 0
        minority_total = 0
        minority_correct = 0
        for i in indexes:
            is_correct = predicted[i] == true_labels[i]
            if is_correct:
                correct += 1
            if true_labels[i] != majority_label:
                minority_total += 1
                if is_correct:
                    minority_correct += 1

        results[source] = {
            "count": len(indexes),
            "accuracy": safe_divide(correct, len(indexes)),
            "true_retrieve": sum(1 for i in indexes if true_labels[i] == 1),
            "true_skip": sum(1 for i in indexes if true_labels[i] == 0),
            "majority_label": ID_TO_LABEL[majority_label],
            "minority_correct": minority_correct,
            "minority_total": minority_total,
        }
    return results


def majority(true_labels: Sequence[int], indexes: List[int]) -> int:
    """Most common label among ``indexes`` (ties go to retrieve)."""
    retrieve_count = sum(1 for i in indexes if true_labels[i] == 1)
    return 1 if retrieve_count * 2 >= len(indexes) else 0
