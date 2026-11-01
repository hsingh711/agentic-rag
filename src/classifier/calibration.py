"""Probability calibration and confidence intervals for the necessity classifier.

* Temperature scaling: P(retrieve) = sigmoid(margin / T), where margin = logit_retrieve -
  logit_skip. One number T is fitted on the validation set by minimising log-loss. It does
  NOT change any decision at threshold 0.5 (margin 0 stays 0); it only makes the
  probabilities honest instead of almost always ~0 or ~1.
* Expected calibration error (ECE): average gap between confidence and accuracy over bins.
* Bootstrap confidence intervals for test metrics (the test set is small).

Plain Python + math only, so every step is easy to follow and test.
"""

import math
import random
from typing import Callable, Dict, List, Sequence

TEMPERATURE_GRID_SIZE = 200
TEMPERATURE_MIN = 0.25
TEMPERATURE_MAX = 50.0
PROBABILITY_EPSILON = 1e-12


def sigmoid(value: float) -> float:
    """Numerically safe logistic function."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def probabilities_from_margins(margins: Sequence[float], temperature: float) -> List[float]:
    """P(retrieve) for each margin after dividing by the temperature."""
    return [sigmoid(margin / temperature) for margin in margins]


def log_loss(true_labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Mean negative log-likelihood of the true labels."""
    total = 0.0
    for label, probability in zip(true_labels, probabilities):
        probability = min(max(probability, PROBABILITY_EPSILON), 1 - PROBABILITY_EPSILON)
        total += -math.log(probability if label == 1 else 1 - probability)
    return total / len(true_labels)


def fit_temperature(true_labels: Sequence[int], margins: Sequence[float]) -> float:
    """Find the temperature with the lowest validation log-loss (log-spaced grid search)."""
    best_temperature = 1.0
    best_loss = log_loss(true_labels, probabilities_from_margins(margins, 1.0))
    ratio = TEMPERATURE_MAX / TEMPERATURE_MIN
    for step in range(TEMPERATURE_GRID_SIZE + 1):
        temperature = TEMPERATURE_MIN * ratio ** (step / TEMPERATURE_GRID_SIZE)
        loss = log_loss(true_labels, probabilities_from_margins(margins, temperature))
        if loss < best_loss:
            best_temperature, best_loss = temperature, loss
    return best_temperature


def expected_calibration_error(true_labels: Sequence[int], probabilities: Sequence[float], bins: int = 10) -> float:
    """ECE over equal-width confidence bins (confidence = probability of the predicted class)."""
    bin_totals = [0] * bins
    bin_correct = [0] * bins
    bin_confidence = [0.0] * bins
    for label, probability in zip(true_labels, probabilities):
        predicted = 1 if probability >= 0.5 else 0
        confidence = probability if predicted == 1 else 1 - probability
        index = min(int((confidence - 0.5) / 0.5 * bins), bins - 1)  # confidence is in [0.5, 1]
        bin_totals[index] += 1
        bin_correct[index] += int(predicted == label)
        bin_confidence[index] += confidence

    error = 0.0
    for index in range(bins):
        if bin_totals[index] == 0:
            continue
        accuracy = bin_correct[index] / bin_totals[index]
        average_confidence = bin_confidence[index] / bin_totals[index]
        error += bin_totals[index] / len(true_labels) * abs(accuracy - average_confidence)
    return error


def bootstrap_interval(
    true_labels: Sequence[int],
    probabilities: Sequence[float],
    metric: Callable[[List[int], List[float]], float],
    samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 13,
) -> Dict[str, float]:
    """Percentile bootstrap interval of ``metric`` over resampled test sets.

    Returns:
        ``{"low", "high", "samples"}``.
    """
    rng = random.Random(seed)
    size = len(true_labels)
    values = []
    for _ in range(samples):
        indexes = [rng.randrange(size) for _ in range(size)]
        values.append(metric([true_labels[i] for i in indexes], [probabilities[i] for i in indexes]))
    values.sort()
    tail = (1 - confidence_level) / 2
    low_index = int(tail * samples)
    high_index = min(int((1 - tail) * samples), samples - 1)
    return {"low": values[low_index], "high": values[high_index], "samples": samples}
