"""Use the trained retrieval-necessity classifier.

    from src.classifier.predict import should_retrieve
    decision, confidence = should_retrieve("What FID does NEPA-DiT-XL reach on ImageNet?")

``confidence`` is the model's probability for the decision it returned
(P(retrieve) when deciding to retrieve, 1 - P(retrieve) when deciding to skip).
Probabilities are temperature-scaled (``models/necessity/calibration.json``) when available.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.classifier.calibration import probabilities_from_margins
from src.config import PROJECT_ROOT, load_config

THRESHOLD_FILE_NAME = "threshold.json"
CALIBRATION_FILE_NAME = "calibration.json"
SKIP_CLASS_ID = 0
RETRIEVE_CLASS_ID = 1


def resolve_threshold(config: Dict[str, Any], model_dir: Path) -> float:
    """Threshold from config if set, otherwise the one chosen on validation (threshold.json)."""
    configured = config["classifier"]["decision_threshold"]
    if configured is not None:
        return float(configured)
    threshold_path = model_dir / THRESHOLD_FILE_NAME
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"{threshold_path} not found. Train the classifier (scripts/train_classifier.py) "
            "or set classifier.decision_threshold in configs/config.yaml."
        )
    return float(json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"])


def load_temperature(model_dir: Path) -> float:
    """Temperature fitted on validation (calibration.json), or 1.0 (uncalibrated) if absent."""
    calibration_path = model_dir / CALIBRATION_FILE_NAME
    if not calibration_path.is_file():
        return 1.0
    return float(json.loads(calibration_path.read_text(encoding="utf-8"))["temperature"])


class NecessityClassifier:
    """Wraps the fine-tuned model: P(retrieve) for queries and a retrieve/skip decision."""

    def __init__(self, model, tokenizer, threshold: float, max_length: int = 64, temperature: float = 1.0):
        """Use an already-loaded model and tokenizer (see ``load`` for loading from disk).

        ``temperature`` softens the probabilities (temperature scaling); it never changes
        a decision at threshold 0.5.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.max_length = max_length
        self.temperature = temperature
        self.model.eval()

    @classmethod
    def load(
        cls, model_dir: Path, threshold: float, max_length: int = 64, temperature: Optional[float] = None
    ) -> "NecessityClassifier":
        """Load model and tokenizer saved by training (temperature from calibration.json if not given)."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        if temperature is None:
            temperature = load_temperature(model_dir)
        return cls(model, tokenizer, threshold, max_length, temperature)

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "NecessityClassifier":
        """Load the model from ``classifier.output_dir`` with the configured or chosen threshold."""
        config = config or load_config()
        model_dir = PROJECT_ROOT / config["classifier"]["output_dir"]
        threshold = resolve_threshold(config, model_dir)
        return cls.load(model_dir, threshold, config["classifier"]["max_length"])

    def predict_margins(self, queries: List[str], batch_size: int = 64) -> List[float]:
        """Logit margin (retrieve minus skip) for each query; > 0 means "retrieve" at threshold 0.5."""
        margins: List[float] = []
        for start in range(0, len(queries), batch_size):
            batch = queries[start : start + batch_size]
            encoded = self.tokenizer(batch, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**encoded).logits
            batch_margins = logits[:, RETRIEVE_CLASS_ID] - logits[:, SKIP_CLASS_ID]
            margins.extend(float(value) for value in batch_margins)
        return margins

    def predict_proba(self, queries: List[str], batch_size: int = 64) -> List[float]:
        """Calibrated P(retrieve) for each query (sigmoid of margin / temperature)."""
        return probabilities_from_margins(self.predict_margins(queries, batch_size), self.temperature)

    def should_retrieve(self, query: str) -> Tuple[bool, float]:
        """Decide whether ``query`` needs retrieval. Returns ``(decision, confidence)``."""
        retrieve_probability = self.predict_proba([query])[0]
        if retrieve_probability >= self.threshold:
            return True, retrieve_probability
        return False, 1.0 - retrieve_probability


_default_classifier: Optional[NecessityClassifier] = None


def should_retrieve(query: str) -> Tuple[bool, float]:
    """Module-level shortcut; loads the trained model on first use."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = NecessityClassifier.from_config()
    return _default_classifier.should_retrieve(query)
