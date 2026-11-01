"""Fine-tune DistilBERT to decide retrieve (1) vs skip (0) for a query.

Steps: load data/labels/necessity.csv, remove duplicate queries, make a stratified
80/10/10 split (saved as CSVs), train with the Hugging Face Trainer using class-weighted
loss and early stopping on validation macro-F1, and save the best model to
models/necessity/.
"""

import csv
import logging
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.classifier.evaluate import LABEL_TO_ID, class_metrics
from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

RARE_GROUP_MIN = 10  # label/source groups smaller than this are stratified by label only


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_labelled_rows(csv_path: Path, extra_csv_paths: Optional[List[Path]] = None) -> List[Dict[str, Any]]:
    """Read the label CSV (plus any extra CSVs that exist) and drop repeated queries (case-insensitive), keeping the first."""
    rows = []
    seen_queries = set()
    paths = [csv_path] + [path for path in (extra_csv_paths or []) if path.is_file()]
    for path in paths:
        with open(path, newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                key = row["query"].strip().lower()
                if key in seen_queries:
                    continue
                seen_queries.add(key)
                rows.append({"query": row["query"].strip(), "label": LABEL_TO_ID[row["label"]], "source": row["source"]})
    return rows


def stratify_keys(rows: List[Dict[str, Any]]) -> List[str]:
    """Stratify by label AND source, so every split gets the same mix of sources.

    A label/source group too small to split (fewer than RARE_GROUP_MIN rows, possibly a
    single row) is folded into the biggest group with the same label.
    """
    group_counts = Counter(f"{row['label']}|{row['source']}" for row in rows)

    biggest_group_for_label: Dict[int, str] = {}
    for key, count in group_counts.most_common():
        label = int(key.split("|")[0])
        if label not in biggest_group_for_label:
            biggest_group_for_label[label] = key

    keys = []
    for row in rows:
        key = f"{row['label']}|{row['source']}"
        if group_counts[key] < RARE_GROUP_MIN:
            key = biggest_group_for_label[row["label"]]
        keys.append(key)
    return keys


def split_rows(
    rows: List[Dict[str, Any]], val_fraction: float, test_fraction: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stratified train / validation / test split."""
    holdout_fraction = val_fraction + test_fraction
    train_rows, holdout_rows = train_test_split(
        rows, test_size=holdout_fraction, stratify=stratify_keys(rows), random_state=seed
    )
    val_rows, test_rows = train_test_split(
        holdout_rows,
        test_size=test_fraction / holdout_fraction,
        stratify=stratify_keys(holdout_rows),
        random_state=seed,
    )
    return train_rows, val_rows, test_rows


def select_challenge_rows(
    rows: List[Dict[str, Any]], challenge_sources: List[str], size: int, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Hold out part of the "hard example" rows as a challenge set that is never trained on.

    At most half of the hard rows are held out, so the other half can still teach the
    model. Stratified by label and source.

    Returns:
        ``(challenge_rows, remaining_rows)``.
    """
    candidates = [row for row in rows if row["source"] in challenge_sources]
    others = [row for row in rows if row["source"] not in challenge_sources]
    size = min(size, len(candidates) // 2)
    if size < 2:
        return [], rows
    remaining_candidates, challenge = train_test_split(
        candidates, test_size=size, stratify=stratify_keys(candidates), random_state=seed
    )
    return challenge, others + remaining_candidates


def save_split(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write one split to CSV (query, label as text, source)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    id_to_label = {value: key for key, value in LABEL_TO_ID.items()}
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["query", "label", "source"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"query": row["query"], "label": id_to_label[row["label"]], "source": row["source"]})


def load_split(path: Path) -> List[Dict[str, Any]]:
    """Read a split CSV written by ``save_split``."""
    with open(path, newline="", encoding="utf-8") as csv_file:
        return [
            {"query": row["query"], "label": LABEL_TO_ID[row["label"]], "source": row["source"]}
            for row in csv.DictReader(csv_file)
        ]


def balanced_class_weights(labels: List[int]) -> List[float]:
    """Weights n / (2 * count_c) so both classes contribute equally to the loss."""
    counts = Counter(labels)
    total = len(labels)
    return [total / (2 * counts[class_id]) for class_id in (0, 1)]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_weighted_trainer_class():
    """Build a Trainer subclass whose loss uses class weights (imported lazily: transformers is slow)."""
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        """Trainer with class-weighted cross-entropy loss."""

        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """Weighted cross-entropy (plain cross-entropy if no weights are set)."""
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            weight = None
            if self.class_weights is not None:
                weight = torch.tensor(self.class_weights, dtype=outputs.logits.dtype, device=outputs.logits.device)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def trainer_metrics(eval_prediction) -> Dict[str, float]:
    """Macro-F1 and retrieve recall at argmax, used by the Trainer for early stopping."""
    logits, labels = eval_prediction
    predicted = np.argmax(logits, axis=-1).tolist()
    labels = labels.tolist()
    retrieve = class_metrics(labels, predicted, positive=1)
    skip = class_metrics(labels, predicted, positive=0)
    return {"macro_f1": (retrieve["f1"] + skip["f1"]) / 2, "retrieve_recall": retrieve["recall"]}


def to_dataset(rows: List[Dict[str, Any]], tokenizer, max_length: int):
    """Tokenize rows into a Hugging Face Dataset."""
    from datasets import Dataset

    dataset = Dataset.from_dict({"text": [row["query"] for row in rows], "labels": [row["label"] for row in rows]})
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )


def train_classifier(config: Dict[str, Any], train_rows, val_rows) -> Dict[str, Any]:
    """Fine-tune the base model and save the best checkpoint to ``classifier.output_dir``.

    Returns:
        Facts about the run (device, epochs run, seconds, class weights, best val macro-F1).
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        TrainingArguments,
        set_seed,
    )

    settings = config["classifier"]
    set_seed(settings["seed"])
    output_dir = PROJECT_ROOT / settings["output_dir"]
    checkpoint_dir = output_dir / "checkpoints"

    tokenizer = AutoTokenizer.from_pretrained(settings["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        settings["base_model"],
        num_labels=2,
        id2label={0: "skip", 1: "retrieve"},
        label2id={"skip": 0, "retrieve": 1},
    )

    class_weights = None
    if settings["class_weighting"] == "balanced":
        class_weights = balanced_class_weights([row["label"] for row in train_rows])

    arguments = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=settings["epochs"],
        per_device_train_batch_size=settings["batch_size"],
        per_device_eval_batch_size=64,
        learning_rate=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        # transformers 5 removed warmup_ratio; a float below 1 here means "ratio of total steps".
        warmup_steps=settings["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=20,
        report_to="none",
        seed=settings["seed"],
    )
    trainer = make_weighted_trainer_class()(
        model=model,
        args=arguments,
        train_dataset=to_dataset(train_rows, tokenizer, settings["max_length"]),
        eval_dataset=to_dataset(val_rows, tokenizer, settings["max_length"]),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=settings["early_stopping_patience"])],
        class_weights=class_weights,
    )

    started = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - started

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    shutil.rmtree(checkpoint_dir, ignore_errors=True)  # best model is saved; checkpoints only waste space

    evaluated_epochs = [entry["epoch"] for entry in trainer.state.log_history if "eval_macro_f1" in entry]
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs_configured": settings["epochs"],
        "epochs_run": max(evaluated_epochs) if evaluated_epochs else None,
        "train_seconds": round(seconds, 1),
        "class_weights": class_weights,
        "best_val_macro_f1_at_argmax": trainer.state.best_metric,
        "val_history": [
            {"epoch": entry["epoch"], "macro_f1": entry["eval_macro_f1"], "retrieve_recall": entry["eval_retrieve_recall"]}
            for entry in trainer.state.log_history
            if "eval_macro_f1" in entry
        ],
    }
