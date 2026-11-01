"""Section 6: train the retrieval-necessity classifier, calibrate, choose the threshold, evaluate.

Run from the project root:
    python scripts/train_classifier.py              # split, train, calibrate, evaluate
    python scripts/train_classifier.py --eval-only  # calibrate + evaluate the saved model again

Writes models/necessity/ (model, threshold.json, calibration.json), data/labels/splits/*.csv
(train, val, test and the held-out challenge set) and results/classifier_metrics.json.
"""

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.calibration import (  # noqa: E402
    bootstrap_interval,
    expected_calibration_error,
    fit_temperature,
    log_loss,
    probabilities_from_margins,
)
from src.classifier.evaluate import choose_threshold, compute_metrics, metrics_by_source  # noqa: E402
from src.classifier.predict import CALIBRATION_FILE_NAME, THRESHOLD_FILE_NAME, NecessityClassifier  # noqa: E402
from src.classifier.train import (  # noqa: E402
    load_labelled_rows,
    load_split,
    save_split,
    select_challenge_rows,
    split_rows,
    train_classifier,
)
from src.config import PROJECT_ROOT, load_config  # noqa: E402

LATENCY_SAMPLE_SIZE = 50
SPLIT_NAMES = ("train", "val", "test", "challenge")


def describe_split(name, rows) -> str:
    """One line with the size and label balance of a split."""
    if not rows:
        return f"{name:9s} (empty)"
    retrieve = sum(1 for row in rows if row["label"] == 1)
    return f"{name:9s} {len(rows):5d} rows, retrieve {retrieve} ({retrieve / len(rows):.1%})"


def median_latency_ms(classifier: NecessityClassifier, queries) -> float:
    """Median time to classify ONE query (as the pipeline will do), in milliseconds."""
    classifier.predict_proba(queries[:1])  # warm-up
    timings = []
    for query in queries[:LATENCY_SAMPLE_SIZE]:
        started = time.perf_counter()
        classifier.predict_proba([query])
        timings.append((time.perf_counter() - started) * 1000)
    return statistics.median(timings)


def rounded(value):
    """Round floats inside nested dicts/lists for a readable JSON file."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def make_metric_function(metric_name: str, threshold: float):
    """Return a function (labels, probabilities) -> one metric value, for bootstrapping."""

    def metric(labels, probabilities):
        return compute_metrics(labels, probabilities, threshold)[metric_name]

    return metric


def confidence_intervals(labels, probabilities, threshold) -> dict:
    """95% bootstrap intervals for the headline metrics at ``threshold``."""
    intervals = {}
    for name in ("accuracy", "macro_f1", "retrieve_recall", "skip_rate"):
        intervals[name] = bootstrap_interval(labels, probabilities, make_metric_function(name, threshold))
    return intervals


def evaluate_split(rows, margins, temperature, threshold) -> dict:
    """Metrics, per-source breakdown, calibration and confidence intervals for one split."""
    labels = [row["label"] for row in rows]
    calibrated = probabilities_from_margins(margins, temperature)
    raw = probabilities_from_margins(margins, 1.0)
    return {
        "metrics": compute_metrics(labels, calibrated, threshold),
        "confidence_intervals_95": confidence_intervals(labels, calibrated, threshold),
        "by_source": metrics_by_source(labels, calibrated, [row["source"] for row in rows], threshold),
        "calibration": {
            "ece_raw": expected_calibration_error(labels, raw),
            "ece_calibrated": expected_calibration_error(labels, calibrated),
            "log_loss_raw": log_loss(labels, raw),
            "log_loss_calibrated": log_loss(labels, calibrated),
        },
    }


def print_split_report(name: str, report: dict) -> None:
    """Human-readable summary of ``evaluate_split`` output."""
    metrics = report["metrics"]
    intervals = report["confidence_intervals_95"]
    calibration = report["calibration"]
    print(f"\n{name.upper()} ({metrics['count']} queries) @ threshold {metrics['threshold']:.3f}")
    for key in ("accuracy", "macro_f1", "retrieve_recall", "skip_rate"):
        print(f"  {key:16s} {metrics[key]:.3f}  (95% CI {intervals[key]['low']:.3f} - {intervals[key]['high']:.3f})")
    print(f"  confusion [true skip, true retrieve] x [pred skip, pred retrieve]: {metrics['confusion_matrix']['matrix']}")
    print(f"  ECE raw {calibration['ece_raw']:.3f} -> calibrated {calibration['ece_calibrated']:.3f}")
    for source, result in report["by_source"].items():
        minority = "skip" if result["majority_label"] == "retrieve" else "retrieve"
        print(
            f"  {source:13s} n={result['count']:4d} accuracy {result['accuracy']:.3f}  "
            f"minority ({minority}) correct {result['minority_correct']}/{result['minority_total']}"
        )


def main() -> None:
    """Split, train, calibrate, choose the threshold on validation, evaluate on test and challenge."""
    parser = argparse.ArgumentParser(description="Train and evaluate the retrieval-necessity classifier.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; evaluate the saved model.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy_logger in ["httpx", "huggingface_hub", "datasets"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    config = load_config()
    settings = config["classifier"]
    labels_csv = PROJECT_ROOT / config["labels"]["dir"] / config["labels"]["output_csv"]
    splits_dir = PROJECT_ROOT / settings["splits_dir"]
    model_dir = PROJECT_ROOT / settings["output_dir"]
    metrics_path = PROJECT_ROOT / settings["metrics_file"]

    if args.eval_only:
        splits = {}
        for name in SPLIT_NAMES:
            path = splits_dir / f"{name}.csv"
            splits[name] = load_split(path) if path.is_file() else []
        # Keep the facts of the training run that produced the saved model.
        training_info = {"note": "evaluation re-run without training"}
        if metrics_path.is_file():
            training_info = json.loads(metrics_path.read_text(encoding="utf-8")).get("training", training_info)
    else:
        rows = load_labelled_rows(labels_csv, [PROJECT_ROOT / config["labels"]["dir"] / name for name in config["labels"].get("extra_csvs", [])])
        challenge_rows, remaining_rows = select_challenge_rows(
            rows, settings["challenge_sources"], settings["challenge_size"], settings["seed"]
        )
        train_rows, val_rows, test_rows = split_rows(
            remaining_rows, settings["val_fraction"], settings["test_fraction"], settings["seed"]
        )
        splits = {"train": train_rows, "val": val_rows, "test": test_rows, "challenge": challenge_rows}
        for name, split in splits.items():
            save_split(split, splits_dir / f"{name}.csv")
        print(f"Loaded {len(rows)} unique labelled queries from {labels_csv}")
        training_info = train_classifier(config, train_rows, val_rows)

    for name in SPLIT_NAMES:
        print(describe_split(name, splits[name]))

    classifier = NecessityClassifier.load(model_dir, threshold=0.5, max_length=settings["max_length"], temperature=1.0)
    margins = {name: classifier.predict_margins([row["query"] for row in rows]) for name, rows in splits.items() if rows}

    # Calibration and threshold are fitted on VALIDATION only.
    val_labels = [row["label"] for row in splits["val"]]
    temperature = fit_temperature(val_labels, margins["val"])
    calibration = {
        "temperature": temperature,
        "fitted_on": "validation",
        "val_log_loss_raw": log_loss(val_labels, probabilities_from_margins(margins["val"], 1.0)),
        "val_log_loss_calibrated": log_loss(val_labels, probabilities_from_margins(margins["val"], temperature)),
    }
    (model_dir / CALIBRATION_FILE_NAME).write_text(json.dumps(rounded(calibration), indent=2), encoding="utf-8")

    val_probabilities = probabilities_from_margins(margins["val"], temperature)
    chosen = choose_threshold(val_labels, val_probabilities, settings["target_retrieve_recall"])
    chosen["chosen_on"] = "validation"
    (model_dir / THRESHOLD_FILE_NAME).write_text(json.dumps(rounded(chosen), indent=2), encoding="utf-8")

    test_labels = [row["label"] for row in splits["test"]]
    always_retrieve = compute_metrics(test_labels, [1.0] * len(test_labels), 0.5)
    classifier.temperature = temperature
    metrics = {
        "training": training_info,
        "splits": {name: len(rows) for name, rows in splits.items()},
        "calibration": calibration,
        "threshold_selection": chosen,
        "test": evaluate_split(splits["test"], margins["test"], temperature, chosen["threshold"]),
        "baseline_always_retrieve_test": {
            "accuracy": always_retrieve["accuracy"],
            "macro_f1": always_retrieve["macro_f1"],
            "retrieve_recall": always_retrieve["retrieve_recall"],
            "skip_rate": always_retrieve["skip_rate"],
        },
        "cpu_latency_ms_per_query_median": median_latency_ms(classifier, [row["query"] for row in splits["test"]]),
    }
    if splits["challenge"]:
        metrics["challenge"] = evaluate_split(splits["challenge"], margins["challenge"], temperature, chosen["threshold"])

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(rounded(metrics), indent=2), encoding="utf-8")

    print(f"\nTemperature {temperature:.2f} (validation log-loss {calibration['val_log_loss_raw']:.3f} -> "
          f"{calibration['val_log_loss_calibrated']:.3f})")
    print(f"Threshold {chosen['threshold']:.3f}: {chosen['rule']} (val retrieve recall {chosen['retrieve_recall']:.3f})")
    print_split_report("test", metrics["test"])
    if "challenge" in metrics:
        print_split_report("challenge", metrics["challenge"])
    print(f"\nAlways-retrieve baseline on test: accuracy {always_retrieve['accuracy']:.3f}, macro-F1 {always_retrieve['macro_f1']:.3f}")
    print(f"CPU latency per query (median): {metrics['cpu_latency_ms_per_query_median']:.1f} ms")
    print(f"Training: {training_info}")
    print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
