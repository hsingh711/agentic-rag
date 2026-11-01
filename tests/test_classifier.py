"""Tests for section 6: metrics, threshold choice, splitting and the predictor (no training, no downloads)."""

import copy
import json
from collections import Counter

import pytest
import torch

from src.classifier.evaluate import choose_threshold, compute_metrics, metrics_by_source
from src.classifier.predict import NecessityClassifier, resolve_threshold
from src.classifier.train import balanced_class_weights, load_labelled_rows, split_rows, stratify_keys
from src.config import load_config

# -- metrics --------------------------------------------------------------------------------


def test_compute_metrics_known_values():
    labels = [1, 1, 1, 0, 0]
    probabilities = [0.9, 0.8, 0.2, 0.7, 0.1]  # at 0.5: TP=2, FN=1, FP=1, TN=1
    result = compute_metrics(labels, probabilities, 0.5)
    assert result["accuracy"] == pytest.approx(3 / 5)
    assert result["retrieve"]["precision"] == pytest.approx(2 / 3)
    assert result["retrieve_recall"] == pytest.approx(2 / 3)
    assert result["skip"]["recall"] == pytest.approx(1 / 2)
    assert result["confusion_matrix"]["matrix"] == [[1, 1], [1, 2]]
    assert result["skip_rate"] == pytest.approx(2 / 5)


def test_choose_threshold_keeps_default_when_recall_is_enough():
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    probabilities = [0.99, 0.98, 0.9, 0.6, 0.4, 0.2, 0.1, 0.05]
    chosen = choose_threshold(labels, probabilities, target_recall=0.75)
    # 0.5 gives recall 1.0; the threshold must NOT be raised to spend the recall budget.
    assert chosen["threshold"] == pytest.approx(0.5)
    assert chosen["retrieve_recall"] == pytest.approx(1.0)


def test_choose_threshold_lowers_until_target_is_met():
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    probabilities = [0.95, 0.9, 0.3, 0.2, 0.5, 0.25, 0.1, 0.05]
    chosen = choose_threshold(labels, probabilities, target_recall=0.75)  # 0.5 only gives 0.5 recall
    assert chosen["threshold"] == pytest.approx(0.3)
    assert chosen["retrieve_recall"] == pytest.approx(0.75)
    full = choose_threshold(labels, probabilities, target_recall=1.0)
    assert full["threshold"] == pytest.approx(0.2)


def test_metrics_by_source_counts_minority():
    labels = [1, 1, 0, 0, 0, 1]
    probabilities = [0.9, 0.9, 0.1, 0.1, 0.9, 0.1]
    sources = ["arxiv", "arxiv", "arxiv", "trivia", "trivia", "trivia"]
    result = metrics_by_source(labels, probabilities, sources, 0.5)
    assert result["arxiv"]["majority_label"] == "retrieve"
    assert (result["arxiv"]["minority_correct"], result["arxiv"]["minority_total"]) == (1, 1)
    assert result["trivia"]["majority_label"] == "skip"
    assert (result["trivia"]["minority_correct"], result["trivia"]["minority_total"]) == (0, 1)


# -- data and split -----------------------------------------------------------------------------


def make_rows():
    rows = []
    for index in range(300):
        rows.append({"query": f"research q{index}", "label": 1, "source": "arxiv_chunk"})
    for index in range(200):
        rows.append({"query": f"trivia q{index}", "label": 0, "source": "triviaqa"})
    for index in range(30):
        rows.append({"query": f"trivia retrieve q{index}", "label": 1, "source": "triviaqa"})
    for index in range(4):
        rows.append({"query": f"sum q{index}", "label": 1, "source": "arithmetic"})
    return rows


def test_split_is_80_10_10_disjoint_and_stratified():
    rows = make_rows()
    train, val, test = split_rows(rows, 0.1, 0.1, seed=1)
    assert len(train) + len(val) + len(test) == len(rows)
    assert abs(len(val) - len(rows) * 0.1) <= 2 and abs(len(test) - len(rows) * 0.1) <= 2
    queries = [set(row["query"] for row in part) for part in (train, val, test)]
    assert not (queries[0] & queries[1]) and not (queries[0] & queries[2]) and not (queries[1] & queries[2])
    # The rare "trivia that needs retrieval" group appears in validation and test.
    for part in (val, test):
        assert any(row["source"] == "triviaqa" and row["label"] == 1 for row in part)


def test_small_groups_join_biggest_group_of_same_label():
    rows = make_rows()
    keys = stratify_keys(rows)
    arithmetic_keys = {key for row, key in zip(rows, keys) if row["source"] == "arithmetic"}
    assert arithmetic_keys == {"1|arxiv_chunk"}  # 4 rows, too few to split on their own
    assert "1|triviaqa" in keys  # 30 rows, big enough


def test_single_row_group_does_not_break_split():
    rows = make_rows() + [{"query": "lonely", "label": 1, "source": "nq"}]
    train, val, test = split_rows(rows, 0.1, 0.1, seed=1)
    assert len(train) + len(val) + len(test) == len(rows)


def test_duplicate_queries_are_removed(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "query,label,closed_book_score,rag_score,source\n"
        "What is X?,retrieve,0,1,arxiv_chunk\nwhat is x? ,skip,1,1,triviaqa\nOther?,skip,1,0,nq\n",
        encoding="utf-8",
    )
    rows = load_labelled_rows(path)
    assert [row["query"] for row in rows] == ["What is X?", "Other?"]
    assert rows[0]["label"] == 1


def test_balanced_class_weights():
    weights = balanced_class_weights([1, 1, 1, 0])
    assert weights[0] == pytest.approx(2.0) and weights[1] == pytest.approx(4 / 6)


# -- predictor -------------------------------------------------------------------------------------


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": torch.zeros((len(texts), 3), dtype=torch.long)}


class FakeModel:
    """Returns logits that give P(retrieve) = 0.8 for every query."""

    def eval(self):
        return self

    def __call__(self, input_ids):
        logit = torch.log(torch.tensor([0.2, 0.8]))
        return type("Output", (), {"logits": logit.repeat(input_ids.shape[0], 1)})()


def test_should_retrieve_returns_decision_and_confidence():
    retrieve_side = NecessityClassifier(FakeModel(), FakeTokenizer(), threshold=0.5)
    assert retrieve_side.predict_proba(["a", "b"]) == pytest.approx([0.8, 0.8])
    decision, confidence = retrieve_side.should_retrieve("q")
    assert decision is True and confidence == pytest.approx(0.8)

    skip_side = NecessityClassifier(FakeModel(), FakeTokenizer(), threshold=0.9)
    decision, confidence = skip_side.should_retrieve("q")
    assert decision is False and confidence == pytest.approx(0.2)


def test_threshold_from_config_or_file(tmp_path):
    config = copy.deepcopy(load_config())
    config["classifier"]["decision_threshold"] = 0.42
    assert resolve_threshold(config, tmp_path) == pytest.approx(0.42)

    config["classifier"]["decision_threshold"] = None
    with pytest.raises(FileNotFoundError):
        resolve_threshold(config, tmp_path)
    (tmp_path / "threshold.json").write_text(json.dumps({"threshold": 0.31}), encoding="utf-8")
    assert resolve_threshold(config, tmp_path) == pytest.approx(0.31)
