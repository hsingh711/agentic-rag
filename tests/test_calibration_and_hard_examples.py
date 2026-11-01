"""Tests for calibration, bootstrap intervals, the challenge split and hard-example generation."""

import copy
import math
from types import SimpleNamespace

import pytest

from src.classifier import query_pool
from src.classifier.calibration import (
    bootstrap_interval,
    expected_calibration_error,
    fit_temperature,
    log_loss,
    probabilities_from_margins,
    sigmoid,
)
from src.classifier.predict import NecessityClassifier
from src.classifier.query_pool import (
    ConceptBatch,
    ConceptItem,
    QuestionBatch,
    QuestionItem,
    concept_queries_from_batch,
    generate_concept_questions,
    load_state,
    queries_from_batch,
    validate_concept_item,
)
from src.classifier.train import select_challenge_rows
from src.config import load_config
from src.ingest.io_utils import read_jsonl

# -- calibration --------------------------------------------------------------------------------


def test_sigmoid_is_safe_for_large_values():
    assert sigmoid(0) == pytest.approx(0.5)
    assert sigmoid(1000) == pytest.approx(1.0)
    assert sigmoid(-1000) == pytest.approx(0.0)


def test_overconfident_margins_get_temperature_above_one():
    # Margins of +-8 (probabilities ~0.9997) but 20% of predictions are wrong: overconfident.
    labels = [1] * 40 + [0] * 10 + [0] * 40 + [1] * 10
    margins = [8.0] * 50 + [-8.0] * 50
    temperature = fit_temperature(labels, margins)
    assert temperature > 1.5
    raw_loss = log_loss(labels, probabilities_from_margins(margins, 1.0))
    calibrated_loss = log_loss(labels, probabilities_from_margins(margins, temperature))
    assert calibrated_loss < raw_loss
    raw_ece = expected_calibration_error(labels, probabilities_from_margins(margins, 1.0))
    calibrated_ece = expected_calibration_error(labels, probabilities_from_margins(margins, temperature))
    assert calibrated_ece < raw_ece


def test_temperature_never_changes_decisions_at_half():
    margins = [-3.0, -0.1, 0.2, 5.0]
    for temperature in (0.5, 1.0, 4.0):
        decisions = [p >= 0.5 for p in probabilities_from_margins(margins, temperature)]
        assert decisions == [False, False, True, True]


def test_perfectly_calibrated_predictions_have_small_ece():
    labels = [1, 1, 1, 0] * 25  # 75% retrieve
    probabilities = [0.75] * 100
    assert expected_calibration_error(labels, probabilities) == pytest.approx(0.0, abs=1e-9)


def test_bootstrap_interval_brackets_the_point_estimate():
    labels = [1, 0] * 50
    probabilities = [0.9, 0.1] * 45 + [0.1, 0.9] * 5  # 90% accurate

    def accuracy(l, p):
        return sum(1 for a, b in zip(l, p) if (b >= 0.5) == (a == 1)) / len(l)

    interval = bootstrap_interval(labels, probabilities, accuracy, samples=300)
    assert interval["low"] <= 0.9 <= interval["high"]
    assert interval["high"] - interval["low"] < 0.2


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        import torch

        return {"input_ids": torch.zeros((len(texts), 3), dtype=torch.long)}


class FakeModel:
    def eval(self):
        return self

    def __call__(self, input_ids):
        import torch

        logits = torch.log(torch.tensor([0.2, 0.8]))  # margin = log(4)
        return SimpleNamespace(logits=logits.repeat(input_ids.shape[0], 1))


def test_predictor_applies_temperature():
    classifier = NecessityClassifier(FakeModel(), FakeTokenizer(), threshold=0.5, temperature=2.0)
    assert classifier.predict_margins(["q"]) == pytest.approx([math.log(4)])
    assert classifier.predict_proba(["q"]) == pytest.approx([2 / 3])  # sigmoid(log(4) / 2) = 2/3


# -- challenge split -------------------------------------------------------------------------------


def test_challenge_rows_are_held_out_and_at_most_half():
    rows = [{"query": f"concept {i}", "label": i % 2, "source": "concept_question"} for i in range(60)]
    rows += [{"query": f"plain {i}", "label": 1, "source": "plain_chunk"} for i in range(40)]
    rows += [{"query": f"arxiv {i}", "label": 1, "source": "arxiv_chunk"} for i in range(200)]
    challenge, remaining = select_challenge_rows(rows, ["concept_question", "plain_chunk"], size=80, seed=1)
    assert len(challenge) == 50  # capped at half of the 100 hard rows
    assert len(challenge) + len(remaining) == len(rows)
    assert not ({row["query"] for row in challenge} & {row["query"] for row in remaining})
    assert all(row["source"] != "arxiv_chunk" for row in challenge)
    assert {row["source"] for row in challenge} == {"concept_question", "plain_chunk"}


def test_no_challenge_when_no_hard_rows():
    rows = [{"query": f"q{i}", "label": 1, "source": "arxiv_chunk"} for i in range(10)]
    challenge, remaining = select_challenge_rows(rows, ["concept_question"], size=100, seed=1)
    assert challenge == [] and remaining == rows


# -- hard examples ------------------------------------------------------------------------------------


def test_validate_concept_item():
    assert validate_concept_item(ConceptItem(question="Which technique randomly zeroes activations during training?", answer="dropout"))
    assert not validate_concept_item(ConceptItem(question="What does dropout randomly zero during training?", answer="dropout"))
    assert not validate_concept_item(ConceptItem(question="Name the regularizer used in training?", answer="a very long answer with many words"))
    assert not validate_concept_item(ConceptItem(question="Dropout?", answer="regularizer"))


def test_concept_queries_dedupe_and_ids():
    item = ConceptItem(question="Which technique randomly zeroes activations during training?", answer="dropout")
    batch = ConceptBatch(items=[item, item])
    queries = concept_queries_from_batch(batch, topic_index=3, seen_questions=set())
    assert [query["query_id"] for query in queries] == ["concept:3:0"]
    assert queries[0]["source"] == "concept_question"


def test_plain_chunk_queries_use_their_own_source_and_prefix():
    chunk = {"chunk_id": "c9", "title": "GALA", "text": "The method reaches 60fps on mobile devices for avatar animation."}
    item = QuestionItem(excerpt=1, question="How fast can the new avatar animation method run on phones?", answer="60fps")
    queries = queries_from_batch(QuestionBatch(items=[item]), [chunk], set(), source="plain_chunk", id_prefix="plain")
    assert queries[0]["query_id"] == "plain:c9:0"
    assert queries[0]["source"] == "plain_chunk"


def test_generate_concept_questions_is_resumable(tmp_path, monkeypatch):
    config = copy.deepcopy(load_config())
    config["labels"]["hard_examples"]["concept_topics"] = ["topic a", "topic b", "topic c"]
    calls = []

    def fake_call(prompt, system=None, role=None, response_model=None, json_mode=False):
        calls.append(prompt)
        number = len(calls)
        item = ConceptItem(question=f"Which standard technique number {number} is used here?", answer=f"answer{number}")
        return SimpleNamespace(parsed=ConceptBatch(items=[item]))

    monkeypatch.setattr(query_pool, "call_llm", fake_call)
    pool_path, state_path = tmp_path / "pool.jsonl", tmp_path / "state.json"
    state = load_state(state_path)
    state["concept_topics_done"] = [1]  # topic b already done in an earlier run
    assert generate_concept_questions(config, pool_path, state, state_path) == 2
    assert len(calls) == 2 and "topic b" not in " ".join(calls)
    assert sorted(load_state(state_path)["concept_topics_done"]) == [0, 1, 2]
    assert len(list(read_jsonl(pool_path))) == 2
