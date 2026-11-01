"""Tests for section 5 (scoring, pool building, labelling loop). No network, no real models."""

import copy
import csv
from types import SimpleNamespace

import pytest

from src.classifier import make_labels, query_pool
from src.classifier.make_labels import decide_label, report, run_labelling, write_csv
from src.classifier.query_pool import (
    QuestionBatch,
    QuestionItem,
    generate_chunk_questions,
    load_state,
    make_arithmetic_queries,
    queries_from_batch,
    validate_generated_item,
)
from src.classifier.scoring import contains_answer, exact_match, normalize_answer, score_answer, token_f1
from src.config import load_config
from src.ingest.io_utils import append_jsonl, read_jsonl
from src.llm import DailyLimitReached

CHUNK = {
    "chunk_id": "2401.1:abstract:0",
    "title": "MatRAG",
    "text": "MatRAG organizes the corpus into a DAG of clusters and improves recall by 12.5 points on HotpotQA.",
}


# -- scoring ----------------------------------------------------------------------------


def test_normalize_answer():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("8-expert") == "8 expert"


def test_hyphenated_reference_still_matches_its_number():
    assert score_answer("8", ["8-expert"], 0.5)[0] == 1


def test_decimals_stay_one_token():
    assert normalize_answer("0.843") == "0843"
    assert normalize_answer("1,642 tools") == "1642 tools"
    assert score_answer("0.65", ["0.843"], 0.5)[0] == 0  # used to share the token "0"


def test_strict_threshold_rejects_near_miss_names():
    assert score_answer("Brian Wilson", ["Carl Wilson"], 0.8)[0] == 0
    assert score_answer("2 years", ["4 years"], 0.8)[0] == 0


def test_rescore_relabels_without_llm_calls(tmp_path):
    pool_path, details_path, csv_path = tmp_path / "pool.jsonl", tmp_path / "details.jsonl", tmp_path / "out.csv"
    append_jsonl(pool_path, [{"query_id": "q0", "query": "Youngest Beach Boy?", "references": ["Carl Wilson"], "source": "t"}])
    append_jsonl(
        details_path,
        [
            {
                "query_id": "q0", "query": "Youngest Beach Boy?", "source": "t", "references": ["Carl Wilson"],
                "closed_book_answer": "Brian Wilson", "rag_answer": "no idea",
                "closed_book_score": 1, "rag_score": 0, "label": "skip", "source_chunk_retrieved": None,
                "retrieved_chunk_ids": ["c1"],
            }
        ],
    )
    changes = make_labels.rescore(pool_path, details_path, csv_path, 0.8, {"c1": "unrelated text"})
    assert changes == {("skip", "dropped"): 1}
    assert list(read_jsonl(details_path))[0]["label"] == "dropped"


def test_rescore_drops_lucky_rag_answer_without_evidence(tmp_path):
    pool_path, details_path, csv_path = tmp_path / "pool.jsonl", tmp_path / "details.jsonl", tmp_path / "out.csv"
    append_jsonl(pool_path, [{"query_id": "q0", "query": "Third musketeer?", "references": ["Aramis"], "source": "t"}])
    append_jsonl(
        details_path,
        [
            {
                "query_id": "q0", "query": "Third musketeer?", "source": "t", "references": ["Aramis"],
                "closed_book_answer": "Richelieu", "rag_answer": "Aramis",
                "closed_book_score": 0, "rag_score": 1, "label": "retrieve", "source_chunk_retrieved": None,
                "retrieved_chunk_ids": ["c1"],
            }
        ],
    )
    changes = make_labels.rescore(pool_path, details_path, csv_path, 0.8, {"c1": "a paper on graph attention"})
    assert changes == {("retrieve", "dropped"): 1}


def test_exact_match_and_f1():
    assert exact_match("the Paris", "Paris")
    assert token_f1("Paris France", "Paris") == pytest.approx(2 / 3)
    assert token_f1("London", "Paris") == 0.0


def test_contains_answer_uses_whole_words():
    assert contains_answer("It is Paris, France.", "Paris")
    assert not contains_answer("1990", "1")  # no partial-number matches


def test_score_answer_any_reference_counts():
    assert score_answer("Big Apple", ["New York City", "Big Apple"], 0.5) == (1, 1.0)
    assert score_answer("unknown", ["Paris"], 0.5)[0] == 0


# -- labelling rule ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "closed_book, rag, evidence, expected",
    [
        (1, 1, True, "skip"),
        (1, 0, False, "skip"),
        (0, 1, True, "retrieve"),
        (0, 1, False, "dropped"),  # correct with context but answer not in the context: lucky guess
        (0, 0, True, "dropped"),
    ],
)
def test_decide_label(closed_book, rag, evidence, expected):
    assert decide_label(closed_book, rag, evidence) == expected


def test_evidence_contains_reference():
    assert make_labels.evidence_contains_reference(["MatRAG improves recall by 12.5 points"], ["12.5 points"])
    assert not make_labels.evidence_contains_reference(["A paper about retrieval"], ["Aramis"])


# -- pool -------------------------------------------------------------------------------


def test_arithmetic_queries_are_correct_and_deterministic():
    first = make_arithmetic_queries(6, seed=1)
    assert first == make_arithmetic_queries(6, seed=1)
    for query in first:
        words = query["query"].rstrip("?").split()
        left, operator, right = int(words[2]), words[3], int(words[4])
        expected = {"plus": left + right, "minus": left - right, "times": left * right}[operator]
        assert query["references"] == [str(expected)]


def test_generated_item_validation():
    good = QuestionItem(excerpt=1, question="By how many points does MatRAG improve recall on HotpotQA?", answer="12.5 points")
    assert validate_generated_item(good, CHUNK)
    not_in_text = QuestionItem(excerpt=1, question="Which conference published the MatRAG work?", answer="NeurIPS")
    assert not validate_generated_item(not_in_text, CHUNK)
    gives_away = QuestionItem(excerpt=1, question="Does MatRAG organize the corpus into a DAG of clusters?", answer="DAG of clusters")
    assert not validate_generated_item(gives_away, CHUNK)
    too_short = QuestionItem(excerpt=1, question="MatRAG gain?", answer="12.5 points")
    assert not validate_generated_item(too_short, CHUNK)


def test_queries_from_batch_drops_bad_excerpt_numbers_and_duplicates():
    item = QuestionItem(excerpt=1, question="By how many points does MatRAG improve recall on HotpotQA?", answer="12.5 points")
    batch = QuestionBatch(items=[item, item, QuestionItem(excerpt=7, question=item.question + " x", answer="12.5 points")])
    queries = queries_from_batch(batch, [CHUNK], set())
    assert len(queries) == 1
    assert queries[0]["query_id"] == "chunk:2401.1:abstract:0:0"
    assert queries[0]["source_chunk_id"] == CHUNK["chunk_id"]


def fake_result(parsed=None, text=""):
    return SimpleNamespace(parsed=parsed, text=text)


def test_generate_chunk_questions_saves_state_and_resumes(tmp_path, monkeypatch):
    chunks = [dict(CHUNK, chunk_id=f"c{i}") for i in range(5)]
    item = QuestionItem(excerpt=1, question="By how many points does MatRAG improve recall on HotpotQA?", answer="12.5 points")
    calls = []

    def fake_call(prompt, system=None, role=None, response_model=None, json_mode=False):
        calls.append(role)
        if len(calls) == 2:
            raise DailyLimitReached("limit")
        return fake_result(parsed=QuestionBatch(items=[item]))

    monkeypatch.setattr(query_pool, "call_llm", fake_call)
    pool_path, state_path = tmp_path / "pool.jsonl", tmp_path / "state.json"
    state = load_state(state_path)
    with pytest.raises(DailyLimitReached):
        generate_chunk_questions(chunks, pool_path, state, state_path, chunks_per_call=2, questions_per_chunk=1)

    saved = load_state(state_path)
    assert saved["chunks_done"] == ["c0", "c1"]  # only the finished call is marked done
    assert calls == ["label_generation", "label_generation"]
    assert len(list(read_jsonl(pool_path))) == 1


# -- labelling loop -----------------------------------------------------------------------


class FakeStore:
    def __init__(self):
        self.searches = 0

    def search(self, query, top_k):
        self.searches += 1
        return [{"chunk_id": "c1", "title": "T", "text": "MatRAG gains 12.5 on HotpotQA"}]


def make_fake_generator(answers):
    """closed-book answers come from answers[query][0], RAG answers from answers[query][1]."""

    def fake_call(prompt, system=None, role=None, **kwargs):
        assert role == "generator"
        for question, (closed_book, rag) in answers.items():
            if question in prompt:
                return SimpleNamespace(text=rag if "Context:" in prompt else closed_book)
        raise AssertionError("unexpected prompt")

    return fake_call


def labels_config(tmp_path, batch_size=2):
    config = copy.deepcopy(load_config())
    config["labels"]["batch_size"] = batch_size
    return config


def write_pool(path, questions):
    append_jsonl(
        path,
        [{"query_id": f"q{i}", "query": q, "references": [ref], "source": "test"} for i, (q, ref) in enumerate(questions)],
    )


def test_run_labelling_labels_writes_csv_and_resumes(tmp_path, monkeypatch):
    answers = {
        "Capital of France?": ("Paris", "Paris"),           # skip
        "MatRAG gain on HotpotQA?": ("unknown", "12.5"),   # retrieve
        "Obscure fact?": ("no idea", "still no idea"),     # dropped
    }
    monkeypatch.setattr(make_labels, "call_llm", make_fake_generator(answers))
    pool_path, details_path, csv_path = tmp_path / "pool.jsonl", tmp_path / "details.jsonl", tmp_path / "out.csv"
    write_pool(pool_path, [("Capital of France?", "Paris"), ("MatRAG gain on HotpotQA?", "12.5"), ("Obscure fact?", "42")])
    config = labels_config(tmp_path)
    store = FakeStore()

    assert run_labelling(config, store, pool_path, details_path, csv_path, limit=2) == 2
    assert run_labelling(config, store, pool_path, details_path, csv_path) == 1  # resumes with the last one
    assert run_labelling(config, store, pool_path, details_path, csv_path) == 0  # nothing left

    labels = {record["query"]: record["label"] for record in read_jsonl(details_path)}
    assert labels == {"Capital of France?": "skip", "MatRAG gain on HotpotQA?": "retrieve", "Obscure fact?": "dropped"}

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert list(rows[0].keys()) == ["query", "label", "closed_book_score", "rag_score", "source"]
    assert sorted(row["label"] for row in rows) == ["retrieve", "skip"]  # dropped rows excluded


def test_partial_batch_is_saved_when_a_call_fails(tmp_path, monkeypatch):
    calls = {"count": 0}

    def failing_call(prompt, system=None, role=None, **kwargs):
        calls["count"] += 1
        if calls["count"] > 2:  # first query = 2 calls succeed, second query fails
            raise DailyLimitReached("stop")
        return SimpleNamespace(text="Paris")

    monkeypatch.setattr(make_labels, "call_llm", failing_call)
    pool_path, details_path, csv_path = tmp_path / "pool.jsonl", tmp_path / "details.jsonl", tmp_path / "out.csv"
    write_pool(pool_path, [("Q one?", "Paris"), ("Q two?", "Paris")])
    with pytest.raises(DailyLimitReached):
        run_labelling(labels_config(tmp_path, batch_size=10), FakeStore(), pool_path, details_path, csv_path)
    assert len(list(read_jsonl(details_path))) == 1


@pytest.mark.parametrize("role_value", ["flash_lite_31", ["flash_lite_31", "flash_lite_35"]])
def test_estimate_accepts_single_model_or_list(tmp_path, role_value):
    config = copy.deepcopy(load_config())
    config["llm"]["roles"]["label_generation"] = role_value
    state = {"static_done": False, "chunks_done": [], "failed_calls": 0}
    text = make_labels.estimate(config, [CHUNK] * 8, tmp_path / "pool.jsonl", state, tmp_path / "details.jsonl")
    assert "2 calls for 8 chunks" in text
    assert "gemini-3.1-flash-lite" in text


def test_report_mentions_balance_and_samples(tmp_path):
    details_path = tmp_path / "details.jsonl"
    records = []
    for index, label in enumerate(["retrieve", "skip", "skip", "dropped"]):
        records.append(
            {
                "query_id": f"q{index}", "query": f"question {index}", "source": "test", "references": ["a"],
                "closed_book_answer": "x", "rag_answer": "y", "closed_book_score": 0, "rag_score": 0,
                "label": label, "source_chunk_retrieved": None,
            }
        )
    append_jsonl(details_path, records)
    text = report(details_path, sample_count=2)
    assert "retrieve      1" in text and "dropped       1" in text
    assert "Balance of kept rows: retrieve 33.3%" in text
