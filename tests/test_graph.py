"""Tests for the LangGraph pipeline (section 8b). Every component is a fake: no models, no Qdrant."""

import pytest

from src.graph.pipeline import (
    build_graph,
    compose_answer,
    contexts_for_part,
    decide_stop,
    merge_contexts,
    missing_similarity,
    run_pipeline,
    select_contexts,
)

EARLY_STOP_OFF = {"no_new_chunks": False, "same_missing": False, "missing_similarity": 0.8}
EARLY_STOP_ON = {"no_new_chunks": True, "same_missing": True, "missing_similarity": 0.8}


class FakeStore:
    """Returns fixed hits per query and records every search."""

    def __init__(self, hits_by_query=None, default_hits=None):
        self.hits_by_query = hits_by_query or {}
        self.default_hits = default_hits or []
        self.searches = []

    def search(self, query, top_k):
        self.searches.append(query)
        return self.hits_by_query.get(query, self.default_hits)[:top_k]


class ExplodingStore:
    """Fails the test if the skip path ever searches."""

    def search(self, query, top_k):
        raise AssertionError("the skip path must not call the vector store")


def hit(chunk_id, score=0.5):
    return {"chunk_id": chunk_id, "title": chunk_id, "text": f"text of {chunk_id}", "score": score}


def fake_reformulate(query, feedback=None, previous_queries=None):
    if feedback:
        return {"queries": [f"retry {len(previous_queries)}"], "reason": "", "mode": "retry"}
    return {"queries": ["sub a", "sub b"], "reason": "", "mode": "decompose"}


def judge_always(verdict):
    calls = []

    def fake_judge(query, chunks):
        calls.append([chunk["chunk_id"] for chunk in chunks])
        return {"verdict": verdict, "missing": "the missing fact" if verdict == "not_sufficient" else ""}

    return fake_judge, calls


def fake_generate(query, contexts):
    return {"answer": f"answer from {len(contexts)} chunks"}


def build(store, classifier_decision=True, judge_fn=None, max_retries=2, early_stop=None, per_part=False, **kwargs):
    """Graph with fakes; early stop and per-part generation are off unless a test turns them on."""
    judge_fn = judge_fn or judge_always("sufficient")[0]
    return build_graph(
        store,
        classifier=lambda query: (classifier_decision, 0.9),
        reformulate_fn=fake_reformulate,
        judge_fn=judge_fn,
        generate_fn=fake_generate,
        max_retries=max_retries,
        top_k=2,
        max_contexts=6,
        early_stop=early_stop or EARLY_STOP_OFF,
        per_part_generation=per_part,
        per_part_contexts=2,
        **kwargs,
    )


def steps(state):
    return [entry["step"] for entry in state["trace"]]


# -- paths ---------------------------------------------------------------------------


def test_skip_path_never_calls_the_vector_store():
    judge_fn, judge_calls = judge_always("not_sufficient")
    state = run_pipeline(build(ExplodingStore(), classifier_decision=False, judge_fn=judge_fn), "capital of France?")
    assert steps(state) == ["classify", "generate"]
    assert state["answer"] == "answer from 0 chunks"
    assert judge_calls == []


def test_sufficient_on_first_pass_generates_without_retry():
    store = FakeStore(default_hits=[hit("c1"), hit("c2")])
    state = run_pipeline(build(store), "q")
    assert steps(state) == ["classify", "reformulate", "retrieve", "critique", "generate"]
    assert state["retries"] == 0
    assert store.searches == ["sub a", "sub b"]  # every sub-query is searched


@pytest.mark.parametrize("max_retries", [0, 1, 2, 3])
def test_loop_terminates_at_max_retries(max_retries):
    judge_fn, judge_calls = judge_always("not_sufficient")
    store = FakeStore(default_hits=[hit("c1")])
    state = run_pipeline(build(store, judge_fn=judge_fn, max_retries=max_retries), "q")
    assert state["retries"] == max_retries
    assert len(judge_calls) == max_retries + 1
    assert steps(state).count("reformulate") == max_retries + 1
    assert steps(state)[-1] == "generate"


def test_retry_uses_feedback_and_previous_queries():
    judge_fn, _ = judge_always("not_sufficient")
    state = run_pipeline(build(FakeStore(default_hits=[hit("c1")]), judge_fn=judge_fn, max_retries=1), "q")
    reformulations = [entry for entry in state["trace"] if entry["step"] == "reformulate"]
    assert reformulations[0]["mode"] == "decompose"
    assert reformulations[1]["mode"] == "retry" and reformulations[1]["queries"] == ["retry 2"]
    assert state["all_queries"] == ["sub a", "sub b", "retry 2"]


def test_classifier_off_always_retrieves():
    store = FakeStore(default_hits=[hit("c1")])
    state = run_pipeline(build(store, classifier_decision=False, use_classifier=False), "capital of France?")
    assert state["trace"][0]["decision"] == "retrieve" and not state["trace"][0]["classifier_used"]
    assert store.searches  # the store was used


# -- dedup and context selection -------------------------------------------------------


def test_chunks_are_deduplicated_across_sub_queries_and_retries():
    store = FakeStore(
        hits_by_query={"sub a": [hit("c1"), hit("c2")], "sub b": [hit("c2"), hit("c3")], "retry 2": [hit("c3"), hit("c4")]}
    )
    judge_fn, _ = judge_always("not_sufficient")
    state = run_pipeline(build(store, judge_fn=judge_fn, max_retries=1), "q")
    chunk_ids = [chunk["chunk_id"] for chunk in state["contexts"]]
    assert chunk_ids == ["c1", "c2", "c3", "c4"]
    retrievals = [entry for entry in state["trace"] if entry["step"] == "retrieve"]
    assert [entry["new_chunks"] for entry in retrievals] == [3, 1]


def test_merge_contexts_records_rank_and_round():
    merged, added = merge_contexts([], [[hit("a"), hit("b")], [hit("a"), hit("c")]], round_number=0)
    assert added == 3
    assert [(chunk["chunk_id"], chunk["rank"]) for chunk in merged] == [("a", 1), ("b", 2), ("c", 2)]


def test_select_contexts_keeps_every_sub_query_and_newest_round():
    contexts = [
        dict(hit("a1"), rank=1, round=0), dict(hit("a2"), rank=2, round=0), dict(hit("a3"), rank=3, round=0),
        dict(hit("b1"), rank=1, round=0),
        dict(hit("r1"), rank=1, round=1),
    ]
    chosen = [chunk["chunk_id"] for chunk in select_contexts(contexts, 3)]
    assert chosen == ["r1", "a1", "b1"]


def test_generator_gets_at_most_max_contexts():
    store = FakeStore(default_hits=[hit(f"c{i}") for i in range(10)])
    app = build_graph(
        store, classifier=lambda q: (True, 1.0), reformulate_fn=fake_reformulate,
        judge_fn=judge_always("sufficient")[0], generate_fn=fake_generate, max_retries=0, top_k=10, max_contexts=4,
        early_stop=EARLY_STOP_OFF, per_part_generation=False,
    )
    state = run_pipeline(app, "q")
    assert state["answer"] == "answer from 4 chunks"
    assert len(state["trace"][-1]["chunks_used"]) == 4


def test_trace_records_every_step_with_details():
    store = FakeStore(default_hits=[hit("c1")])
    state = run_pipeline(build(store), "q")
    classify, reformulate, retrieve, critique, generate = state["trace"]
    assert classify["decision"] == "retrieve" and classify["confidence"] == 0.9
    assert reformulate["queries"] == ["sub a", "sub b"]
    assert retrieve["total_chunks"] == 1
    assert critique["verdict"] == "sufficient" and critique["chunks_judged"] == ["c1"]
    assert generate["answer"].startswith("answer from")
    assert state["total_seconds"] >= 0


# -- early stop --------------------------------------------------------------------------


def judge_with_missing(texts):
    """Fake critic: always not_sufficient, with the given missing texts in order."""
    texts = list(texts)

    def fake_judge(query, chunks):
        return {"verdict": "not_sufficient", "missing": texts.pop(0)}

    return fake_judge


class GrowingStore:
    """Returns a fresh chunk for every search, so retries always add new chunks."""

    def __init__(self):
        self.count = 0

    def search(self, query, top_k):
        self.count += 1
        return [hit(f"c{self.count}")]


def critique_entries(state):
    return [entry for entry in state["trace"] if entry["step"] == "critique"]


def test_missing_similarity():
    assert missing_similarity("the GLUE score of BERT", "The GLUE score of BERT") == 1.0
    assert missing_similarity("the GLUE score", "") == 0.0
    assert missing_similarity("recall of MatRAG", "publication year of MatRAG") == pytest.approx(2 / 5)  # shared: "of", "matrag"


def test_decide_stop_rules():
    args = dict(max_retries=2, missing="a b c", previous_missing="a b c", last_new_chunks=3, early_stop=EARLY_STOP_ON)
    assert decide_stop("sufficient", retries=1, **args) == "sufficient"
    assert decide_stop("not_sufficient", retries=0, **args) is None          # first pass always retries
    assert decide_stop("not_sufficient", retries=1, **args) == "same_missing"
    assert decide_stop("not_sufficient", retries=2, **args) == "max_retries"
    assert decide_stop("not_sufficient", retries=1, **dict(args, last_new_chunks=0)) == "no_new_chunks"
    assert decide_stop("not_sufficient", retries=1, **dict(args, previous_missing="x y z")) is None
    assert decide_stop("not_sufficient", retries=1, **dict(args, early_stop=EARLY_STOP_OFF)) is None


def test_repeated_missing_stops_after_one_retry():
    judge_fn = judge_with_missing(["the GLUE score of BERT"] * 3)
    state = run_pipeline(build(GrowingStore(), judge_fn=judge_fn, max_retries=2, early_stop=EARLY_STOP_ON), "q")
    assert state["retries"] == 1
    assert [entry["stop_reason"] for entry in critique_entries(state)] == [None, "same_missing"]
    assert steps(state)[-1] == "generate"


def test_different_missing_keeps_retrying_to_max():
    judge_fn = judge_with_missing(["dates of A", "birthplace of B", "population of C"])
    state = run_pipeline(build(GrowingStore(), judge_fn=judge_fn, max_retries=2, early_stop=EARLY_STOP_ON), "q")
    assert state["retries"] == 2
    assert critique_entries(state)[-1]["stop_reason"] == "max_retries"


def test_no_new_chunks_stops():
    judge_fn = judge_with_missing(["dates of A", "birthplace of B", "population of C"])
    store = FakeStore(default_hits=[hit("c1")])  # every search returns the same chunk
    state = run_pipeline(build(store, judge_fn=judge_fn, max_retries=2, early_stop=EARLY_STOP_ON), "q")
    assert state["retries"] == 1
    assert critique_entries(state)[-1]["stop_reason"] == "no_new_chunks"


def test_early_stop_off_uses_every_retry():
    judge_fn = judge_with_missing(["same gap"] * 3)
    state = run_pipeline(build(GrowingStore(), judge_fn=judge_fn, max_retries=2, early_stop=EARLY_STOP_OFF), "q")
    assert state["retries"] == 2
    assert critique_entries(state)[-1]["stop_reason"] == "max_retries"


def test_switches_are_independent():
    args = dict(max_retries=2, missing="a b c", previous_missing="a b c", early_stop=None)
    only_chunks = {"no_new_chunks": True, "same_missing": False, "missing_similarity": 0.8}
    only_missing = {"no_new_chunks": False, "same_missing": True, "missing_similarity": 0.8}
    assert decide_stop("not_sufficient", retries=1, last_new_chunks=3, **dict(args, early_stop=only_chunks)) is None
    assert decide_stop("not_sufficient", retries=1, last_new_chunks=0, **dict(args, early_stop=only_chunks)) == "no_new_chunks"
    assert decide_stop("not_sufficient", retries=1, last_new_chunks=0, **dict(args, early_stop=only_missing)) == "same_missing"


def test_project_default_keeps_same_missing_off():
    from src.config import load_config

    early_stop = load_config()["graph"]["early_stop"]
    assert early_stop["no_new_chunks"] is True and early_stop["same_missing"] is False


def test_sufficient_records_stop_reason():
    state = run_pipeline(build(FakeStore(default_hits=[hit("c1")])), "q")
    assert critique_entries(state)[0]["stop_reason"] == "sufficient"


# -- per-part generation for decomposed questions ----------------------------------------


def test_merge_contexts_records_every_query_of_a_shared_chunk():
    merged, _ = merge_contexts([], [[hit("a"), hit("shared")], [hit("shared")]], 0, ["q1", "q2"])
    shared = [chunk for chunk in merged if chunk["chunk_id"] == "shared"][0]
    assert shared["queries"] == ["q1", "q2"] and shared["rank"] == 1


def test_contexts_for_part_uses_own_hits_plus_best_retry_hit():
    contexts = [
        dict(hit("a1"), rank=1, round=0, queries=["part a"]),
        dict(hit("a2"), rank=2, round=0, queries=["part a"]),
        dict(hit("a3"), rank=3, round=0, queries=["part a"]),
        dict(hit("b1"), rank=1, round=0, queries=["part b"]),
        dict(hit("r1"), rank=1, round=1, queries=["retry query"]),
    ]
    assert [c["chunk_id"] for c in contexts_for_part(contexts, "part a", 3)] == ["a1", "a2", "r1"]
    assert [c["chunk_id"] for c in contexts_for_part(contexts, "part b", 3)] == ["b1", "r1"]
    no_retry = [chunk for chunk in contexts if chunk["round"] == 0]
    assert [c["chunk_id"] for c in contexts_for_part(no_retry, "part a", 3)] == ["a1", "a2", "a3"]


def test_compose_answer_cites_chunks_and_marks_not_found():
    parts = [
        {"question": "Which benchmark has 1,300 utterances?", "answer": "SHAMS",
         "contexts": [{"chunk_id": "c1", "text": "The SHAMS benchmark has 1,300 utterances."}, {"chunk_id": "c2", "text": "other"}]},
        {"question": "Who wrote it?", "answer": "Not found.", "contexts": []},
    ]
    assert compose_answer(parts) == (
        "1) Which benchmark has 1,300 utterances? -> SHAMS [c1]\n"
        "2) Who wrote it? -> not found in the retrieved papers"
    )


def test_decomposed_question_is_answered_part_by_part():
    store = FakeStore(hits_by_query={"sub a": [hit("a1"), hit("a2")], "sub b": [hit("b1"), hit("b2")]})
    part_calls = []

    def fake_part(question, contexts):
        part_calls.append((question, [chunk["chunk_id"] for chunk in contexts]))
        return {"answer": f"answer to {question}"}

    app = build(store, per_part=True, part_generate_fn=fake_part)
    state = run_pipeline(app, "two-part question")
    assert part_calls == [("sub a", ["a1", "a2"]), ("sub b", ["b1", "b2"])]
    assert state["answer"] == "1) sub a -> answer to sub a\n2) sub b -> answer to sub b"
    generate_entry = state["trace"][-1]
    assert generate_entry["per_part"] and len(generate_entry["parts"]) == 2


def test_part_not_found_retries_with_all_chunks():
    store = FakeStore(hits_by_query={"sub a": [hit("a1")], "sub b": [hit("b1")]})
    calls = []

    def fake_part(question, contexts):
        ids = [chunk["chunk_id"] for chunk in contexts]
        calls.append((question, ids))
        if question == "sub b" and ids == ["b1"]:
            return {"answer": "not found"}
        return {"answer": "found"}

    state = run_pipeline(build(store, per_part=True, part_generate_fn=fake_part), "q")
    assert calls == [("sub a", ["a1"]), ("sub b", ["b1"]), ("sub b", ["a1", "b1"])]
    parts = state["trace"][-1]["parts"]
    assert parts[1]["used_all_chunks"] and parts[1]["answer"] == "found"
    assert not parts[0]["used_all_chunks"]


def test_part_still_not_found_keeps_not_found():
    store = FakeStore(hits_by_query={"sub a": [hit("a1")], "sub b": [hit("b1")]})
    state = run_pipeline(
        build(store, per_part=True, part_generate_fn=lambda q, c: {"answer": "Not found."}), "q"
    )
    assert state["answer"].count("not found in the retrieved papers") == 2


def test_rewritten_question_still_uses_one_generation_call():
    def rewrite_only(query, feedback=None, previous_queries=None):
        return {"queries": ["one query"], "reason": "", "mode": "rewrite"}

    def failing_part(question, contexts):
        raise AssertionError("per-part generation must not run for a rewrite")

    app = build_graph(
        FakeStore(default_hits=[hit("c1")]), classifier=lambda q: (True, 1.0), reformulate_fn=rewrite_only,
        judge_fn=judge_always("sufficient")[0], generate_fn=fake_generate, part_generate_fn=failing_part,
        max_retries=0, top_k=2, max_contexts=6, early_stop=EARLY_STOP_OFF, per_part_generation=True,
    )
    assert run_pipeline(app, "q")["answer"] == "answer from 1 chunks"


def test_parts_come_from_first_pass_not_retry():
    store = FakeStore(default_hits=[hit("c1")], hits_by_query={"retry 2": [hit("r1")]})
    judge_fn = judge_with_missing(["gap one", "gap two"])
    seen = []

    def fake_part(question, contexts):
        seen.append(question)
        return {"answer": "x"}

    app = build(store, judge_fn=judge_fn, max_retries=1, per_part=True, part_generate_fn=fake_part)
    run_pipeline(app, "q")
    assert seen == ["sub a", "sub b"]


def test_compose_answer_joins_multi_line_part_answers():
    parts = [{"question": "Which components?", "answer": "KOptim\nFIWARE\n",
              "contexts": [{"chunk_id": "c1", "text": "built on KOptim and FIWARE"}]}]
    assert compose_answer(parts) == "1) Which components? -> KOptim, FIWARE [c1]"
