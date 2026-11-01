"""Tests for the section 10 evaluation pieces: benchmark data, answer scoring, harness summaries. No models."""

from src.classifier.train import load_labelled_rows
from src.eval import benchmarks
from src.eval.answer_scoring import abstained, acronym_match, is_null_question, relaxed_match, score_record
from src.eval.harness import build_record, gold_recall, summarize, trace_fields, verification_fields
from src.ingest.chunker import WhitespaceTokenizer


# -- answer scoring ----------------------------------------------------------------------


def test_abstained_detects_refusals():
    assert abstained("Insufficient evidence")
    assert abstained("I cannot find this in the context.")
    assert not abstained("The answer is Paris.")


def test_relaxed_accepts_terse_and_abbreviated_answers():
    assert relaxed_match("MRL", "Matryoshka Representation Learning")
    assert relaxed_match("Matryoshka Representation Learning (MRL)", "MRL")
    assert relaxed_match("KOptim and FIWARE", "KOptim / FIWARE")
    assert relaxed_match("a curvature-aware objective", "curvature-aware")
    assert not relaxed_match("Paris", "London")


def test_acronym_match_needs_two_letters():
    assert acronym_match("POD", "Proper Orthogonal Decomposition")
    assert not acronym_match("A", "Apple")


def test_relaxed_is_never_stricter_than_strict():
    scores = score_record("The SHAMS benchmark has 1,300 utterances.", ["1,300"], 0.8)
    assert scores == {"strict": 1, "relaxed": 1, "abstained": False}


def test_null_question_is_scored_on_abstaining():
    assert is_null_question(["Insufficient information."])
    assert score_record("Insufficient evidence", ["Insufficient information."], 0.8)["relaxed"] == 1
    assert score_record("It was Google.", ["Insufficient information."], 0.8)["relaxed"] == 0


def test_abstaining_on_an_answerable_question_is_wrong():
    scores = score_record("Insufficient evidence", ["Paris"], 0.8)
    assert scores == {"strict": 0, "relaxed": 0, "abstained": True}


# -- benchmark data ----------------------------------------------------------------------


def hotpot_row(row_id, question, answer, titles, kind="bridge"):
    return {
        "id": row_id, "question": question, "answer": answer, "type": kind,
        "supporting_facts": {"title": titles[:1], "sent_id": [0]},
        "context": {"title": titles, "sentences": [[f"{title} sentence one. ", "Two."] for title in titles]},
    }


def test_hotpotqa_records_dedupe_paragraphs_across_questions():
    rows = [hot for hot in (hotpot_row("a", "Q1?", "A1", ["X", "Y"]), hotpot_row("b", "Q2?", "A2", ["Y", "Z"]))]
    questions, chunks = benchmarks.hotpotqa_records(rows)
    assert [q["query_id"] for q in questions] == ["hotpotqa:a", "hotpotqa:b"]
    assert len(chunks) == 3
    assert questions[0]["gold_titles"] == ["X"]
    assert chunks[0]["text"] == "X sentence one. Two."


def test_multihoprag_records_chunk_articles_and_sample_questions():
    articles = [{"title": "T", "body": " ".join(f"w{i}" for i in range(50)), "source": "S", "url": "u"}]
    queries = [
        {"query": f"q{i}", "answer": "a", "question_type": "null_query" if i == 0 else "comparison_query",
         "evidence_list": [{"title": "T"}]}
        for i in range(10)
    ]
    questions, chunks = benchmarks.multihoprag_records(queries, articles, 5, 13, WhitespaceTokenizer(), 20, 5)
    assert len(questions) == 5 and len(chunks) == 3
    assert chunks[0]["chunk_id"] == "multihoprag:0:0"
    assert all(q["gold_titles"] == ["T"] for q in questions)
    # Same seed, same sample.
    again, _ = benchmarks.multihoprag_records(queries, articles, 5, 13, WhitespaceTokenizer(), 20, 5)
    assert [q["query_id"] for q in again] == [q["query_id"] for q in questions]


def test_label_pool_excludes_eval_null_and_yes_no(monkeypatch, tmp_path):
    questions = [
        {"query_id": f"d:{i}", "query": f"q{i}", "references": [answer], "question_type": kind}
        for i, (answer, kind) in enumerate([
            ("eval one", "x"), ("eval two", "x"), ("Yes", "x"), ("no", "x"),
            ("Insufficient information.", "null_query"), ("Real answer", "x"),
        ])
    ]
    path = tmp_path / "q.jsonl"
    benchmarks.write_jsonl(path, questions)
    monkeypatch.setattr(benchmarks, "benchmark_paths", lambda name: (path, path))
    config = {"benchmarks": {"eval_questions": 2}}
    pool = benchmarks.label_pool("demo", config)
    assert [q["query_id"] for q in pool] == ["d:5"]
    assert pool[0]["source"] == "demo"
    assert [q["query_id"] for q in benchmarks.eval_questions("demo", config)] == ["d:0", "d:1"]


def test_load_labelled_rows_merges_extra_csvs_and_dedupes(tmp_path):
    header = "query,label,closed_book_score,rag_score,source\n"
    main = tmp_path / "main.csv"
    extra = tmp_path / "extra.csv"
    main.write_text(header + "What is A?,retrieve,0,1,arxiv_chunk\n", encoding="utf-8")
    extra.write_text(header + "what is a?,skip,1,1,hotpotqa\nWhat is B?,skip,1,1,hotpotqa\n", encoding="utf-8")
    rows = load_labelled_rows(main, [extra, tmp_path / "missing.csv"])
    assert [row["query"] for row in rows] == ["What is A?", "What is B?"]
    assert rows[1]["source"] == "hotpotqa"


# -- harness -----------------------------------------------------------------------------


def test_gold_recall():
    contexts = [{"title": "X"}, {"title": "Y"}]
    assert gold_recall(contexts, ["X", "Z"]) == 0.5
    assert gold_recall(contexts, []) is None


def test_trace_fields_split_model_wait_and_other_time():
    trace = [
        {"step": "classify", "seconds": 0.1},
        {"step": "reformulate", "model": "flash_lite_31", "latency_seconds": 1.0, "wait_seconds": 0.5, "tokens": 100},
        {"step": "retrieve", "seconds": 0.2},
        {"step": "generate", "model": "ollama_qwen", "latency_seconds": 2.0, "wait_seconds": 0.0, "tokens": 300},
    ]
    fields = trace_fields(trace)
    assert fields["model_seconds"] == 3.0 and fields["wait_seconds"] == 0.5
    assert abs(fields["other_seconds"] - 0.3) < 1e-9
    assert fields["tokens_local"] == 300 and fields["tokens_gemini"] == 100 and fields["lite_calls"] == 1


def test_verification_fields():
    report = {"status": "verified", "claims": [{"label": "supported"}, {"label": "contradicted"}, {"label": "unsupported"}]}
    assert verification_fields(report) == {"n_claims": 3, "n_supported": 1, "n_contradicted": 1}
    assert verification_fields(None)["n_claims"] == 0


def make_record(query_id, answer, references, decision="retrieve", contexts=(), question_type="t"):
    question = {"query_id": query_id, "query": "q", "references": references, "question_type": question_type,
                "gold_titles": ["X"]}
    output = {
        "answer": answer, "decision": decision, "contexts": list(contexts), "model_seconds": 1.0, "wait_seconds": 0.0,
        "other_seconds": 0.1, "tokens_local": 10, "tokens_gemini": 0, "lite_calls": 0,
    }
    return build_record("vanilla", "demo", question, output, 0.8)


def test_build_record_has_no_gold_recall_without_contexts():
    assert make_record("a", "Paris", ["Paris"], decision="skip")["gold_recall"] is None
    assert make_record("a", "Paris", ["Paris"], contexts=[{"chunk_id": "c", "title": "X"}])["gold_recall"] == 1.0


def test_summarize_separates_null_questions_and_never_mixes_types():
    records = [
        make_record("a", "Paris", ["Paris"], question_type="one"),
        make_record("b", "Berlin", ["Paris"], question_type="one"),
        make_record("c", "Insufficient evidence", ["Insufficient information."], question_type="null_query"),
    ]
    summary = summarize(records)
    assert summary["n"] == 3 and summary["n_null_questions"] == 1
    assert abs(summary["relaxed_accuracy"] - 2 / 3) < 1e-9
    assert summary["answerable_relaxed_accuracy"] == 0.5
    assert summary["null_abstention_rate"] == 1.0
    assert summary["by_question_type"]["one"]["n"] == 2


# -- RAGAS adapter -----------------------------------------------------------------------


def test_ragas_judge_uses_call_llm_and_remembers_daily_limit(monkeypatch):
    from types import SimpleNamespace

    from src.eval import ragas_eval
    from src.llm import DailyLimitReached

    prompts = []

    def fake_call(prompt, role=None, **kwargs):
        prompts.append((prompt, role))
        return SimpleNamespace(text="reply")

    monkeypatch.setattr(ragas_eval, "call_llm", fake_call)
    judge = ragas_eval.CallLLMRagas()
    result = judge.generate_text(SimpleNamespace(to_string=lambda: "p"), n=2)
    assert [generation.text for generation in result.generations[0]] == ["reply", "reply"]
    assert prompts == [("p", "ragas_judge")] * 2

    def limit(*args, **kwargs):
        raise DailyLimitReached("done for today")

    monkeypatch.setattr(ragas_eval, "call_llm", limit)
    try:
        judge.generate_text(SimpleNamespace(to_string=lambda: "p"))
    except DailyLimitReached:
        pass
    assert isinstance(judge.limit_error, DailyLimitReached)


def test_ragas_only_scores_answers_with_contexts_that_did_not_abstain():
    from src.eval.ragas_eval import scorable

    assert scorable({"retrieved_chunk_ids": ["a"], "abstained": False})
    assert not scorable({"retrieved_chunk_ids": [], "abstained": False})
    assert not scorable({"retrieved_chunk_ids": ["a"], "abstained": True})


def test_abstained_covers_cannot_be_found():
    assert abstained("Cannot be found in the context.")
    assert abstained("The answer could not be found.")


def test_yes_no_span_is_not_proof_of_support():
    from src.generation.verifier import verify_answer as verify
    from tests.test_generation import SETTINGS, FakeScorer, chunk

    report = verify("1) Is it so? -> Yes", [chunk("a", "Yes, it is mentioned here.")], FakeScorer(), SETTINGS)
    assert report["claims"][0]["method"] == "nli"


def test_refusal_text_is_not_a_claim():
    from src.generation.verifier import split_claims

    assert split_claims("Cannot be found in the context.", question="Q?") == []
    assert split_claims("Cannot be found in the context.") == []


def test_not_specified_is_an_abstention_and_yes_no_is_not_a_claim():
    from src.generation.verifier import split_claims

    assert abstained("Not specified in the given context")
    assert abstained("The context does not specify which president.")
    assert split_claims("Yes.", question="Do the two articles agree?") == []
    assert split_claims("No", question="Do the two articles agree?") == []


def test_local_ragas_embeddings_wrap_the_project_embedder():
    import numpy as np

    from src.eval.ragas_eval import LocalRagasEmbeddings

    class FakeEmbedder:
        def encode(self, texts):
            return np.array([[float(len(text)), 1.0] for text in texts])

    embeddings = LocalRagasEmbeddings(FakeEmbedder())
    assert embeddings.embed_query("abc") == [3.0, 1.0]
    assert embeddings.embed_documents(["a", "bb"]) == [[1.0, 1.0], [2.0, 1.0]]
