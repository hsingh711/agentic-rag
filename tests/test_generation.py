"""Tests for grounded generation and answer verification (section 9). No models, no network."""

from types import SimpleNamespace

from src.generation import grounded
from src.generation.grounded import clean_citations, grounded_generate, is_insufficient
from src.generation.verifier import best_windows, classify_claim, sentence_windows, split_claims, verify_answer
from tests.test_graph import FakeStore, build, fake_generate, hit, run_pipeline, steps

SETTINGS = {
    "max_claims": 8,
    "supported_threshold": 0.7,
    "contradicted_threshold": 0.7,
    "use_llm_fallback": False,
}


def fake_llm_result(text):
    return SimpleNamespace(text=text, model_key="fake", latency_seconds=0.1, wait_seconds=0.0, total_tokens=10, cached=False)


class FakeScorer:
    """Entailment 0.9 when the claim text appears in the premise, otherwise neutral; 'NOT ' claims contradict."""

    def __init__(self):
        self.pairs = []

    def score_pairs(self, pairs):
        self.pairs.extend(pairs)
        scores = []
        for premise, claim in pairs:
            if claim.startswith("NOT "):
                scores.append({"entailment": 0.05, "neutral": 0.05, "contradiction": 0.9})
            elif claim.lower().rstrip(".") in premise.lower():
                scores.append({"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05})
            else:
                scores.append({"entailment": 0.1, "neutral": 0.8, "contradiction": 0.1})
        return scores


def chunk(chunk_id, text):
    return {"chunk_id": chunk_id, "title": chunk_id, "text": text}


# -- citations -------------------------------------------------------------------------


def test_clean_citations_keeps_valid_and_drops_invalid():
    answer, valid, invalid = clean_citations("It is X [a:0]. Also Y [7] and Z [b:1, fake].", ["a:0", "b:1"])
    assert valid == ["a:0", "b:1"]
    assert invalid == ["7", "fake"]
    assert answer == "It is X [a:0]. Also Y and Z [b:1]."


def test_is_insufficient():
    assert is_insufficient("Insufficient evidence.")
    assert not is_insufficient("The answer is 42.")


def test_grounded_generate_cites_and_flags(monkeypatch):
    monkeypatch.setattr(grounded, "call_llm", lambda *a, **k: fake_llm_result("SHAMS has 1,300 utterances [a:0][zzz]."))
    out = grounded_generate("q", [chunk("a:0", "t")])
    assert out["citations"] == ["a:0"]
    assert out["invalid_citations"] == ["zzz"]
    assert out["grounded"] and not out["uncited"] and not out["insufficient"]


def test_grounded_generate_flags_uncited_answer(monkeypatch):
    monkeypatch.setattr(grounded, "call_llm", lambda *a, **k: fake_llm_result("It is 42."))
    assert grounded_generate("q", [chunk("a:0", "t")])["uncited"]


def test_grounded_generate_normalises_refusal(monkeypatch):
    monkeypatch.setattr(grounded, "call_llm", lambda *a, **k: fake_llm_result("insufficient evidence to say [a:0]"))
    out = grounded_generate("q", [chunk("a:0", "t")])
    assert out["insufficient"] and out["citations"] == []
    assert out["answer"] == grounded.insufficient_text()


def test_grounded_generate_without_contexts_is_closed_book(monkeypatch):
    prompts = []

    def fake_call(prompt, system=None, role=None, **kwargs):
        prompts.append(system)
        return fake_llm_result("Paris")

    monkeypatch.setattr(grounded, "call_llm", fake_call)
    out = grounded_generate("capital of France?", [])
    assert out["answer"] == "Paris" and not out["grounded"] and not out["uncited"]
    assert "ONLY the passages" not in prompts[0]


# -- claim splitting -------------------------------------------------------------------


def test_split_claims_sentences_and_citations():
    claims = split_claims("SHAMS has 1,300 utterances [a:0]. It was built in 2026 [b:1].")
    assert [item["claim"] for item in claims] == ["SHAMS has 1,300 utterances.", "It was built in 2026."]


def test_split_claims_per_part_lines_and_refusals():
    answer = "1) Which method? -> POD [a:0]\n2) Which benchmark? -> not found in the retrieved papers"
    assert split_claims(answer) == [{"claim": "Which method? POD", "span": "POD"}]


def test_split_claims_insufficient_and_cap():
    assert split_claims("Insufficient evidence") == []
    assert len(split_claims("One. Two. Three. Four.", max_claims=2)) == 2


def test_sentence_windows_and_best_windows():
    assert sentence_windows("One. Two. Three.", 2) == ["One. Two.", "Two. Three."]
    assert sentence_windows("Only one.", 2) == ["Only one."]
    contexts = [chunk("a", "Cats sleep. Dogs bark loudly. Birds fly."), chunk("b", "Fish swim.")]
    found = best_windows("dogs bark", contexts, 2, 1)
    assert found == [("a", "Cats sleep. Dogs bark loudly.")] or found[0][0] == "a"


def test_extractive_part_answer_is_supported_by_string_match():
    contexts = [chunk("a:0", "The method is Proper Orthogonal Decomposition (POD).")]
    report = verify_answer("1) Which method? -> POD [a:0]", contexts, FakeScorer(), SETTINGS)
    assert report["claims"][0]["label"] == "supported"
    assert report["claims"][0]["method"] == "string"


# -- verification ----------------------------------------------------------------------


def test_classify_claim_thresholds():
    assert classify_claim(0.8, 0.0, 0.7, 0.7) == "supported"
    assert classify_claim(0.2, 0.9, 0.7, 0.7) == "contradicted"
    assert classify_claim(0.2, 0.2, 0.7, 0.7) == "unsupported"


def test_verify_answer_labels_each_claim():
    contexts = [chunk("a:0", "SHAMS has 1,300 utterances."), chunk("b:1", "Other text.")]
    answer = "SHAMS has 1,300 utterances [a:0]. The sky is green. NOT true."
    report = verify_answer(answer, contexts, FakeScorer(), SETTINGS)
    labels = [item["label"] for item in report["claims"]]
    assert labels == ["supported", "unsupported", "contradicted"]
    assert report["claims"][0]["chunk_id"] == "a:0"
    assert report["n_supported"] == 1 and abs(report["supported_ratio"] - 1 / 3) < 1e-9


def test_verify_answer_scores_only_the_best_windows_per_claim():
    scorer = FakeScorer()
    contexts = [chunk("a", "One. Two. Three. Four."), chunk("b", "Five. Six.")]
    verify_answer("One fact. Two fact.", contexts, scorer, dict(SETTINGS, windows_per_claim=2))
    assert len(scorer.pairs) == 4


def test_verify_answer_statuses():
    scorer = FakeScorer()
    assert verify_answer("A claim.", [], scorer, SETTINGS)["status"] == "no_contexts"
    assert verify_answer("Insufficient evidence", [chunk("a", "x")], scorer, SETTINGS)["status"] == "no_claims"
    assert scorer.pairs == []


def test_llm_fallback_only_for_unsupported_claims_and_survives_failure():
    contexts = [chunk("a:0", "SHAMS has 1,300 utterances.")]
    asked = []

    def fallback(claim, ctx):
        asked.append(claim)
        return "supported"

    settings = dict(SETTINGS, use_llm_fallback=True)
    report = verify_answer("SHAMS has 1,300 utterances. Paraphrased fact.", contexts, FakeScorer(), settings, fallback)
    assert asked == ["Paraphrased fact."]
    assert [item["method"] for item in report["claims"]] == ["nli", "llm"]
    assert report["n_supported"] == 2

    report = verify_answer("Paraphrased fact.", contexts, FakeScorer(), settings, lambda claim, ctx: None)
    assert report["claims"][0]["label"] == "unsupported" and report["claims"][0]["method"] == "nli"


# -- graph integration -----------------------------------------------------------------


def test_graph_verify_step_checks_the_chunks_used_for_the_answer():
    seen = {}

    def fake_verify(answer, contexts, question=None):
        seen["contexts"] = [c["chunk_id"] for c in contexts]
        return {"status": "verified", "claims": [], "n_claims": 0, "n_supported": 0, "supported_ratio": None}

    store = FakeStore(default_hits=[hit("a:0"), hit("b:1")])
    state = run_pipeline(build(store, verify_fn=fake_verify), "q")
    assert steps(state)[-2:] == ["generate", "verify"]
    assert seen["contexts"] == ["a:0", "b:1"]
    assert state["verification"]["status"] == "verified"


def test_graph_without_verify_fn_has_no_verify_step():
    state = run_pipeline(build(FakeStore(default_hits=[hit("a:0")])), "q")
    assert "verify" not in steps(state)


def test_refuse_when_not_sufficient_skips_the_generator():
    from tests.test_graph import judge_always

    called = []

    def generator(query, contexts):
        called.append(query)
        return {"answer": "guess"}

    store = FakeStore(default_hits=[hit("a:0")])
    judge_fn, _ = judge_always("not_sufficient")
    from src.graph.pipeline import build_graph
    from tests.test_graph import EARLY_STOP_OFF, fake_reformulate

    app = build_graph(
        store, classifier=lambda q: (True, 0.9), reformulate_fn=fake_reformulate, judge_fn=judge_fn,
        generate_fn=generator, max_retries=1, top_k=2, max_contexts=6, early_stop=EARLY_STOP_OFF,
        per_part_generation=False, per_part_contexts=2, refuse_when_not_sufficient=True,
    )
    state = run_pipeline(app, "q")
    assert called == []
    assert state["answer"] == grounded.insufficient_text()
    assert fake_generate("q", [])["answer"]  # helper still importable


def test_bare_short_answer_is_checked_as_question_plus_answer():
    assert split_claims("1,300", question="How many utterances?") == [{"claim": "How many utterances? 1,300", "span": "1,300"}]
    assert split_claims("1,300") == [{"claim": "1,300", "span": None}]
    assert split_claims("Insufficient evidence", question="Q?") == []
    long_answer = "SHAMS has 1,300 utterances in total across all of its splits."
    assert split_claims(long_answer, question="Q?")[0]["span"] is None


def test_bare_short_answer_found_in_context_is_supported_by_string_match():
    contexts = [chunk("a:0", "The benchmark has 1,300 utterances.")]
    report = verify_answer("1,300", contexts, FakeScorer(), SETTINGS, question="How many utterances?")
    assert report["claims"][0]["label"] == "supported" and report["claims"][0]["method"] == "string"


def test_citing_chunks_short_answer_and_long_answer():
    from src.generation.grounded import citing_chunks

    contexts = [chunk("a", "The benchmark has 1,300 utterances."), chunk("b", "Unrelated text about cats.")]
    assert citing_chunks("1,300", contexts) == ["a"]
    assert citing_chunks("nothing like this", contexts) == []
    long_answer = "The benchmark has 1,300 utterances recorded from speakers across the whole region today"
    assert citing_chunks(long_answer, contexts) in (["a"], [])


def test_short_cited_generate_attaches_citations_and_keeps_answer_plain(monkeypatch):
    monkeypatch.setattr(grounded, "call_llm", lambda *a, **k: fake_llm_result("1,300 [zzz]"))
    out = grounded.short_cited_generate("How many?", [chunk("a:0", "It has 1,300 utterances."), chunk("b:1", "Other.")])
    assert out["answer"] == "1,300" and out["citations"] == ["a:0"] and not out["uncited"]
    assert grounded.format_with_citations(out["answer"], out["citations"]) == "1,300 [a:0]"


def test_short_cited_generate_refusal_and_uncited(monkeypatch):
    monkeypatch.setattr(grounded, "call_llm", lambda *a, **k: fake_llm_result("Insufficient evidence"))
    out = grounded.short_cited_generate("Q?", [chunk("a:0", "t")])
    assert out["insufficient"] and out["citations"] == [] and not out["uncited"]
    monkeypatch.setattr(grounded, "call_llm", lambda *a, **k: fake_llm_result("Paris"))
    assert grounded.short_cited_generate("Q?", [chunk("a:0", "t")])["uncited"]
