"""Tests for hybrid retrieval and retrieval metrics (section 11). No models, no Qdrant."""

from src.eval.retrieval_eval import evaluate_retriever, gold_items, question_metrics
from src.index.hybrid import BM25Index, HybridStore, reciprocal_rank_fusion, tokenize


def chunk(chunk_id, text, title=""):
    return {"chunk_id": chunk_id, "title": title or chunk_id, "text": text}


CHUNKS = [
    chunk("a", "The SHAMS benchmark has 1,300 utterances of Levantine Arabic speech."),
    chunk("b", "Matryoshka representation learning nests embeddings of several sizes."),
    chunk("c", "A completely unrelated recipe for tomato soup."),
]


def test_tokenize_drops_stop_words_and_punctuation():
    assert tokenize("The SHAMS benchmark, of 1,300 utterances!") == ["shams", "benchmark", "1", "300", "utterances"]


def test_bm25_ranks_the_matching_chunk_first_and_skips_non_matches():
    index = BM25Index(CHUNKS)
    hits = index.search("how many utterances in SHAMS", 5)
    assert hits[0]["chunk_id"] == "a"
    assert "c" not in [hit["chunk_id"] for hit in hits]
    assert index.search("zzzz qqqq", 5) == []


def test_bm25_prefers_rarer_terms():
    chunks = [chunk("x", "common common common rare"), chunk("y", "common common common common")]
    assert BM25Index(chunks).search("common rare", 2)[0]["chunk_id"] == "x"


def test_reciprocal_rank_fusion_rewards_agreement_and_respects_weights():
    first = [chunk("a", ""), chunk("b", ""), chunk("c", "")]
    second = [chunk("b", ""), chunk("a", ""), chunk("d", "")]
    fused = [item["chunk_id"] for item in reciprocal_rank_fusion([first, second])]
    assert set(fused[:2]) == {"a", "b"} and set(fused) == {"a", "b", "c", "d"}
    heavy_second = [item["chunk_id"] for item in reciprocal_rank_fusion([first, second], weights=[1.0, 5.0])]
    assert heavy_second[0] == "b"


class FakeDense:
    def __init__(self, ranking):
        self.ranking = ranking
        self.asked = []

    def search(self, query, top_k):
        self.asked.append(top_k)
        return [dict(item) for item in self.ranking][:top_k]


class FakeReranker:
    def rerank(self, query, chunks, top_k):
        return sorted(chunks, key=lambda item: item["chunk_id"], reverse=True)[:top_k]


def test_hybrid_store_fuses_dense_and_bm25_with_candidate_limit():
    dense = FakeDense([CHUNKS[1], CHUNKS[2], CHUNKS[0]])
    store = HybridStore(dense, BM25Index(CHUNKS), candidates=3)
    hits = store.search("SHAMS utterances", 2)
    assert len(hits) == 2 and dense.asked == [3]
    assert "a" in [hit["chunk_id"] for hit in hits]


def test_hybrid_store_applies_the_reranker_to_the_fused_head():
    dense = FakeDense(CHUNKS)
    store = HybridStore(dense, BM25Index(CHUNKS), candidates=3, reranker=FakeReranker(), rerank_candidates=3)
    assert [hit["chunk_id"] for hit in store.search("SHAMS", 2)] == ["c", "b"]


def test_question_metrics_for_titles_and_chunks():
    hits = [{"chunk_id": "c1", "title": "X"}, {"chunk_id": "c2", "title": "Y"}]
    assert question_metrics(hits, ["X", "Z"], "hotpotqa") == {"hit": 1.0, "gold_recall": 0.5, "all_gold": 0.0}
    assert question_metrics(hits, ["c2"], "arxiv") == {"hit": 1.0, "gold_recall": 1.0, "all_gold": 1.0}
    assert question_metrics(hits, ["zz"], "arxiv")["hit"] == 0.0


def test_gold_items_and_evaluate_retriever():
    assert gold_items({"gold_titles": ["X"]}, "hotpotqa") == ["X"]
    assert gold_items({"source_chunk_id": "c1"}, "arxiv") == ["c1"]
    assert gold_items({}, "arxiv") == []
    questions = [{"query": "q1", "gold_titles": ["X"]}, {"query": "q2", "gold_titles": ["Q"]}]
    result = evaluate_retriever(lambda query, k: [{"chunk_id": "c", "title": "X"}], questions, "hotpotqa", 5)
    assert result["n"] == 2 and result["hit"] == 0.5 and result["all_gold"] == 0.5
