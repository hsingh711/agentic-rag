"""Tests for the chunker and the ingest helpers (no network needed)."""

import json
from datetime import date

import pytest

from src.ingest.arxiv_fetch import (
    build_category_query,
    clean_text,
    load_completed_windows,
    make_windows,
    save_completed_windows,
    strip_version,
)
from src.ingest.chunker import WhitespaceTokenizer, build_chunks_file, chunk_paper, chunk_text
from src.ingest.io_utils import read_jsonl

TOKENIZER = WhitespaceTokenizer()


def numbered_words(count):
    """Return 'w0 w1 w2 ...' with ``count`` words, so overlaps are easy to check."""
    return " ".join(f"w{i}" for i in range(count))


def test_short_text_gives_one_chunk():
    assert chunk_text("a b c", TOKENIZER, chunk_size=10, overlap=2) == ["a b c"]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ", TOKENIZER, chunk_size=10, overlap=2) == []


def test_chunks_have_requested_size_and_overlap():
    chunks = chunk_text(numbered_words(25), TOKENIZER, chunk_size=10, overlap=3)
    words = [chunk.split() for chunk in chunks]
    assert [len(w) for w in words] == [10, 10, 10, 4]
    # Consecutive chunks share exactly 3 words.
    assert words[0][-3:] == words[1][:3]
    assert words[1][-3:] == words[2][:3]


def test_every_word_is_covered_and_last_chunk_reaches_end():
    chunks = chunk_text(numbered_words(23), TOKENIZER, chunk_size=8, overlap=2)
    covered = set(word for chunk in chunks for word in chunk.split())
    assert covered == {f"w{i}" for i in range(23)}
    assert chunks[-1].split()[-1] == "w22"


def test_exact_fit_has_no_extra_chunk():
    chunks = chunk_text(numbered_words(10), TOKENIZER, chunk_size=10, overlap=3)
    assert len(chunks) == 1


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("a b c", TOKENIZER, chunk_size=5, overlap=5)
    with pytest.raises(ValueError):
        chunk_text("a b c", TOKENIZER, chunk_size=0, overlap=0)


def test_chunk_paper_fields_and_ids():
    paper = {"arxiv_id": "2401.00001", "title": "A Title", "abstract": numbered_words(15)}
    chunks = chunk_paper(paper, TOKENIZER, chunk_size=10, overlap=2, fulltext=numbered_words(5))
    assert [c["chunk_id"] for c in chunks] == [
        "2401.00001:abstract:0",
        "2401.00001:abstract:1",
        "2401.00001:fulltext:0",
    ]
    for chunk in chunks:
        assert chunk["arxiv_id"] == "2401.00001"
        assert chunk["title"] == "A Title"
        assert chunk["text"]


def test_chunk_paper_uses_a_smaller_window_for_fulltext():
    """Section 11: full text is cut at fulltext_chunk_size, abstracts keep chunk_size."""
    paper = {"arxiv_id": "2401.00001", "title": "A Title", "abstract": numbered_words(12)}
    chunks = chunk_paper(
        paper, TOKENIZER, chunk_size=20, overlap=2, fulltext=numbered_words(20),
        fulltext_chunk_size=5, fulltext_overlap=1,
    )
    abstract_chunks = [chunk for chunk in chunks if chunk["source"] == "abstract"]
    fulltext_chunks = [chunk for chunk in chunks if chunk["source"] == "fulltext"]
    assert len(abstract_chunks) == 1
    assert len(abstract_chunks[0]["text"].split()) == 12
    assert all(len(chunk["text"].split()) <= 5 for chunk in fulltext_chunks)
    assert len(fulltext_chunks) == 5   # 20 words, window 5, step 4


def test_chunk_paper_fulltext_window_defaults_to_chunk_size():
    paper = {"arxiv_id": "1", "title": "A", "abstract": "short"}
    chunks = chunk_paper(paper, TOKENIZER, chunk_size=10, overlap=0, fulltext=numbered_words(20))
    fulltext_chunks = [chunk for chunk in chunks if chunk["source"] == "fulltext"]
    assert len(fulltext_chunks) == 2
    assert all(len(chunk["text"].split()) == 10 for chunk in fulltext_chunks)


def write_papers(path, papers):
    with open(path, "w", encoding="utf-8") as papers_file:
        for paper in papers:
            papers_file.write(json.dumps(paper) + "\n")


def test_build_chunks_file_is_idempotent_and_dedupes(tmp_path):
    papers_path = tmp_path / "papers.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    paper_a = {"arxiv_id": "1", "title": "A", "abstract": numbered_words(15)}
    paper_b = {"arxiv_id": "2", "title": "B", "abstract": numbered_words(5)}
    write_papers(papers_path, [paper_a, paper_b, paper_a])  # duplicate paper_a

    first = build_chunks_file(papers_path, chunks_path, TOKENIZER, 10, 2)
    second = build_chunks_file(papers_path, chunks_path, TOKENIZER, 10, 2)

    chunk_ids = [chunk["chunk_id"] for chunk in read_jsonl(chunks_path)]
    assert first == second == len(chunk_ids) == 3
    assert len(set(chunk_ids)) == 3


def test_build_chunks_file_uses_fulltext(tmp_path):
    papers_path = tmp_path / "papers.jsonl"
    fulltext_dir = tmp_path / "fulltext"
    fulltext_dir.mkdir()
    (fulltext_dir / "1.txt").write_text(numbered_words(30), encoding="utf-8")
    write_papers(papers_path, [{"arxiv_id": "1", "title": "A", "abstract": "short abstract"}])

    build_chunks_file(papers_path, tmp_path / "chunks.jsonl", TOKENIZER, 10, 2, fulltext_dir=fulltext_dir)

    sources = [chunk["source"] for chunk in read_jsonl(tmp_path / "chunks.jsonl")]
    assert sources[0] == "abstract"
    assert sources.count("fulltext") >= 3


def test_build_chunks_file_passes_the_fulltext_window(tmp_path):
    papers_path = tmp_path / "papers.jsonl"
    fulltext_dir = tmp_path / "fulltext"
    fulltext_dir.mkdir()
    (fulltext_dir / "1.txt").write_text(numbered_words(30), encoding="utf-8")
    write_papers(papers_path, [{"arxiv_id": "1", "title": "A", "abstract": numbered_words(20)}])

    build_chunks_file(
        papers_path, tmp_path / "chunks.jsonl", TOKENIZER, 20, 2, fulltext_dir=fulltext_dir,
        fulltext_chunk_size=5, fulltext_overlap=0,
    )

    chunks = list(read_jsonl(tmp_path / "chunks.jsonl"))
    abstract_chunks = [chunk for chunk in chunks if chunk["source"] == "abstract"]
    fulltext_chunks = [chunk for chunk in chunks if chunk["source"] == "fulltext"]
    assert len(abstract_chunks) == 1
    assert len(fulltext_chunks) == 6
    assert all(len(chunk["text"].split()) <= 5 for chunk in fulltext_chunks)


def test_read_jsonl_skips_truncated_last_line(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n{"a": 2, "b"', encoding="utf-8")
    assert list(read_jsonl(path)) == [{"a": 1}]


def test_text_helpers():
    assert clean_text("a \n  b\tc ") == "a b c"
    assert strip_version("2401.00001v3") == "2401.00001"
    assert strip_version("cs/0112017v1") == "cs/0112017"
    assert build_category_query(["cs.CL", "cs.IR"]) == "cat:cs.CL OR cat:cs.IR"


def test_make_windows_are_contiguous_newest_first():
    windows = make_windows(date(2026, 1, 1), window_days=7, today=date(2026, 1, 20))
    assert windows[0]["name"] == "20260114-20260120"
    assert windows[1]["name"] == "20260107-20260113"
    assert windows[-1]["name"] == "20260101-20260106"  # clipped at earliest_date
    assert len(windows) == 3


def test_window_state_roundtrip(tmp_path):
    state_path = tmp_path / "state.json"
    assert load_completed_windows(state_path) == set()
    save_completed_windows(state_path, {"b", "a"})
    assert load_completed_windows(state_path) == {"a", "b"}
