"""Section 10a: multi-hop benchmark data (HotpotQA distractor, MultiHop-RAG).

Each benchmark gets its own questions file and chunks file in ``benchmarks.dir`` and its own
Qdrant collection, so benchmark documents are never mixed with the arXiv corpus.

Question record: ``{query_id, query, references, question_type, gold_titles, dataset}``.
Chunk record: ``{chunk_id, title, text, source, dataset}`` (the same shape the arXiv chunks use).
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.config import PROJECT_ROOT, load_config
from src.ingest.chunker import chunk_text


def title_key(title: str) -> str:
    """Short stable id for a document title (used inside chunk_ids)."""
    return hashlib.md5(title.encode("utf-8")).hexdigest()[:10]


def hotpotqa_records(rows: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Turn HotpotQA distractor rows into question records and one chunk per unique paragraph.

    Args:
        rows: Dataset rows with ``id, question, answer, type, supporting_facts, context``.

    Returns:
        (questions, chunks). Paragraphs are short, so each paragraph is one chunk.
    """
    questions: List[Dict[str, Any]] = []
    chunks: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            chunk_id = f"hotpotqa:{title_key(title)}:0"
            if chunk_id not in chunks:
                chunks[chunk_id] = {
                    "chunk_id": chunk_id,
                    "title": title,
                    "text": " ".join(sentence.strip() for sentence in sentences),
                    "source": "hotpotqa",
                    "dataset": "hotpotqa",
                }
        questions.append({
            "query_id": f"hotpotqa:{row['id']}",
            "query": row["question"],
            "references": [row["answer"]],
            "question_type": row["type"],
            "gold_titles": sorted(set(row["supporting_facts"]["title"])),
            "dataset": "hotpotqa",
        })
    return questions, list(chunks.values())


def multihoprag_records(
    queries: List[Dict[str, Any]],
    articles: List[Dict[str, Any]],
    count: int,
    seed: int,
    tokenizer: Any,
    chunk_size: int,
    overlap: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Sample ``count`` MultiHop-RAG questions and chunk every article.

    Questions of type ``null_query`` have no answer in the corpus (the gold answer is an
    "insufficient information" statement), which is what the insufficient-evidence evaluation needs.
    """
    chunks: List[Dict[str, Any]] = []
    for number, article in enumerate(articles):
        for part, text in enumerate(chunk_text(article["body"], tokenizer, chunk_size, overlap)):
            chunks.append({
                "chunk_id": f"multihoprag:{number}:{part}",
                "title": article["title"],
                "text": text,
                "source": article.get("source", ""),
                "url": article.get("url", ""),
                "dataset": "multihoprag",
            })

    sampled = random.Random(seed).sample(range(len(queries)), min(count, len(queries)))
    questions = []
    for index in sorted(sampled):
        item = queries[index]
        questions.append({
            "query_id": f"multihoprag:{index}",
            "query": item["query"],
            "references": [item["answer"]],
            "question_type": item["question_type"],
            "gold_titles": sorted({evidence["title"] for evidence in item.get("evidence_list", [])}),
            "dataset": "multihoprag",
        })
    return questions, chunks


def benchmark_paths(name: str) -> Tuple[Path, Path]:
    """(questions file, chunks file) for a benchmark."""
    directory = PROJECT_ROOT / load_config()["benchmarks"]["dir"]
    return directory / f"questions_{name}.jsonl", directory / f"chunks_{name}.jsonl"


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write records as JSON lines, creating the folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_hotpotqa_rows(count: int) -> List[Dict[str, Any]]:
    """First ``count`` HotpotQA distractor validation rows (streamed from Hugging Face)."""
    from datasets import load_dataset

    stream = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    rows = []
    for row in stream:
        rows.append(row)
        if len(rows) >= count:
            break
    return rows


def load_multihoprag_files(settings: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Download (if needed) and read MultiHopRAG.json and corpus.json."""
    from huggingface_hub import hf_hub_download

    raw_dir = PROJECT_ROOT / settings["raw_dir"]
    loaded = []
    for name in ("MultiHopRAG.json", "corpus.json"):
        path = hf_hub_download(settings["repo"], name, repo_type="dataset", local_dir=str(raw_dir))
        with open(path, encoding="utf-8") as handle:
            loaded.append(json.load(handle))
    return loaded[0], loaded[1]


def eval_questions(name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The evaluation set of a benchmark: its first ``benchmarks.eval_questions`` questions."""
    from src.ingest.io_utils import read_jsonl

    questions = list(read_jsonl(benchmark_paths(name)[0]))
    return questions[: config["benchmarks"]["eval_questions"]]


def label_pool(name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Questions of a benchmark that may be used for classifier labels (never the evaluation set).

    Left out: ``null_query`` questions (their gold answer is "Insufficient information", which
    the closed-book vs RAG experiment cannot score) and yes/no questions (a closed-book guess is
    right half the time, which would label retrieval-dependent questions "skip" by luck).
    """
    from src.ingest.io_utils import read_jsonl

    questions = list(read_jsonl(benchmark_paths(name)[0]))[config["benchmarks"]["eval_questions"] :]
    return [
        dict(question, source=name)
        for question in questions
        if question["question_type"] != "null_query" and question["references"][0].strip().lower() not in ("yes", "no")
    ]
