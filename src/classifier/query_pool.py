"""Build the pool of candidate queries (with reference answers) for the labelling experiment.

Sources:
  * ``arxiv_chunk`` - questions written by the ``label_generation`` LLM from arXiv chunks
                      (likely "retrieve": the fact is only in our corpus).
  * ``triviaqa``    - TriviaQA rc.nocontext validation questions (likely "skip").
  * ``nq``          - Natural Questions open validation questions (likely "skip").
  * ``arithmetic``  - generated sums/products with exact answers (likely "skip").

The label is NOT decided here; it comes from the experiment in make_labels.py.
Pool building is resumable: generated chunks are recorded in a state file.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Set

from pydantic import BaseModel

from src.agents.prompts import (
    CHUNK_QUESTION_SYSTEM,
    build_chunk_question_prompt,
    build_concept_question_prompt,
    build_plain_chunk_question_prompt,
)
from src.classifier.scoring import normalize_answer
from src.ingest.io_utils import append_jsonl, read_jsonl
from src.llm import DailyLimitReached, LLMJSONError, call_llm

logger = logging.getLogger(__name__)

MAX_ANSWER_WORDS = 8
MIN_QUESTION_WORDS = 5


class QuestionItem(BaseModel):
    """One generated question as returned by the LLM."""

    excerpt: int
    question: str
    answer: str


class QuestionBatch(BaseModel):
    """The JSON object the LLM must return."""

    items: List[QuestionItem]


# ---------------------------------------------------------------------------
# State file
# ---------------------------------------------------------------------------


def load_state(state_path: Path) -> Dict[str, Any]:
    """Read pool-building progress (which static sources and chunks are done)."""
    if not state_path.is_file():
        return {"static_done": False, "chunks_done": [], "failed_calls": 0}
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    """Write pool-building progress."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Static sources (no LLM)
# ---------------------------------------------------------------------------


def make_arithmetic_queries(count: int, seed: int) -> List[Dict[str, Any]]:
    """Generate simple arithmetic questions with exact answers."""
    rng = random.Random(seed)
    queries = []
    for index in range(count):
        kind = index % 3
        if kind == 0:
            left, right = rng.randint(10, 999), rng.randint(10, 999)
            question, answer = f"What is {left} plus {right}?", left + right
        elif kind == 1:
            left, right = rng.randint(100, 999), rng.randint(10, 99)
            question, answer = f"What is {left} minus {right}?", left - right
        else:
            left, right = rng.randint(11, 99), rng.randint(2, 19)
            question, answer = f"What is {left} times {right}?", left * right
        queries.append(
            {"query_id": f"arith:{index}", "query": question, "references": [str(answer)], "source": "arithmetic"}
        )
    return queries


def load_triviaqa_queries(count: int) -> List[Dict[str, Any]]:
    """First ``count`` TriviaQA rc.nocontext validation questions (streamed, no full download)."""
    from datasets import load_dataset

    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True)
    queries = []
    for row in dataset:
        references = [row["answer"]["value"]] + list(row["answer"]["aliases"])
        unique_references = list(dict.fromkeys(reference for reference in references if reference))
        queries.append(
            {
                "query_id": f"triviaqa:{row['question_id']}",
                "query": row["question"],
                "references": unique_references,
                "source": "triviaqa",
            }
        )
        if len(queries) >= count:
            break
    return queries


def load_nq_queries(count: int) -> List[Dict[str, Any]]:
    """First ``count`` Natural Questions (open) validation questions (streamed)."""
    from datasets import load_dataset

    dataset = load_dataset("google-research-datasets/nq_open", split="validation", streaming=True)
    queries = []
    for index, row in enumerate(dataset):
        question = row["question"].strip()
        if not question.endswith("?"):
            question += "?"
        queries.append({"query_id": f"nq:{index}", "query": question, "references": list(row["answer"]), "source": "nq"})
        if len(queries) >= count:
            break
    return queries


# ---------------------------------------------------------------------------
# LLM-generated questions from chunks
# ---------------------------------------------------------------------------


def validate_generated_item(item: QuestionItem, chunk: Dict[str, Any]) -> bool:
    """Keep only questions whose answer is a short span really present in the chunk."""
    question = item.question.strip()
    answer = item.answer.strip()
    if not question or not answer:
        return False
    if len(question.split()) < MIN_QUESTION_WORDS or len(answer.split()) > MAX_ANSWER_WORDS:
        return False
    normalized_answer = normalize_answer(answer)
    if not normalized_answer:
        return False
    if f" {normalized_answer} " not in f" {normalize_answer(chunk['text'])} ":
        return False  # answer not copied from the excerpt -> likely hallucinated
    if f" {normalized_answer} " in f" {normalize_answer(question)} ":
        return False  # question gives the answer away
    return True


def queries_from_batch(
    batch: QuestionBatch,
    chunks: List[Dict[str, Any]],
    seen_questions: Set[str],
    source: str = "arxiv_chunk",
    id_prefix: str = "chunk",
) -> List[Dict[str, Any]]:
    """Turn a validated LLM reply into pool records (drops invalid and duplicate questions)."""
    queries = []
    per_chunk_counter: Dict[int, int] = {}
    for item in batch.items:
        if item.excerpt < 1 or item.excerpt > len(chunks):
            continue
        chunk = chunks[item.excerpt - 1]
        if not validate_generated_item(item, chunk):
            continue
        normalized_question = normalize_answer(item.question)
        if normalized_question in seen_questions:
            continue
        seen_questions.add(normalized_question)
        index = per_chunk_counter.get(item.excerpt, 0)
        per_chunk_counter[item.excerpt] = index + 1
        queries.append(
            {
                "query_id": f"{id_prefix}:{chunk['chunk_id']}:{index}",
                "query": item.question.strip(),
                "references": [item.answer.strip()],
                "source": source,
                "source_chunk_id": chunk["chunk_id"],
            }
        )
    return queries


def generate_chunk_questions(
    chunks: List[Dict[str, Any]],
    pool_path: Path,
    state: Dict[str, Any],
    state_path: Path,
    chunks_per_call: int,
    questions_per_chunk: int,
) -> int:
    """Ask the ``label_generation`` LLM for questions, a few chunks per call.

    Progress is saved after every call. ``DailyLimitReached`` is re-raised after saving,
    so the caller can stop and resume later.

    Returns:
        Number of new questions added to the pool.
    """
    done_ids = set(state["chunks_done"])
    remaining = [chunk for chunk in chunks if chunk["chunk_id"] not in done_ids]
    seen_questions = {normalize_answer(record["query"]) for record in read_jsonl(pool_path)}
    added = 0

    for start in range(0, len(remaining), chunks_per_call):
        group = remaining[start : start + chunks_per_call]
        prompt = build_chunk_question_prompt(group, questions_per_chunk)
        try:
            result = call_llm(prompt, system=CHUNK_QUESTION_SYSTEM, role="label_generation", response_model=QuestionBatch)
            new_queries = queries_from_batch(result.parsed, group, seen_questions)
            append_jsonl(pool_path, new_queries)
            added += len(new_queries)
        except LLMJSONError as error:
            logger.warning("Skipping %d chunks after invalid JSON twice: %s", len(group), error)
            state["failed_calls"] = state.get("failed_calls", 0) + 1
        except DailyLimitReached:
            save_state(state_path, state)
            raise

        state["chunks_done"].extend(chunk["chunk_id"] for chunk in group)
        save_state(state_path, state)
        logger.info("Chunk questions: %d / %d chunks done, %d questions added", len(state["chunks_done"]), len(chunks), added)
    return added


# ---------------------------------------------------------------------------
# Hard examples (break the "research jargon -> retrieve" shortcut)
# ---------------------------------------------------------------------------

MAX_CONCEPT_ANSWER_WORDS = 4


class ConceptItem(BaseModel):
    """One textbook question as returned by the LLM."""

    question: str
    answer: str


class ConceptBatch(BaseModel):
    """The JSON object the LLM must return for concept questions."""

    items: List[ConceptItem]


def validate_concept_item(item: ConceptItem) -> bool:
    """Keep questions with a short single-term answer that the question does not give away."""
    question = item.question.strip()
    answer = item.answer.strip()
    if len(question.split()) < MIN_QUESTION_WORDS:
        return False
    if not answer or len(answer.split()) > MAX_CONCEPT_ANSWER_WORDS:
        return False
    normalized_answer = normalize_answer(answer)
    if not normalized_answer:
        return False
    return f" {normalized_answer} " not in f" {normalize_answer(question)} "


def concept_queries_from_batch(batch: ConceptBatch, topic_index: int, seen_questions: Set[str]) -> List[Dict[str, Any]]:
    """Turn a validated concept-question reply into pool records."""
    queries = []
    for item in batch.items:
        if not validate_concept_item(item):
            continue
        normalized_question = normalize_answer(item.question)
        if normalized_question in seen_questions:
            continue
        seen_questions.add(normalized_question)
        queries.append(
            {
                "query_id": f"concept:{topic_index}:{len(queries)}",
                "query": item.question.strip(),
                "references": [item.answer.strip()],
                "source": "concept_question",
            }
        )
    return queries


def generate_concept_questions(config: Dict[str, Any], pool_path: Path, state: Dict[str, Any], state_path: Path) -> int:
    """One ``label_generation`` call per topic; progress saved after each topic."""
    hard_config = config["labels"]["hard_examples"]
    done_topics = set(state.setdefault("concept_topics_done", []))
    seen_questions = {normalize_answer(record["query"]) for record in read_jsonl(pool_path)}
    added = 0
    for topic_index, topic in enumerate(hard_config["concept_topics"]):
        if topic_index in done_topics:
            continue
        prompt = build_concept_question_prompt(topic, hard_config["questions_per_topic"])
        try:
            result = call_llm(prompt, system=CHUNK_QUESTION_SYSTEM, role="label_generation", response_model=ConceptBatch)
            new_queries = concept_queries_from_batch(result.parsed, topic_index, seen_questions)
            append_jsonl(pool_path, new_queries)
            added += len(new_queries)
        except LLMJSONError as error:
            logger.warning("Skipping topic %r after invalid JSON twice: %s", topic, error)
            state["failed_calls"] = state.get("failed_calls", 0) + 1
        except DailyLimitReached:
            save_state(state_path, state)
            raise
        state["concept_topics_done"].append(topic_index)
        save_state(state_path, state)
    logger.info("Concept questions added: %d", added)
    return added


def generate_plain_chunk_questions(
    config: Dict[str, Any], chunks: List[Dict[str, Any]], pool_path: Path, state: Dict[str, Any], state_path: Path
) -> int:
    """Plain-language questions for a fixed random sample of chunks; progress saved per call."""
    hard_config = config["labels"]["hard_examples"]
    rng = random.Random(config["labels"]["seed"])
    sample_size = min(hard_config["plain_chunk_count"], len(chunks))
    sample = rng.sample(chunks, sample_size)

    done_ids = set(state.setdefault("plain_chunks_done", []))
    remaining = [chunk for chunk in sample if chunk["chunk_id"] not in done_ids]
    seen_questions = {normalize_answer(record["query"]) for record in read_jsonl(pool_path)}
    per_call = hard_config["plain_chunks_per_call"]
    added = 0
    for start in range(0, len(remaining), per_call):
        group = remaining[start : start + per_call]
        try:
            result = call_llm(
                build_plain_chunk_question_prompt(group),
                system=CHUNK_QUESTION_SYSTEM,
                role="label_generation",
                response_model=QuestionBatch,
            )
            new_queries = queries_from_batch(result.parsed, group, seen_questions, source="plain_chunk", id_prefix="plain")
            append_jsonl(pool_path, new_queries)
            added += len(new_queries)
        except LLMJSONError as error:
            logger.warning("Skipping %d chunks after invalid JSON twice: %s", len(group), error)
            state["failed_calls"] = state.get("failed_calls", 0) + 1
        except DailyLimitReached:
            save_state(state_path, state)
            raise
        state["plain_chunks_done"].extend(chunk["chunk_id"] for chunk in group)
        save_state(state_path, state)
    logger.info("Plain-language chunk questions added: %d", added)
    return added


def build_hard_examples(config: Dict[str, Any], chunks: List[Dict[str, Any]], pool_path: Path, state_path: Path) -> int:
    """Add concept questions and plain-language chunk questions to the pool (resumable).

    Returns:
        Number of queries in the pool afterwards.
    """
    state = load_state(state_path)
    generate_concept_questions(config, pool_path, state, state_path)
    generate_plain_chunk_questions(config, chunks, pool_path, state, state_path)
    return sum(1 for _ in read_jsonl(pool_path))


def build_pool(config: Dict[str, Any], chunks: List[Dict[str, Any]], pool_path: Path, state_path: Path) -> int:
    """Create or complete the query pool. Safe to call repeatedly.

    Returns:
        Number of queries in the pool afterwards.
    """
    labels_config = config["labels"]
    state = load_state(state_path)

    if not state["static_done"]:
        static_queries = make_arithmetic_queries(labels_config["arithmetic_count"], labels_config["seed"])
        static_queries += load_triviaqa_queries(labels_config["triviaqa_count"])
        static_queries += load_nq_queries(labels_config["nq_count"])
        existing_ids = {record["query_id"] for record in read_jsonl(pool_path)}
        append_jsonl(pool_path, [query for query in static_queries if query["query_id"] not in existing_ids])
        state["static_done"] = True
        save_state(state_path, state)
        logger.info("Added %d static queries (arithmetic, TriviaQA, NQ).", len(static_queries))

    generate_chunk_questions(
        chunks,
        pool_path,
        state,
        state_path,
        labels_config["chunks_per_call"],
        labels_config["questions_per_chunk"],
    )
    return sum(1 for _ in read_jsonl(pool_path))
