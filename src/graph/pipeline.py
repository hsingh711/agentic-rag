"""Blocks 1-4 wired together as a LangGraph state machine with a bounded self-critique loop.

    classify --skip--> generate
    classify --retrieve--> reformulate -> retrieve -> critique
    critique --not_sufficient and retries < max_retries--> reformulate (retry with the critic's feedback)
    critique --otherwise--> generate

Early stop (``graph.early_stop``, retry rounds only), two separate switches:
- ``no_new_chunks``: the retry found no chunk that was not already seen, so another critique
  of the same evidence cannot change anything (on by default).
- ``same_missing``: the critic names the same gap as in the previous round. Saves quota on
  questions the corpus cannot answer, but for single-fact questions the gap is usually the
  question restated, so this effectively lowers max_retries to 1 and lost 1 of 20 correct
  answers in the section 8 follow-up comparison (off by default; a quota-saving option).

Multi-part questions (first reformulation was ``decompose``): with ``graph.per_part_generation``
each sub-question is answered separately from its own chunks and the answers are joined
without another LLM call. The 3B generator answers single facts well but mixes up several
facts in one call (section 8 demo Q4).

The skip path never touches the vector store. Chunks are deduplicated across retries, and
every node appends one entry to ``state["trace"]``. The generate node uses grounded generation
(block 5: short answer with chunk_id citations attached from the chunks that contain it, refuses
with "Insufficient evidence"; closed-book on the skip path).
When a ``verify_fn`` is given, a verify node (block 6) checks the answer's claims against the
chunks it was generated from, after generate.

All components are passed in, so tests can replace each of them with a fake.
"""

import operator
import re
import time
from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.prompts import (
    CLOSED_BOOK_SYSTEM,
    PART_ANSWER_SYSTEM,
    PART_NOT_FOUND,
    VANILLA_RAG_SYSTEM,
    build_closed_book_prompt,
    build_part_answer_prompt,
    build_vanilla_rag_prompt,
)
from src.classifier.scoring import contains_answer
from src.config import load_config
from src.generation.grounded import insufficient_text, short_cited_generate
from src.llm import call_llm


class PipelineState(TypedDict, total=False):
    """Everything the graph knows about one question."""

    query: str
    retrieve: bool                      # classifier decision
    sub_queries: List[str]              # queries of the latest reformulation
    all_queries: List[str]              # every query searched so far (retry avoids repeats)
    contexts: List[Dict[str, Any]]      # unique chunks retrieved so far, with rank and round
    verdict: str                        # latest critic verdict
    missing: str                        # latest critic "missing" text
    retries: int                        # retry rounds done (0 on the first pass)
    first_mode: str                     # mode of the first reformulation (rewrite/decompose/...)
    parts: List[str]                    # sub-questions of a first-pass decomposition
    previous_missing: str               # critic "missing" text of the round before
    last_new_chunks: int                # chunks the latest retrieval added
    stop_reason: Optional[str]          # why the loop ended (None while it continues)
    answer: str
    answer_contexts: List[Dict[str, Any]]   # chunks the answer was generated from (for verification)
    verification: Dict[str, Any]
    trace: Annotated[List[Dict[str, Any]], operator.add]   # nodes append, never overwrite


def missing_similarity(first: str, second: str) -> float:
    """Word-set overlap (Jaccard) of two critic "missing" texts, case-insensitive; 0.0 if either is empty."""
    first_words = set(re.findall(r"\w+", first.lower()))
    second_words = set(re.findall(r"\w+", second.lower()))
    if not first_words or not second_words:
        return 0.0
    return len(first_words & second_words) / len(first_words | second_words)


def decide_stop(
    verdict: str,
    retries: int,
    max_retries: int,
    missing: str,
    previous_missing: str,
    last_new_chunks: int,
    early_stop: Dict[str, Any],
) -> Optional[str]:
    """Return why the critique loop should end now, or None to retry.

    Reasons: ``sufficient``, ``max_retries``, and (early stop, retry rounds only)
    ``no_new_chunks`` or ``same_missing``.
    """
    if verdict == "sufficient":
        return "sufficient"
    if retries >= max_retries:
        return "max_retries"
    if retries > 0:
        if early_stop.get("no_new_chunks") and last_new_chunks == 0:
            return "no_new_chunks"
        if early_stop.get("same_missing") and missing_similarity(missing, previous_missing) >= early_stop["missing_similarity"]:
            return "same_missing"
    return None


def select_contexts(contexts: List[Dict[str, Any]], max_contexts: int) -> List[Dict[str, Any]]:
    """Choose the chunks shown to the critic and the generator.

    Ordered by rank within their own query first (every sub-query's best hit comes before any
    second-best hit), newer retrieval rounds first on ties. This keeps evidence for every
    sub-question and for the latest retry, instead of letting one query fill the budget.
    """
    ordered = sorted(contexts, key=lambda chunk: (chunk["rank"], -chunk["round"]))
    return ordered[:max_contexts]


def merge_contexts(
    existing: List[Dict[str, Any]],
    hits_per_query: List[List[Dict[str, Any]]],
    round_number: int,
    queries: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Add new hits to ``existing``, skipping chunk_ids already present.

    Each chunk keeps its best rank, the round it was first found in, and (when ``queries`` is
    given) every query that retrieved it in ``queries``, so per-part generation can pick the
    chunks of one sub-question.

    Returns:
        The merged list and the number of new chunks added.
    """
    merged = [dict(chunk) for chunk in existing]
    by_id = {chunk["chunk_id"]: chunk for chunk in merged}
    added = 0
    for query_number, query_hits in enumerate(hits_per_query):
        query = queries[query_number] if queries else None
        for rank, hit in enumerate(query_hits, start=1):
            known = by_id.get(hit["chunk_id"])
            if known is not None:
                known["rank"] = min(known["rank"], rank)
                if query is not None and query not in known.get("queries", []):
                    known["queries"] = known.get("queries", []) + [query]
                continue
            chunk = dict(hit, rank=rank, round=round_number, queries=[query] if query is not None else [])
            merged.append(chunk)
            by_id[chunk["chunk_id"]] = chunk
            added += 1
    return merged, added


def contexts_for_part(contexts: List[Dict[str, Any]], part: str, per_part_contexts: int) -> List[Dict[str, Any]]:
    """Chunks for one sub-question: its own best hits, plus the best retry hit if there is one.

    Retry queries are not tied to one sub-question, so one slot goes to the best retry chunk
    (it may hold the fact the critic said was missing); otherwise the part's own hits fill it.
    """
    own = select_contexts([chunk for chunk in contexts if part in chunk.get("queries", [])], per_part_contexts)
    from_retries = select_contexts(
        [chunk for chunk in contexts if chunk["round"] > 0 and part not in chunk.get("queries", [])], 1
    )
    if from_retries and len(own) >= per_part_contexts:
        own = own[: per_part_contexts - 1]
    return (own + from_retries)[:per_part_contexts]


def answer_part(question: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Answer one sub-question with a short phrase from its chunks (generator role)."""
    result = call_llm(build_part_answer_prompt(question, contexts), system=PART_ANSWER_SYSTEM, role="generator")
    return {
        "answer": result.text.strip(),
        "model": result.model_key,
        "latency_seconds": result.latency_seconds,
        "wait_seconds": result.wait_seconds,
        "tokens": result.total_tokens,
        "cached": result.cached,
    }


def is_not_found(answer: str) -> bool:
    """True if a part answer says the context does not contain it."""
    return PART_NOT_FOUND in answer.lower() or not answer.strip()


def compose_answer(parts: List[Dict[str, Any]]) -> str:
    """Join part answers into one numbered answer, citing the chunks that contain each answer."""
    lines = []
    for number, part in enumerate(parts, start=1):
        if is_not_found(part["answer"]):
            lines.append(f"{number}) {part['question']} -> not found in the retrieved papers")
            continue
        # A part answer can come back on several lines ("KOptim\nFIWARE"); keep one line per part.
        answer_lines = [line.strip() for line in part["answer"].splitlines() if line.strip()]
        answer = ", ".join(answer_lines)
        cited = []
        for chunk in part["contexts"]:
            if any(contains_answer(chunk["text"], line) for line in answer_lines):
                cited.append(chunk["chunk_id"])
        citation = " " + " ".join(f"[{chunk_id}]" for chunk_id in cited) if cited else ""
        lines.append(f"{number}) {part['question']} -> {answer}{citation}")
    return "\n".join(lines)


def baseline_generate(query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Placeholder generator until section 9: closed-book without contexts, vanilla RAG prompt with them."""
    if contexts:
        result = call_llm(build_vanilla_rag_prompt(query, contexts), system=VANILLA_RAG_SYSTEM, role="generator")
    else:
        result = call_llm(build_closed_book_prompt(query), system=CLOSED_BOOK_SYSTEM, role="generator")
    return {
        "answer": result.text.strip(),
        "model": result.model_key,
        "latency_seconds": result.latency_seconds,
        "wait_seconds": result.wait_seconds,
        "tokens": result.total_tokens,
        "cached": result.cached,
    }


def _llm_details(result: Dict[str, Any]) -> Dict[str, Any]:
    """The model/timing fields shared by reformulator, critic and generator results."""
    keys = ["model", "latency_seconds", "wait_seconds", "tokens", "cached"]
    return {key: result.get(key) for key in keys}


def build_graph(
    store: Any,
    classifier: Callable[[str], Tuple[bool, float]],
    reformulate_fn: Callable[..., Dict[str, Any]],
    judge_fn: Callable[[str, List[Dict[str, Any]]], Dict[str, Any]],
    generate_fn: Callable[[str, List[Dict[str, Any]]], Dict[str, Any]] = short_cited_generate,
    part_generate_fn: Callable[[str, List[Dict[str, Any]]], Dict[str, Any]] = answer_part,
    max_retries: Optional[int] = None,
    top_k: Optional[int] = None,
    max_contexts: Optional[int] = None,
    use_classifier: Optional[bool] = None,
    early_stop: Optional[Dict[str, Any]] = None,
    per_part_generation: Optional[bool] = None,
    per_part_contexts: Optional[int] = None,
    verify_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    refuse_when_not_sufficient: Optional[bool] = None,
):
    """Build and compile the pipeline graph.

    Args:
        store: Vector store with ``search(query, top_k)``.
        classifier: ``query -> (retrieve: bool, confidence)``, e.g. ``should_retrieve``.
        reformulate_fn: ``reformulate(query, feedback=None, previous_queries=None)``.
        judge_fn: ``judge(query, chunks)`` returning at least ``verdict`` and ``missing``.
        generate_fn: ``(query, contexts) -> {"answer", ...}``; default is short-answer grounded generation.
        part_generate_fn: ``(sub_question, contexts) -> {"answer", ...}`` for per-part generation.
        max_retries, top_k, max_contexts, use_classifier, early_stop, per_part_generation,
            per_part_contexts: default to ``critic.max_retries``, ``retrieval.top_k`` and the
            ``graph`` block of the config.
        verify_fn: ``(answer, contexts, question=...) -> verification dict``; None = no verify step.
        refuse_when_not_sufficient: default ``generation.refuse_when_not_sufficient``.

    Returns:
        A compiled LangGraph app; run it with ``run_pipeline``.
    """
    config = load_config()
    if max_retries is None:
        max_retries = config["critic"]["max_retries"]
    if top_k is None:
        top_k = config["retrieval"]["top_k"]
    if max_contexts is None:
        max_contexts = config["graph"]["max_contexts"]
    if use_classifier is None:
        use_classifier = config["graph"]["use_classifier"]
    if early_stop is None:
        early_stop = config["graph"]["early_stop"]
    if per_part_generation is None:
        per_part_generation = config["graph"]["per_part_generation"]
    if per_part_contexts is None:
        per_part_contexts = config["graph"]["per_part_contexts"]
    if refuse_when_not_sufficient is None:
        refuse_when_not_sufficient = config["generation"]["refuse_when_not_sufficient"]

    def classify(state: PipelineState) -> Dict[str, Any]:
        """Block 1: retrieve or skip."""
        started = time.perf_counter()
        if use_classifier:
            decision, confidence = classifier(state["query"])
        else:
            decision, confidence = True, 1.0
        entry = {
            "step": "classify",
            "decision": "retrieve" if decision else "skip",
            "confidence": round(confidence, 3),
            "classifier_used": use_classifier,
            "seconds": time.perf_counter() - started,
        }
        return {"retrieve": decision, "trace": [entry]}

    def reformulate_node(state: PipelineState) -> Dict[str, Any]:
        """Block 2: first pass (rewrite/decompose) or retry with the critic's feedback."""
        is_retry = state.get("verdict") == "not_sufficient"
        retries = state.get("retries", 0) + (1 if is_retry else 0)
        if is_retry:
            result = reformulate_fn(state["query"], feedback=state["missing"], previous_queries=state.get("all_queries", []))
        else:
            result = reformulate_fn(state["query"])
        entry = {
            "step": "reformulate",
            "retry": retries,
            "mode": result["mode"],
            "queries": result["queries"],
            "reason": result.get("reason", ""),
            **_llm_details(result),
        }
        update = {
            "sub_queries": result["queries"],
            "all_queries": state.get("all_queries", []) + result["queries"],
            "retries": retries,
            "trace": [entry],
        }
        if not is_retry:
            update["first_mode"] = result["mode"]
            update["parts"] = result["queries"] if result["mode"] == "decompose" else []
        return update

    def retrieve(state: PipelineState) -> Dict[str, Any]:
        """Search every sub-query and merge the hits without duplicates."""
        started = time.perf_counter()
        hits_per_query = [store.search(query, top_k) for query in state["sub_queries"]]
        contexts, added = merge_contexts(
            state.get("contexts", []), hits_per_query, state.get("retries", 0), state["sub_queries"]
        )
        entry = {
            "step": "retrieve",
            "retry": state.get("retries", 0),
            "queries": state["sub_queries"],
            "hits": sum(len(hits) for hits in hits_per_query),
            "new_chunks": added,
            "total_chunks": len(contexts),
            "seconds": time.perf_counter() - started,
        }
        return {"contexts": contexts, "last_new_chunks": added, "trace": [entry]}

    def critique(state: PipelineState) -> Dict[str, Any]:
        """Block 3: is the selected evidence sufficient?"""
        shown = select_contexts(state["contexts"], max_contexts)
        result = judge_fn(state["query"], shown)
        missing = result.get("missing", "")
        stop_reason = decide_stop(
            result["verdict"],
            state.get("retries", 0),
            max_retries,
            missing,
            state.get("previous_missing", ""),
            state.get("last_new_chunks", 0),
            early_stop,
        )
        entry = {
            "step": "critique",
            "retry": state.get("retries", 0),
            "verdict": result["verdict"],
            "missing": result.get("missing", ""),
            "confidence": result.get("confidence"),
            "fallback": result.get("fallback", False),
            "chunks_judged": [chunk["chunk_id"] for chunk in shown],
            "stop_reason": stop_reason,
            **_llm_details(result),
        }
        return {
            "verdict": result["verdict"],
            "missing": missing,
            "previous_missing": missing,
            "stop_reason": stop_reason,
            "trace": [entry],
        }

    def generate_per_part(state: PipelineState) -> Dict[str, Any]:
        """Answer each sub-question from its own chunks, then join the answers (no extra LLM call)."""
        parts = []
        for part in state["parts"]:
            part_contexts = contexts_for_part(state["contexts"], part, per_part_contexts)
            result = part_generate_fn(part, part_contexts)
            used_all_chunks = False
            if is_not_found(result["answer"]):
                # The sub-question's own search can miss (e.g. a sub-question that names the wrong
                # paper); one more local call with all selected chunks recovers those cases.
                all_contexts = select_contexts(state["contexts"], max_contexts)
                if [c["chunk_id"] for c in all_contexts] != [c["chunk_id"] for c in part_contexts]:
                    retry_result = part_generate_fn(part, all_contexts)
                    if not is_not_found(retry_result["answer"]):
                        result, part_contexts, used_all_chunks = retry_result, all_contexts, True
            parts.append({"question": part, "contexts": part_contexts, "used_all_chunks": used_all_chunks, **result})
        answer = compose_answer(parts)
        entry = {
            "step": "generate",
            "per_part": True,
            "parts": [
                {
                    "question": part["question"],
                    "answer": part["answer"],
                    "chunks_used": [chunk["chunk_id"] for chunk in part["contexts"]],
                    "used_all_chunks": part["used_all_chunks"],
                }
                for part in parts
            ],
            "chunks_used": sorted({chunk["chunk_id"] for part in parts for chunk in part["contexts"]}),
            "answer": answer,
            "model": parts[0].get("model"),
            "latency_seconds": sum(part.get("latency_seconds") or 0.0 for part in parts),
            "wait_seconds": sum(part.get("wait_seconds") or 0.0 for part in parts),
            "tokens": sum(part.get("tokens") or 0 for part in parts),
            "cached": all(part.get("cached") for part in parts),
        }
        used = {chunk["chunk_id"]: chunk for part in parts for chunk in part["contexts"]}
        return {"answer": answer, "answer_contexts": list(used.values()), "trace": [entry]}

    def generate(state: PipelineState) -> Dict[str, Any]:
        """Answer from the selected chunks (none on the skip path); per part for decomposed questions."""
        if per_part_generation and state.get("retrieve") and len(state.get("parts", [])) > 1:
            return generate_per_part(state)
        contexts = select_contexts(state.get("contexts", []), max_contexts) if state.get("retrieve") else []
        if refuse_when_not_sufficient and state.get("retrieve") and state.get("verdict") == "not_sufficient":
            entry = {"step": "generate", "chunks_used": [], "answer": insufficient_text(), "refused": True}
            return {"answer": insufficient_text(), "answer_contexts": [], "trace": [entry]}
        result = generate_fn(state["query"], contexts)
        entry = {
            "step": "generate",
            "chunks_used": [chunk["chunk_id"] for chunk in contexts],
            "answer": result["answer"],
            "citations": result.get("citations", []),
            "invalid_citations": result.get("invalid_citations", []),
            "insufficient": result.get("insufficient", False),
            "uncited": result.get("uncited", False),
            **_llm_details(result),
        }
        return {"answer": result["answer"], "answer_contexts": contexts, "trace": [entry]}

    def verify(state: PipelineState) -> Dict[str, Any]:
        """Block 6: check the answer's claims against the chunks it was generated from."""
        started = time.perf_counter()
        report = verify_fn(state["answer"], state.get("answer_contexts", []), question=state["query"])
        entry = {
            "step": "verify",
            "status": report["status"],
            "n_claims": report["n_claims"],
            "n_supported": report["n_supported"],
            "supported_ratio": report["supported_ratio"],
            "claims": report["claims"],
            "seconds": time.perf_counter() - started,
        }
        return {"verification": report, "trace": [entry]}

    def route_after_classify(state: PipelineState) -> str:
        """Skip goes straight to generation; retrieve enters the critique loop."""
        return "reformulate" if state["retrieve"] else "generate"

    def route_after_critique(state: PipelineState) -> str:
        """Block 4: bounded retry. The critique node already decided whether (and why) to stop."""
        if state.get("stop_reason") is None:
            return "reformulate"
        return "generate"

    graph = StateGraph(PipelineState)
    graph.add_node("classify", classify)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("retrieve", retrieve)
    graph.add_node("critique", critique)
    graph.add_node("generate", generate)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify, {"reformulate": "reformulate", "generate": "generate"})
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("retrieve", "critique")
    graph.add_conditional_edges("critique", route_after_critique, {"reformulate": "reformulate", "generate": "generate"})
    if verify_fn is not None:
        graph.add_node("verify", verify)
        graph.add_edge("generate", "verify")
        graph.add_edge("verify", END)
    else:
        graph.add_edge("generate", END)
    return graph.compile()


def run_pipeline(app: Any, query: str) -> Dict[str, Any]:
    """Run one question through a compiled graph and return the final state (plus total seconds)."""
    started = time.perf_counter()
    initial: PipelineState = {"query": query, "retries": 0, "contexts": [], "all_queries": [], "trace": []}
    # Each loop round is 3 node visits; the default LangGraph limit (25) would cap max_retries at ~6.
    state = app.invoke(initial, config={"recursion_limit": 100})
    state = dict(state)
    state["total_seconds"] = time.perf_counter() - started
    return state
