"""Vanilla single-shot RAG: retrieve top_k chunks once, stuff them in a prompt, answer.

This is the baseline the agentic system is compared against.
"""

import time
from typing import Any, Dict

from src.agents.prompts import VANILLA_RAG_SYSTEM, build_vanilla_rag_prompt
from src.llm import call_llm


def vanilla_rag(question: str, store, top_k: int, role: str = "generator") -> Dict[str, Any]:
    """Answer ``question`` with one retrieval and one generation call.

    Args:
        question: The user question.
        store: A ``QdrantStore`` (anything with ``search(query, top_k)``).
        top_k: Number of chunks to retrieve.
        role: LLM role from the config used for generation.

    Returns:
        A dict with:
          answer, contexts (retrieved chunks),
          latency (seconds: real retrieval time + model call time; for a cached call the
                   model time is the one measured when the call was first made),
          tokens (input + output), plus input_tokens, output_tokens,
          retrieval_latency, generation_latency, model_id, cached.
    """
    retrieval_start = time.perf_counter()
    contexts = store.search(question, top_k)
    retrieval_latency = time.perf_counter() - retrieval_start

    prompt = build_vanilla_rag_prompt(question, contexts)
    result = call_llm(prompt, system=VANILLA_RAG_SYSTEM, role=role)

    return {
        "answer": result.text.strip(),
        "contexts": contexts,
        "latency": retrieval_latency + result.latency_seconds,
        "tokens": result.total_tokens,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "retrieval_latency": retrieval_latency,
        "generation_latency": result.latency_seconds,
        "model_id": result.model_id,
        "cached": result.cached,
    }
