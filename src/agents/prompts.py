"""All prompt templates live here, not inline in the code that uses them."""

from typing import Any, Dict, List

VANILLA_RAG_SYSTEM = (
    "You answer questions about research papers using only the provided context passages. "
    "If the context does not contain the answer, say that you cannot find it in the context. "
    "Keep the answer short and factual."
)

CLOSED_BOOK_SYSTEM = "You answer questions briefly and factually."


def format_contexts(contexts: List[Dict[str, Any]]) -> str:
    """Number the retrieved chunks so the model can refer to them."""
    blocks = []
    for number, chunk in enumerate(contexts, start=1):
        blocks.append(f"[{number}] {chunk['title']}\n{chunk['text']}")
    return "\n\n".join(blocks)


def build_vanilla_rag_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    """Stuff all retrieved chunks and the question into one prompt."""
    return f"Context:\n{format_contexts(contexts)}\n\nQuestion: {question}\n\nAnswer:"


def build_closed_book_prompt(question: str) -> str:
    """Prompt for answering with no retrieval."""
    return f"Question: {question}\n\nAnswer:"


# ---------------------------------------------------------------------------
# Section 5: label generation
# ---------------------------------------------------------------------------

# Used for BOTH answers of the labelling experiment, so closed-book and RAG differ only
# in whether context is given. Short answers make exact-match / F1 scoring meaningful.
# No "say unknown" escape hatch: in the pilot the 3B model then answered "unknown" even to
# easy facts it knows (e.g. who played the Joker in The Dark Knight), which hid real
# closed-book knowledge and dropped 58% of queries.
LABEL_ANSWER_SYSTEM = (
    "Answer the question with a short phrase only (a name, number, term or a few words), "
    "not a full sentence. Always give your best answer, even if you are not sure."
)

CHUNK_QUESTION_SYSTEM = "You write precise quiz questions about excerpts from research papers."


def build_label_closed_book_prompt(question: str) -> str:
    """Closed-book prompt for the labelling experiment."""
    return f"Question: {question}\nShort answer:"


def build_label_rag_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    """Prompt with retrieved context for the labelling experiment."""
    return f"Context:\n{format_contexts(contexts)}\n\nUse the context to answer.\nQuestion: {question}\nShort answer:"


def build_concept_question_prompt(topic: str, count: int) -> str:
    """Textbook questions whose answer is one standard term (hard 'skip' candidates)."""
    return (
        f"Write {count} different quiz questions about this topic: {topic}.\n"
        "Rules:\n"
        "- Each question asks for well-known textbook knowledge that any machine learning course covers, "
        "not a recent research result.\n"
        "- Use technical vocabulary naturally, the way a researcher would ask.\n"
        "- The answer is ONE standard term, name or short phrase of 1 to 4 words "
        "(for example: 'dropout', 'softmax', 'inverted index').\n"
        "- The answer must not appear in the question.\n"
        "Return only JSON in this exact format:\n"
        '{"items": [{"question": "...", "answer": "..."}]}'
    )


def build_plain_chunk_question_prompt(chunks: List[Dict[str, Any]]) -> str:
    """One plain-language question per excerpt (hard 'retrieve' candidates)."""
    excerpts = []
    for number, chunk in enumerate(chunks, start=1):
        excerpts.append(f"Excerpt {number}:\n{chunk['text']}")
    excerpt_text = "\n\n".join(excerpts)
    return (
        f"{excerpt_text}\n\n"
        "For EACH excerpt above, write 1 question about one specific fact in it.\n"
        "Rules:\n"
        "- Write it in plain everyday language, as a curious non-expert would ask.\n"
        "- Do NOT use the paper's title, the method's or model's name, or acronyms. "
        "Describe the thing instead (e.g. 'the method that speeds up avatar animation on phones').\n"
        "- Still be specific enough that only this excerpt answers it.\n"
        "- The answer is a short span of 1 to 6 words copied exactly from the excerpt "
        "and must not appear in the question.\n"
        "Return only JSON in this exact format:\n"
        '{"items": [{"excerpt": 1, "question": "...", "answer": "..."}]}'
    )


def build_chunk_question_prompt(chunks: List[Dict[str, Any]], questions_per_chunk: int) -> str:
    """Ask for ``questions_per_chunk`` fact questions per excerpt, returned as JSON."""
    excerpts = []
    for number, chunk in enumerate(chunks, start=1):
        excerpts.append(f"Excerpt {number} (paper: {chunk['title']}):\n{chunk['text']}")
    excerpt_text = "\n\n".join(excerpts)
    return (
        f"{excerpt_text}\n\n"
        f"For EACH excerpt above, write {questions_per_chunk} questions.\n"
        "Rules:\n"
        "- Each question asks for one specific fact stated in that excerpt "
        "(a method or model name, dataset, number, result or finding).\n"
        "- The question must make sense on its own: name the method, model or topic. "
        "Never write 'this paper', 'the excerpt', 'the authors' or 'the proposed method'.\n"
        "- The answer is a short span of 1 to 6 words copied exactly from the excerpt.\n"
        "- The answer must not appear in the question.\n"
        "Return only JSON in this exact format:\n"
        '{"items": [{"excerpt": 1, "question": "...", "answer": "..."}]}'
    )


# ---------------------------------------------------------------------------
# Section 7: query reformulation (Block 2)
# ---------------------------------------------------------------------------

REFORMULATOR_SYSTEM = (
    "You turn user questions into search queries for a document retrieval system. "
    "You never answer the question and never add facts that are not in it."
)

# Shared by both prompts. Gemma has no JSON mode, so the format is spelled out in the prompt.
_SUB_QUESTION_RULES = (
    "- Every query must make sense on its own. Never use pronouns such as 'he', 'it', 'they' "
    "or 'that person'; refer to an unknown entity by its description instead "
    "(e.g. 'the director of the film Inception').\n"
    "- Keep names, numbers, titles and technical terms exactly as written in the question.\n"
)


def build_reformulate_prompt(query: str, max_queries: int) -> str:
    """First-pass prompt: the model chooses between rewriting and decomposing the question."""
    return (
        f"Question: {query}\n\n"
        "Choose ONE mode:\n"
        "- decompose: answering needs two or more separate lookups. This is the case when\n"
        "  (a) an intermediate entity must be found first and then a fact about it "
        "(e.g. 'the director of film X', 'the province where Y collected stories'), or\n"
        "  (b) two or more entities are compared (each one needs its own lookup).\n"
        f"  Write 2 to {max_queries} sub-questions, one per lookup, in the order they must be answered.\n"
        "- rewrite: one lookup is enough. Rewrite the question as one clear, specific search query: "
        "spell out vague words and keep it short.\n"
        "Rules:\n"
        f"{_SUB_QUESTION_RULES}"
        # Section 8 follow-up: on joined questions the model attached one part's paper title to
        # the other part ("... proposed in 'Q-SPT ...'"), so that part's search found nothing.
        "- If the question joins independent parts (e.g. '..., and ...'), each sub-question keeps only "
        "the names and titles of its own part. Never attach a paper title, dataset or method name from "
        "one part to another part.\n"
        "Examples:\n"
        'Question: What nationality is the director of Inception?\n'
        '{"mode": "decompose", "queries": ["Who directed the film Inception?", '
        '"What is the nationality of the director of the film Inception?"], "reason": "the director must be found first"}\n'
        "Question: Who is older, Ada Lovelace or Mary Shelley?\n"
        '{"mode": "decompose", "queries": ["When was Ada Lovelace born?", "When was Mary Shelley born?"], '
        '"reason": "comparison of two people"}\n'
        "Question: whats rag\n"
        '{"mode": "rewrite", "queries": ["definition of retrieval-augmented generation (RAG)"], '
        '"reason": "one vague concept"}\n\n'
        "Return only JSON in this exact format:\n"
        '{"mode": "rewrite or decompose", "queries": ["..."], "reason": "one short sentence"}'
    )


def build_retry_prompt(query: str, feedback: str, previous_queries: List[str], max_queries: int) -> str:
    """Retry prompt: the critic said the evidence was not sufficient and named what is missing."""
    previous_text = "\n".join(f"- {previous}" for previous in previous_queries) or "- (none recorded)"
    return (
        f"Original question: {query}\n\n"
        f"Searches already tried:\n{previous_text}\n\n"
        f"A reviewer read the retrieved passages and said this is still missing: {feedback}\n\n"
        f"Write 1 to {max_queries} NEW search queries that target exactly the missing information. "
        "Do not repeat the searches already tried; use different wording, synonyms or a more specific angle.\n"
        "Rules:\n"
        f"{_SUB_QUESTION_RULES}"
        "Return only JSON in this exact format:\n"
        '{"queries": ["..."], "reason": "one short sentence"}'
    )


# ---------------------------------------------------------------------------
# Section 8: retrieval sufficiency critic (Block 3)
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = (
    "You are a strict reviewer for a question answering system. You decide whether the retrieved "
    "passages contain enough information to answer the question. You never use your own knowledge "
    "of the answer; only what the passages state counts."
)


def format_critic_passages(chunks: List[Dict[str, Any]]) -> str:
    """Passages labelled with their chunk_id so the critic can refer to them."""
    blocks = []
    for chunk in chunks:
        blocks.append(f"[{chunk['chunk_id']}] {chunk.get('title', '')}\n{chunk['text']}")
    return "\n\n".join(blocks)


def build_critic_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Ask for a sufficient / not_sufficient verdict plus what is missing."""
    passages = format_critic_passages(chunks) if chunks else "(no passages were retrieved)"
    return (
        f"Question: {question}\n\n"
        f"Retrieved passages:\n{passages}\n\n"
        "Decide:\n"
        "- sufficient: the passages explicitly state everything needed to answer the question. "
        "For a question with several parts, EVERY part must be covered.\n"
        "- not_sufficient: some needed fact is absent, only loosely related, or about a different "
        "entity than the one asked about.\n"
        "If not_sufficient, write in 'missing' exactly which fact is missing, specific enough to search "
        "for (name the entity, e.g. 'the recall@5 of MatRAG on HotpotQA'). If sufficient, leave it empty.\n"
        "'confidence' is how sure you are of the verdict, from 0.0 to 1.0.\n"
        "Return only JSON in this exact format:\n"
        '{"verdict": "sufficient or not_sufficient", "missing": "...", "confidence": 0.0}'
    )


# ---------------------------------------------------------------------------
# Section 8 follow-up: per-part answers for decomposed questions
# ---------------------------------------------------------------------------

PART_NOT_FOUND = "not found"

# "Short phrase" alone made the 3B model drop head nouns ("curvature-aware" for "a curvature-aware
# objective", "Ghost nodes" for "ghost node proxies"), so the full term is asked for explicitly.
PART_ANSWER_SYSTEM = (
    "Answer the question using only the context. Reply with the complete name, number or phrase exactly "
    "as it is written in the context (keep the whole term, e.g. 'a curvature-aware objective', not "
    "'curvature-aware'), not a full sentence. If the context does not contain the answer, reply exactly: "
    f"{PART_NOT_FOUND}"
)


def build_part_answer_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    """Prompt for one sub-question of a decomposed question."""
    return f"Context:\n{format_contexts(contexts)}\n\nQuestion: {question}\nShort answer:"


# ---------------------------------------------------------------------------
# Section 9: grounded generation (Block 5) and answer verification (Block 6)
# ---------------------------------------------------------------------------

INSUFFICIENT_EVIDENCE = "Insufficient evidence"

GROUNDED_SYSTEM = (
    "You answer questions using ONLY the passages provided. Every passage starts with its id in square "
    "brackets. Write a short, factual answer. After every statement, cite the id of the passage that "
    "supports it, copied exactly, like [2610.00554:abstract:0]. Never state anything that the passages do "
    "not say, and never use your own knowledge. If the passages do not contain the answer, reply exactly: "
    f"{INSUFFICIENT_EVIDENCE}"
)


def build_grounded_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Passages labelled with chunk_ids (so the answer can cite them) followed by the question."""
    return f"Passages:\n{format_critic_passages(chunks)}\n\nQuestion: {question}\n\nAnswer (with citations):"


# Diagnosis (section 10): with GROUNDED_SYSTEM the 3B generator often answered with only a citation,
# a paper title or "Insufficient evidence" although the passage held the answer, while the plain
# short-answer prompt below got it right. So the model only writes the short answer and the
# citations are attached afterwards from the chunks that contain it.
SHORT_GROUNDED_SYSTEM = (
    "Answer the question using ONLY the passages. Reply with the shortest complete answer: a name, number or "
    "short phrase (one short sentence only if the question needs it). Do not repeat the question and do not "
    "cite sources. Never use your own knowledge. If the passages do not contain the answer, reply exactly: "
    f"{INSUFFICIENT_EVIDENCE}"
)


def build_short_grounded_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Numbered passages followed by the question; the reply is a short answer without citations."""
    return f"Passages:\n{format_contexts(chunks)}\n\nQuestion: {question}\nShort answer:"


VERIFIER_FALLBACK_SYSTEM = (
    "You check whether a claim is supported by the given passages. Only what the passages state counts; "
    "never use your own knowledge."
)


def build_verifier_fallback_prompt(claim: str, chunks: List[Dict[str, Any]]) -> str:
    """Ask an LLM judge whether one claim is supported by the passages (rare fallback to the NLI model)."""
    return (
        f"Passages:\n{format_critic_passages(chunks)}\n\nClaim: {claim}\n\n"
        "Is the claim supported by the passages? Answer 'supported' only if the passages state it, "
        "'contradicted' if they state the opposite, otherwise 'unsupported'.\n"
        'Return only JSON in this exact format: {"label": "supported or unsupported or contradicted"}'
    )
