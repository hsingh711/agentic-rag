"""Streamlit demo for the agentic RAG pipeline.

Shows what the system decided, not just what it answered: the retrieval decision, the
reformulated queries, the chunks it searched, the critic's verdict in each round, the
cited answer, and the NLI check of every claim.

    streamlit run app/app.py

Needs Qdrant and Ollama running and a Gemini key in `.env`; the sidebar says which of
those is missing rather than failing with a stack trace. When the free Gemini quota for
the day runs out the app says so and explains when it resets.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from app.status import (  # noqa: E402
    ServiceStatus,
    blocking,
    check_all,
    claim_icon,
    summarise,
)
from src.agents.critic import judge  # noqa: E402
from src.agents.reformulator import reformulate  # noqa: E402
from src.classifier.predict import should_retrieve  # noqa: E402
from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.generation.verifier import NLIScorer, verify_answer  # noqa: E402
from src.graph.pipeline import build_graph, run_pipeline  # noqa: E402
from src.index.embedder import Embedder  # noqa: E402
from src.index.qdrant_store import QdrantStore, make_client  # noqa: E402
from src.llm import DailyLimitReached  # noqa: E402


@st.cache_resource(show_spinner=False)
def get_config() -> Dict[str, Any]:
    """Load the config once per session, with `.env` applied first."""
    load_dotenv(PROJECT_ROOT / ".env")
    return load_config()


@st.cache_resource(show_spinner="Loading the embedder and the vector store...")
def get_store(_config: Dict[str, Any]) -> QdrantStore:
    """Build the Qdrant store once; the embedding model is the slow part."""
    vector_config = _config["vector_store"]
    return QdrantStore(
        make_client(vector_config),
        _config["graph"]["collection"],
        Embedder.from_config(_config),
        vector_config,
        use_quantization=False,
    )


@st.cache_resource(show_spinner="Loading the NLI verifier...")
def get_scorer(_config: Dict[str, Any]) -> NLIScorer:
    """Load the NLI model once; it runs on CPU so it can sit beside Ollama."""
    return NLIScorer.from_config(_config)


@st.cache_resource(show_spinner="Building the pipeline...")
def get_graph(_config: Dict[str, Any], use_classifier: bool, verify: bool):
    """Compile the graph for one combination of switches.

    Streamlit caches per argument set, so toggling a switch builds that variant once
    and then reuses it.
    """
    verify_fn = None
    if verify and _config["verifier"]["enabled"]:
        scorer = get_scorer(_config)
        verify_fn = lambda answer, contexts, question=None: verify_answer(  # noqa: E731
            answer, contexts, scorer, question=question
        )
    return build_graph(
        get_store(_config),
        classifier=should_retrieve,
        reformulate_fn=reformulate,
        judge_fn=judge,
        use_classifier=use_classifier,
        verify_fn=verify_fn,
    )


def render_status(statuses: List[ServiceStatus]) -> None:
    """Show one line per dependency in the sidebar."""
    for status in statuses:
        icon = "🟢" if status.ok else "🔴"
        st.sidebar.markdown(f"{icon} **{status.name}** — {status.detail}")
        if not status.ok and status.remedy:
            st.sidebar.caption(status.remedy)


def render_summary(summary: Dict[str, Any]) -> None:
    """Show the run's headline numbers as a row of metrics."""
    columns = st.columns(5)
    columns[0].metric("decision", summary["decision"])
    columns[1].metric("retrieval rounds", summary["retrieval_rounds"])
    columns[2].metric("retries", summary["retries"])
    columns[3].metric("lite calls", summary["lite_calls"])
    seconds = summary["total_seconds"]
    columns[4].metric("total", f"{seconds:.1f}s" if seconds is not None else "—")


def render_trace(trace: List[Dict[str, Any]]) -> None:
    """Show each pipeline step in order, with the detail that step produced."""
    for entry in trace:
        step = entry.get("step")
        if step == "classify":
            with st.expander(
                f"1. Retrieval-necessity classifier → **{entry['decision']}**", expanded=True
            ):
                st.write(
                    f"Confidence {entry['confidence']}, decided in "
                    f"{entry['seconds'] * 1000:.0f} ms, with no LLM call."
                )
                if entry["decision"] == "skip":
                    st.info("The vector store is never queried on the skip path.")
        elif step == "reformulate":
            title = f"2. Reformulation → **{entry['mode']}**"
            if entry.get("retry"):
                title += f" (retry {entry['retry']})"
            with st.expander(title, expanded=True):
                for query in entry["queries"]:
                    st.markdown(f"- `{query}`")
                st.caption(_model_caption(entry))
        elif step == "retrieve":
            with st.expander(
                f"3. Retrieval → {entry['hits']} hits, {entry['new_chunks']} new "
                f"({entry['total_chunks']} unique so far)",
                expanded=False,
            ):
                st.write(f"Searched in {entry['seconds'] * 1000:.0f} ms.")
        elif step == "critique":
            verdict = entry["verdict"]
            icon = "✅" if verdict == "sufficient" else "🔁"
            with st.expander(
                f"4. Sufficiency critic → {icon} **{verdict}**", expanded=verdict != "sufficient"
            ):
                st.write(f"Confidence {entry['confidence']}.")
                if entry.get("missing"):
                    st.warning(f"Missing: {entry['missing']}")
                if entry.get("stop_reason"):
                    st.caption(f"Loop stopped: {entry['stop_reason']}")
                st.caption(_model_caption(entry))
        elif step == "generate":
            with st.expander(
                f"5. Generation → {len(entry['chunks_used'])} chunks used", expanded=False
            ):
                if entry.get("per_part"):
                    st.caption("Each sub-question was answered from its own chunks.")
                st.code(entry["answer"], language=None)
                st.caption(_model_caption(entry))
        elif step == "verify":
            _render_verification(entry)


def _model_caption(entry: Dict[str, Any]) -> str:
    """One line naming the model behind a step, its latency, and whether it was cached."""
    if not entry.get("model"):
        return ""
    cached = ", served from cache" if entry.get("cached") else ""
    latency = entry.get("latency_seconds")
    latency_text = f"{latency:.1f}s" if latency is not None else "—"
    return f"{entry['model']}, {latency_text}{cached}"


def _render_verification(entry: Dict[str, Any]) -> None:
    """Show the NLI check of each claim in the answer."""
    if entry.get("status") != "verified":
        with st.expander(f"6. Verification → {entry.get('status')}", expanded=False):
            st.caption(
                "Nothing to check: the answer makes no checkable claim, or there were no "
                "contexts to check it against."
            )
        return
    supported = entry["n_supported"]
    total = entry["n_claims"]
    with st.expander(
        f"6. Verification → {supported}/{total} claims supported by the evidence",
        expanded=supported < total,
    ):
        for claim in entry["claims"]:
            st.markdown(
                f"{claim_icon(claim['label'])} **{claim['label']}** "
                f"(entailment {claim['entailment']:.2f}) — {claim['claim']}"
            )
            if claim.get("chunk_id"):
                st.caption(f"best matching chunk: `{claim['chunk_id']}`")


def render_contexts(contexts: List[Dict[str, Any]]) -> None:
    """Show the chunks the pipeline retrieved, newest round last."""
    if not contexts:
        st.caption("No chunks were retrieved: the classifier answered from the model's own knowledge.")
        return
    for chunk in contexts:
        header = f"`{chunk['chunk_id']}`"
        if chunk.get("title"):
            header += f" — {chunk['title']}"
        with st.expander(header, expanded=False):
            score = chunk.get("score")
            if score is not None:
                st.caption(f"score {score:.3f}, round {chunk.get('round', 0)}")
            st.write(chunk.get("text", ""))


def render_answer(state: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Show the answer, flagging a refusal as a deliberate outcome rather than a failure."""
    answer = state.get("answer", "")
    refusal = config["generation"]["insufficient_text"].lower()
    st.subheader("Answer")
    if answer.strip().lower().startswith(refusal):
        st.warning(f"**{answer}**")
        st.caption(
            "The critic judged the retrieved evidence insufficient, so the system refused "
            "rather than guessing. On MultiHop-RAG's unanswerable questions it does this 88% "
            "of the time, against 19% for vanilla RAG."
        )
    else:
        st.success(answer)


def main() -> None:
    """Draw the page and run a question when one is submitted."""
    config = get_config()
    app_config = config["app"]

    st.set_page_config(page_title=app_config["title"], page_icon=app_config["icon"], layout="wide")
    st.title(app_config["title"])
    st.caption(app_config["subtitle"])

    st.sidebar.header("Services")
    statuses = check_all(config)
    render_status(statuses)
    if st.sidebar.button("Re-check"):
        st.cache_resource.clear()
        st.rerun()

    st.sidebar.header("Pipeline")
    use_classifier = st.sidebar.toggle(
        "Block 1: retrieval-necessity classifier",
        value=config["graph"]["use_classifier"],
        help="Off means always retrieve, which the ablations show costs 20 accuracy points "
        "and about three times the time per question.",
    )
    verify = st.sidebar.toggle(
        "Block 6: NLI answer verification",
        value=config["verifier"]["enabled"],
        help="Splits the answer into claims and checks each against the retrieved contexts.",
    )
    st.sidebar.caption(
        f"Critic retries are capped at {config['critic']['max_retries']} "
        f"(`critic.max_retries`), and every model reply is cached, so asking the same "
        "question twice costs no quota."
    )

    failed = blocking(statuses)
    if failed:
        st.error(
            "Not ready to answer: "
            + "; ".join(f"{status.name} — {status.detail}" for status in failed)
            + ". See the sidebar for how to fix each one."
        )

    examples = config["graph"]["demo_questions"]
    chosen = st.selectbox(
        "Example questions",
        ["(type your own)"] + examples,
        help="The first example is answerable without retrieval, so the classifier should skip.",
    )
    default = "" if chosen == "(type your own)" else chosen
    question = st.text_input("Question", value=default, placeholder="Ask about the indexed papers...")
    run = st.button("Run the pipeline", type="primary", disabled=bool(failed) or not question.strip())

    if not run:
        return

    graph = get_graph(config, use_classifier, verify)
    try:
        with st.spinner("Running blocks 1-6..."):
            state = run_pipeline(graph, question.strip())
    except DailyLimitReached as error:
        st.warning(
            f"**The free Gemini quota for today is used up.** {error}\n\n"
            "The daily counter resets at midnight Pacific time. Questions answered earlier "
            "are cached and can be re-asked for free in the meantime, and turning off the "
            "classifier toggle will not help: it increases the number of calls."
        )
        return
    except Exception as error:  # noqa: BLE001 - the page should explain, not traceback
        st.error(f"The run failed: {type(error).__name__}: {error}")
        st.caption(
            "If this mentions a connection, check Qdrant and Ollama in the sidebar and "
            "press Re-check."
        )
        return

    render_answer(state, config)
    render_summary(summarise(state))

    trace_tab, chunks_tab = st.tabs(["Pipeline trace", f"Retrieved chunks ({len(state.get('contexts', []))})"])
    with trace_tab:
        render_trace(state.get("trace", []))
    with chunks_tab:
        render_contexts(state.get("contexts", []))


if __name__ == "__main__":
    main()
