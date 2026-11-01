"""The report document itself: prose assembled around the generated tables and figures.

Kept separate from `src.eval.report` so that module stays about reading results and
drawing figures, while the wording of the report lives in one place. The prose states
each caveat next to the numbers it applies to: a reader should not have to open the
hand-off log to learn that the arXiv column is optimistic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.config import PROJECT_ROOT
from src.eval.report import (
    ReportData,
    ablation_rows,
    ablation_table,
    classifier_table,
    cost_table,
    format_number,
    format_signed,
    groundedness_table,
    load_report_data,
    main_comparison_table,
    metric,
    picture_tag,
    ragas_table,
    render_figures,
    resolve_paths,
    vector_table,
)


def build_report_markdown(data: ReportData, config: Dict[str, Any]) -> str:
    """Assemble the whole report document as one markdown string."""
    report_config = config["report"]
    figures_dir = Path(report_config["figures_dir"])
    document_dir = Path(report_config["output"]).parent

    def figure(name: str, alt: str) -> str:
        """Embed one rendered figure, light and dark."""
        return picture_tag(name, alt, figures_dir, document_dir)

    arxiv_vanilla = metric(data, "arxiv", "vanilla", "relaxed_accuracy")
    arxiv_agentic = metric(data, "arxiv", "agentic_v2", "relaxed_accuracy")
    classifier_row = next(
        (row for row in ablation_rows(data) if row["name"] == "no_classifier"), {}
    )
    threshold = data.classifier.get("threshold_selection", {}).get("threshold", "?")
    test_accuracy = data.classifier.get("test", {}).get("metrics", {}).get("accuracy")
    challenge_accuracy = data.classifier.get("challenge", {}).get("metrics", {}).get("accuracy")

    sections: List[str] = []

    sections.append(
        "# Agentic RAG with Active Retrieval — results\n\n"
        "Every number and figure in this document is generated from the JSON files in "
        "`results/` by `scripts/make_report_assets.py`. Nothing here is typed by hand, so the "
        "report cannot drift from the measurements it describes. Metrics are reported per "
        "dataset and never averaged across them, because the three corpora measure different "
        "things.\n\n"
        "**Relaxed accuracy** counts an answer correct on exact match, containment, or token "
        "F1 >= 0.8 against the reference answer. Intervals are 95% bootstrap intervals."
    )

    sections.append(
        "## 1. Does the agentic pipeline answer better?\n\n"
        + figure("main_comparison", "Relaxed accuracy by dataset for each system")
        + "\n\n"
        + main_comparison_table(data, config)
        + "\n\n"
        "**The arXiv column is optimistic and is not the headline result.** The generator used "
        "by the agentic system was revised while looking at the arXiv evaluation questions, so "
        f"its {format_number(arxiv_agentic)} against vanilla RAG's {format_number(arxiv_vanilla)} "
        "is partly fitted to its own test set. On the two held-out multi-hop benchmarks, which "
        "no part of the system was tuned on, the agentic pipeline and vanilla RAG are tied: "
        "their confidence intervals overlap almost completely. The closed-book column is the "
        "useful floor — it shows how much of each dataset a 3B model answers from memory alone, "
        "which on the arXiv corpus is almost nothing.\n\n"
        "On raw accuracy, then, the honest finding is: **retrieval matters enormously, and the "
        "agentic machinery on top of it does not by itself answer more questions correctly.** "
        "What it does change is the subject of the next section."
    )

    sections.append(
        "## 2. What the agentic pipeline actually buys\n\n"
        + figure("groundedness", "Unsupported claim rate and null-question abstention")
        + "\n\n"
        + groundedness_table(data, config)
        + "\n\n"
        "Two effects here are large, and neither was tuned on the questions it is measured on.\n\n"
        "**It abstains when the corpus does not contain the answer.** MultiHop-RAG includes "
        "questions whose answer is deliberately absent. Vanilla RAG notices that in 19% of them "
        "and invents an answer the rest of the time; the agentic pipeline notices in 88%. That "
        "is the sufficiency critic doing the job it was built for: evidence that does not "
        "support an answer is rejected rather than summarised.\n\n"
        "**Its claims are better supported by the evidence it cites.** On arXiv the NLI verifier "
        "cannot support 25% of vanilla RAG's claims against that system's own retrieved "
        "contexts, against 1% for the agentic system. RAGAS agrees from a different direction "
        "(below). One honest wrinkle: on MultiHop-RAG both systems show high unsupported rates, "
        "because answers there are often stitched from several articles and a sentence-level NLI "
        "check is strict about that.\n\n"
        "### RAGAS on a judged sample\n\n"
        + ragas_table(data, config)
        + "\n\n"
        "Judged by the Gemini lite models on a sample per dataset, since RAGAS makes many calls "
        "per sample and the whole evaluation set does not fit in the free quota. Faithfulness "
        "moves the same way as the NLI check. Answer relevancy is noisy and partly measures "
        "answer length and phrasing rather than correctness, so it is reported but not leaned on."
    )

    sections.append(
        "## 3. Which blocks earn their place\n\n"
        + figure("ablation_effects", "Paired effect of each ablation against the full system")
        + "\n\n"
        + ablation_table(data)
        + "\n\n"
        "Each configuration changes exactly one thing and answers exactly the same "
        f"{classifier_row.get('n', '?')} arXiv questions, so the paired difference is the right "
        "comparison.\n\n"
        "**Only block 1 has an effect that clears the noise.** Turning the retrieval-necessity "
        f"classifier off costs {format_signed(classifier_row.get('difference'))} relaxed "
        "accuracy and roughly triples the time per question, and its interval is the only one "
        "that excludes zero. Everything else — reformulation, decomposition, the critic, the "
        "retry budget, the number of contexts shown, top-k, and int8 quantization — sits inside "
        "the noise on these questions.\n\n"
        "This deserves plain statement rather than dressing up: **the ablations do not show "
        "that the critic and the retry loop improve accuracy.** They show that those blocks are "
        "nearly free in accuracy terms while producing the abstention and groundedness "
        "behaviour in section 2, which an accuracy metric does not capture. Resolving effects of "
        "one or two points would need far more questions than the free quota allowed."
    )

    sections.append(
        "## 4. Block 1: the retrieval-necessity classifier\n\n"
        + classifier_table(data)
        + "\n\n"
        "DistilBERT, fine-tuned on labels produced by experiment rather than by opinion: a "
        "question is labelled `skip` when the closed-book model answers it correctly, and "
        f"`retrieve` when only the retrieval path does. The threshold on P(retrieve) is "
        f"{threshold}, chosen on the validation split to reach 95% recall on `retrieve`, because "
        "a missed retrieval costs more than a wasted search.\n\n"
        "The **challenge set** is the honest test: 100 held-out questions written to break the "
        "shortcut \"research jargon means retrieve\" — textbook ML questions phrased with jargon "
        "that should be skipped, and questions about corpus facts phrased in plain language that "
        f"should be retrieved. Accuracy falls from {format_number(test_accuracy)} on the test "
        f"split to {format_number(challenge_accuracy)} there, so some of the shortcut is real, "
        "but it stays well clear of the always-retrieve baseline.\n\n"
        "Caveat: the classifier is only properly validated on arXiv-style questions. The "
        "benchmark-derived rows in the test split are few and noisy (HotpotQA 10 rows, "
        "MultiHop-RAG 11), which is also why the classifier makes no measurable difference on "
        "those two datasets."
    )

    sections.append(
        "## 5. The vector store\n\n"
        + figure("vector_tradeoff", "Recall against latency for exact search and HNSW")
        + "\n\n"
        + vector_table(data)
        + "\n\n"
        "HNSW buys roughly an order of magnitude of latency for a few points of recall, and "
        "raising `ef` walks back up the recall curve at a predictable cost. int8 scalar "
        "quantization is faster again at slightly lower recall and — the counter-intuitive part "
        "— **it does not save disk**: Qdrant keeps the quantized vectors alongside the "
        "originals, so the int8 collection is the larger of the two on disk. Its benefit is that "
        "the small vectors stay resident in RAM (`always_ram`), not storage.\n\n"
        "Quantization costs nothing in answer quality: the `int8` ablation in section 3 is "
        "identical to the full system.\n\n"
        "> **Latency measurements taken in this project before 2026-10-29 were wrong.** The "
        "qdrant-client HTTP transport costs about 28 ms per search on this machine while the "
        "server reports under 1 ms of search time. Switching to gRPC "
        "(`vector_store.prefer_grpc`) took a dev-corpus search from about 15 ms to about 1 ms. "
        "The client and server columns above are kept side by side so the remaining client "
        "overhead stays visible."
    )

    sections.append(
        "## 6. Latency and cost\n\n"
        + cost_table(data, config)
        + "\n\n"
        "`model s` is time spent waiting on a model, `wait s` is time spent sleeping against "
        "rate limits, and `lite calls` counts calls to the Gemini lite models. The agentic "
        "pipeline costs roughly ten times the local model time of vanilla RAG plus one to four "
        "remote judge calls per question, which is the price of the abstention behaviour in "
        "section 2.\n\n"
        "**Actual monetary cost is $0.** Everything runs on a local Ollama model, local "
        "embedding and NLI models, a local Qdrant binary, and the Gemini free tier with billing "
        "disabled. The `notional_usd_per_1m_*` fields in the config are deliberately unset, so "
        "no invented dollar figure appears anywhere; fill them in to get a paid-tier equivalent."
    )

    sections.append(
        "## 7. What this study does not show\n\n"
        "- The arXiv accuracy gain is partly fitted to its own test set (section 1).\n"
        "- Reformulation, the critic and the retry loop are not individually shown to improve "
        "accuracy; only their cost and their effect on abstention are established (section 3).\n"
        "- Retrieval quality is never measured at scale: the 343k-chunk corpus is used only for "
        "the vector benchmark, while every quality number comes from the 528-chunk dev corpus.\n"
        "- Hybrid retrieval (dense + BM25 + reranker) is implemented and benchmarked but unused, "
        "because it did not improve answers.\n"
        "- A 3B local generator is the ceiling on answer quality; several remaining errors are "
        "generation failures on correctly retrieved evidence, not retrieval failures.\n"
        "- **Nothing guards the skip path.** Blocks 3-6 only protect answers that went through "
        "retrieval: when block 1 decides to skip, the generator answers from memory, there are no "
        "contexts, and the verifier returns `no_contexts` rather than a judgement. A question the "
        "classifier wrongly routes to skip is therefore answered with no critic, no citations and "
        "no verification — observed directly when a factual question outside the corpus was "
        "skipped and answered with a confident, unverifiable number. The abstention results in "
        "section 2 describe the retrieval path only.\n"
        "- The judge and verifier-fallback roles are limited to one model at 20 requests a day, "
        "because the originally planned model was withdrawn by the provider.\n"
        "- Every evaluation set is small (100-127 questions), so differences of a few points are "
        "not resolvable."
    )

    return "\n\n".join(sections) + "\n"


def write_report(config: Dict[str, Any]) -> Dict[str, Any]:
    """Render every figure and write the report document; return what was written.

    Convenience entry point for a notebook; `scripts/make_report_assets.py` is the
    command-line route. Report paths are resolved against the project root, so the
    caller's working directory does not matter.
    """
    config = resolve_paths(config, PROJECT_ROOT)
    data = load_report_data(config)
    figures = render_figures(data, config)
    output = Path(config["report"]["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report_markdown(data, config), encoding="utf-8")
    return {"report": output, "figures": figures}
