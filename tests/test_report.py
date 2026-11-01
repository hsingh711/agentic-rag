"""Tests for the section 12 report assets (synthetic results; no files from results/, no LLM)."""

from pathlib import Path

import pytest

from src.eval.report import (
    ReportData,
    ablation_rows,
    ablation_table,
    classifier_table,
    confidence_interval,
    cost_table,
    figure_main_comparison,
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
    significant,
    vector_rows,
    vector_table,
)
from src.eval.report_document import build_report_markdown


@pytest.fixture
def data():
    """A small stand-in for the real results files, with the same shape."""
    return ReportData(
        eval_summary={
            "arxiv": {
                "closed_book": {"n": 10, "relaxed_accuracy": 0.3, "relaxed_accuracy_ci95": [0.2, 0.4]},
                "vanilla": {
                    "n": 10,
                    "relaxed_accuracy": 0.6,
                    "relaxed_accuracy_ci95": [0.5, 0.7],
                    "unsupported_claim_rate": 0.25,
                    "contradicted_claim_rate": 0.1,
                    "null_abstention_rate": None,
                    "false_abstention_rate": 0.2,
                    "skip_rate": 0.0,
                    "mean_model_seconds": 0.8,
                    "mean_wait_seconds": 0.0,
                    "mean_lite_calls": 0.0,
                    "mean_tokens_local": 1395.0,
                },
                "agentic_v2": {
                    "n": 10,
                    "relaxed_accuracy": 0.9,
                    "relaxed_accuracy_ci95": [0.8, 0.95],
                    "unsupported_claim_rate": 0.01,
                    "contradicted_claim_rate": 0.0,
                    "null_abstention_rate": None,
                    "false_abstention_rate": 0.01,
                    "skip_rate": 0.4,
                    "mean_model_seconds": 8.6,
                    "mean_wait_seconds": 0.0,
                    "mean_lite_calls": 1.3,
                    "mean_tokens_local": 919.0,
                },
            },
            "hotpotqa": {
                "closed_book": {"n": 10, "relaxed_accuracy": 0.26, "relaxed_accuracy_ci95": [0.18, 0.35]},
                "vanilla": {
                    "n": 10,
                    "relaxed_accuracy": 0.52,
                    "relaxed_accuracy_ci95": [0.43, 0.62],
                    "unsupported_claim_rate": 0.23,
                    "null_abstention_rate": None,
                },
                "agentic_v2": {
                    "n": 10,
                    "relaxed_accuracy": 0.52,
                    "relaxed_accuracy_ci95": [0.43, 0.62],
                    "unsupported_claim_rate": 0.06,
                    "null_abstention_rate": None,
                },
            },
            "multihoprag": {
                "closed_book": {"n": 10, "relaxed_accuracy": 0.19, "relaxed_accuracy_ci95": [0.12, 0.27]},
                "vanilla": {
                    "n": 10,
                    "relaxed_accuracy": 0.4,
                    "relaxed_accuracy_ci95": [0.3, 0.5],
                    "unsupported_claim_rate": 0.56,
                    "null_abstention_rate": 0.19,
                },
                "agentic_v2": {
                    "n": 10,
                    "relaxed_accuracy": 0.42,
                    "relaxed_accuracy_ci95": [0.32, 0.52],
                    "unsupported_claim_rate": 0.23,
                    "null_abstention_rate": 0.88,
                },
            },
        },
        ragas={
            "arxiv": {
                "vanilla": {"n": 40, "faithfulness": 0.69, "answer_relevancy": 0.34},
                "agentic_v2": {"n": 30, "faithfulness": 0.92, "answer_relevancy": 0.3},
            }
        },
        ablations={
            "full": {
                "n": 103,
                "relaxed_accuracy": 0.903,
                "description": "current default",
                "mean_model_seconds": 8.6,
                "mean_lite_calls": 1.3,
            },
            "no_classifier": {
                "n": 103,
                "relaxed_accuracy": 0.699,
                "description": "block 1 off",
                "mean_model_seconds": 25.3,
                "mean_lite_calls": 3.61,
                "paired_vs_full": {"difference": -0.204, "ci95": [-0.291, -0.117]},
            },
            "no_reformulation": {
                "n": 103,
                "relaxed_accuracy": 0.922,
                "description": "block 2 off",
                "mean_model_seconds": 4.3,
                "mean_lite_calls": 0.61,
                "paired_vs_full": {"difference": 0.019, "ci95": [-0.019, 0.058]},
            },
        },
        classifier={
            "threshold_selection": {"threshold": 0.1592},
            "test": {
                "metrics": {
                    "accuracy": 0.8919,
                    "macro_f1": 0.889,
                    "retrieve_recall": 0.9176,
                    "skip_rate": 0.4122,
                    "count": 148,
                },
                "confidence_intervals_95": {"accuracy": {"low": 0.8446, "high": 0.9392}},
            },
            "challenge": {
                "metrics": {
                    "accuracy": 0.8,
                    "macro_f1": 0.7993,
                    "retrieve_recall": 0.86,
                    "skip_rate": 0.44,
                    "count": 100,
                }
            },
            "baseline_always_retrieve_test": {
                "accuracy": 0.5743,
                "macro_f1": 0.3648,
                "retrieve_recall": 1.0,
                "skip_rate": 0.0,
            },
        },
        vector={
            "exact": {"engine": "faiss_flat_ip", "recall_at_k": 1.0, "p50_ms": 21.1},
            "approximate": {
                "fp32@ef32": {
                    "quantization": "none",
                    "hnsw_ef": 32,
                    "recall_at_k": 0.87,
                    "p50_ms": 2.11,
                    "server_p50_ms": 0.62,
                    "disk_mb": 1100.8,
                },
                "fp32@ef256": {
                    "quantization": "none",
                    "hnsw_ef": 256,
                    "recall_at_k": 0.99,
                    "p50_ms": 3.84,
                    "server_p50_ms": 1.95,
                    "disk_mb": 1100.8,
                },
                "int8@ef32": {
                    "quantization": "scalar",
                    "hnsw_ef": 32,
                    "recall_at_k": 0.855,
                    "p50_ms": 1.91,
                    "server_p50_ms": 0.45,
                    "disk_mb": 1234.0,
                },
                "int8@ef256": {
                    "quantization": "scalar",
                    "hnsw_ef": 256,
                    "recall_at_k": 0.958,
                    "p50_ms": 2.53,
                    "server_p50_ms": 0.94,
                    "disk_mb": 1234.0,
                },
            },
            "settings": {"n_vectors": 343235, "n_queries": 200, "top_k": 10},
        },
    )


@pytest.fixture
def config(tmp_path):
    """A report config pointing at a temporary folder."""
    return {
        "report": {
            "output": str(tmp_path / "docs" / "REPORT.md"),
            "figures_dir": str(tmp_path / "docs" / "figures"),
            "figure_dpi": 60,
            "themes": ["light", "dark"],
            "eval_summary": str(tmp_path / "summary.json"),
            "ragas_summary": str(tmp_path / "ragas.json"),
            "ablations_summary": str(tmp_path / "ablations.json"),
            "classifier_metrics": str(tmp_path / "classifier.json"),
            "vector_benchmark": str(tmp_path / "vector.json"),
            "datasets": ["arxiv", "hotpotqa", "multihoprag"],
            "systems": ["closed_book", "vanilla", "agentic_v2"],
            "grounded_systems": ["vanilla", "agentic_v2"],
            "null_dataset": "multihoprag",
            "labels": {
                "arxiv": "arXiv",
                "hotpotqa": "HotpotQA",
                "multihoprag": "MultiHop-RAG",
                "closed_book": "Closed-book",
                "vanilla": "Vanilla RAG",
                "agentic_v2": "Agentic (v2)",
            },
        }
    }


def test_format_number_renders_missing_as_dash():
    """A missing metric must not print as "None" in a table."""
    assert format_number(None) == "—"
    assert format_number(0.8919) == "0.892"
    assert format_number(8.555, 1) == "8.6"


def test_format_signed_always_shows_the_sign():
    """A difference reads as a direction, so the sign is never implicit."""
    assert format_signed(0.019) == "+0.019"
    assert format_signed(-0.204) == "-0.204"
    assert format_signed(0.0) == "+0.000"
    assert format_signed(None) == "—"


def test_metric_returns_none_for_anything_absent(data):
    """A partial evaluation run must not raise when the report is built."""
    assert metric(data, "arxiv", "vanilla", "relaxed_accuracy") == 0.6
    assert metric(data, "arxiv", "vanilla", "not_a_metric") is None
    assert metric(data, "arxiv", "no_such_system", "relaxed_accuracy") is None
    assert metric(data, "no_such_dataset", "vanilla", "relaxed_accuracy") is None


def test_confidence_interval_is_read_as_a_pair(data):
    """The interval comes back as (low, high) floats."""
    assert confidence_interval(data, "arxiv", "agentic_v2") == (0.8, 0.95)
    assert confidence_interval(data, "arxiv", "no_such_system") is None


def test_main_comparison_table_has_a_row_per_dataset(data, config):
    """Datasets stay separate; nothing is averaged into a single number."""
    table = main_comparison_table(data, config)
    lines = table.splitlines()
    assert len(lines) == 2 + 3  # header, separator, one row per dataset
    assert "arXiv" in table and "HotpotQA" in table and "MultiHop-RAG" in table
    assert "0.900 [0.800, 0.950]" in table  # value with its interval
    assert "Closed-book" in table and "Vanilla RAG" in table and "Agentic (v2)" in table


def test_ablation_rows_put_the_baseline_first_then_sort_by_effect(data):
    """The worst regression leads, so the table and the figure agree on order."""
    rows = ablation_rows(data)
    assert rows[0]["name"] == "full"
    assert rows[0]["difference"] is None
    assert [row["name"] for row in rows[1:]] == ["no_classifier", "no_reformulation"]


def test_significant_only_when_the_interval_excludes_zero(data):
    """An interval spanning zero is not evidence of an effect."""
    rows = {row["name"]: row for row in ablation_rows(data)}
    assert significant(rows["no_classifier"]) is True
    assert significant(rows["no_reformulation"]) is False
    assert significant(rows["full"]) is False


def test_ablation_table_marks_the_significant_row(data):
    """The one real effect is marked, and the baseline is labelled as such."""
    table = ablation_table(data)
    classifier_line = next(line for line in table.splitlines() if "no_classifier" in line)
    assert "-0.204" in classifier_line
    assert "**yes**" in classifier_line
    reformulation_line = next(line for line in table.splitlines() if "no_reformulation" in line)
    assert "| no |" in reformulation_line
    assert "(baseline)" in table


def test_groundedness_table_handles_a_missing_null_rate(data, config):
    """Only MultiHop-RAG has null questions; the others must print a dash, not a crash."""
    table = groundedness_table(data, config)
    arxiv_line = next(line for line in table.splitlines() if "arXiv" in line)
    assert "—" in arxiv_line
    multihop_line = next(
        line for line in table.splitlines() if "MultiHop-RAG" in line and "Agentic" in line
    )
    assert "0.88" in multihop_line


def test_classifier_table_includes_the_always_retrieve_baseline(data):
    """The classifier is only meaningful against the trivial policy."""
    table = classifier_table(data)
    assert "always retrieve, test" in table
    assert "0.574" in table
    assert "challenge set" in table
    assert "[0.845, 0.939]" in table


def test_vector_rows_start_with_exact_search(data):
    """Exact search is the reference row, so it leads the table."""
    rows = vector_rows(data)
    assert rows[0]["name"].endswith("(exact)")
    assert rows[0]["recall_at_k"] == 1.0
    assert len(rows) == 5


def test_vector_table_reports_client_and_server_latency(data):
    """Both columns are kept, because the client overhead was once mistaken for search time."""
    table = vector_table(data)
    assert "client p50 ms" in table and "server p50 ms" in table
    assert "343235 vectors" in table


def test_ragas_table_lists_each_judged_system(data, config):
    """Every dataset/system pair in the RAGAS summary appears once."""
    table = ragas_table(data, config)
    assert "Vanilla RAG" in table and "Agentic (v2)" in table
    assert "0.920" in table


def test_cost_table_skips_systems_that_were_not_run(data, config):
    """A system missing from a dataset is left out rather than printed empty."""
    table = cost_table(data, config)
    assert len([line for line in table.splitlines() if line.startswith("| ")]) >= 4


def test_resolve_paths_makes_every_report_path_absolute(config, tmp_path):
    """The report can be generated from any working directory."""
    relative = {
        "report": dict(config["report"], output="docs/REPORT.md", figures_dir="docs/figures")
    }
    resolved = resolve_paths(relative, tmp_path)
    assert Path(resolved["report"]["output"]).is_absolute()
    assert Path(resolved["report"]["figures_dir"]).is_absolute()
    assert relative["report"]["output"] == "docs/REPORT.md"  # the original is untouched


def test_load_report_data_names_the_missing_file(config):
    """A half-finished project gets a clear message, not a KeyError."""
    with pytest.raises(FileNotFoundError, match="eval_summary"):
        load_report_data(config)


def test_picture_tag_uses_a_relative_source_and_both_themes(tmp_path):
    """The embedded figure must resolve from the document, in either theme."""
    tag = picture_tag("main_comparison", "alt text", tmp_path / "docs" / "figures", tmp_path / "docs")
    assert 'srcset="figures/main_comparison_dark.png"' in tag
    assert 'src="figures/main_comparison_light.png"' in tag
    assert 'alt="alt text"' in tag


def test_figures_are_written_for_every_theme(data, config, tmp_path):
    """Each figure is rendered once per theme, and the files are not empty."""
    written = render_figures(data, config)
    assert set(written) == {"main_comparison", "ablation_effects", "vector_tradeoff", "groundedness"}
    for by_theme in written.values():
        assert set(by_theme) == {"light", "dark"}
        for path in by_theme.values():
            assert path.exists() and path.stat().st_size > 0


def test_a_figure_renders_when_a_metric_is_missing(data, config, tmp_path):
    """A dataset with no result for a system must not break the drawing code."""
    data.eval_summary["hotpotqa"].pop("agentic_v2")
    path = figure_main_comparison(data, config, "light", tmp_path / "partial.png")
    assert path.exists() and path.stat().st_size > 0


def test_report_document_has_every_section_and_states_the_caveat(data, config):
    """The report keeps its structure, and the arXiv caveat travels with the numbers."""
    markdown = build_report_markdown(data, config)
    for heading in (
        "## 1. Does the agentic pipeline answer better?",
        "## 2. What the agentic pipeline actually buys",
        "## 3. Which blocks earn their place",
        "## 4. Block 1: the retrieval-necessity classifier",
        "## 5. The vector store",
        "## 6. Latency and cost",
        "## 7. What this study does not show",
    ):
        assert heading in markdown
    assert "optimistic" in markdown
    assert "<picture>" in markdown
    assert "$0" in markdown
