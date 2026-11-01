"""Report assets for section 12: tables and figures built from the stored results.

Nothing in this module re-runs the pipeline, touches Qdrant or calls an LLM. Every
number is read from the JSON files that earlier sections wrote into ``results/``, so
the report can be regenerated at any time and always agrees with the measurements
it describes.

Figures are rendered twice, once for a light page and once for a dark one, from the
same validated categorical palette; the report embeds them in a ``<picture>`` element
so a reader's theme picks the right one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # no display on this machine; figures are written to disk
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

# Validated categorical palette (dataviz reference instance). Light and dark are the
# same hues stepped for their own surface, not an automatic flip. Both sets passed the
# six-check validator for the slot counts used here, in "all pairs" mode.
THEMES: Dict[str, Dict[str, Any]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary_ink": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "positive": "#2a78d6",
        "negative": "#e34948",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "secondary_ink": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "positive": "#3987e5",
        "negative": "#e66767",
    },
}

FONT_STACK = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def _use_font_stack() -> None:
    """Point matplotlib at the system sans stack (no display or serif face anywhere)."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = FONT_STACK


@dataclass
class ReportData:
    """Every results file the report needs, already parsed."""

    eval_summary: Dict[str, Any]
    ragas: Dict[str, Any]
    ablations: Dict[str, Any]
    classifier: Dict[str, Any]
    vector: Dict[str, Any]


PATH_KEYS = (
    "output",
    "figures_dir",
    "eval_summary",
    "ragas_summary",
    "ablations_summary",
    "classifier_metrics",
    "vector_benchmark",
)


def resolve_paths(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Return a copy of the config whose report paths are absolute.

    Lets the report be generated from any working directory, while the paths stay
    short and readable in ``configs/config.yaml``.
    """
    resolved = dict(config)
    report_config = dict(config["report"])
    for key in PATH_KEYS:
        report_config[key] = str(Path(project_root) / report_config[key])
    resolved["report"] = report_config
    return resolved


def load_json(path: Path) -> Dict[str, Any]:
    """Read one JSON file and return its contents as a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_report_data(config: Dict[str, Any]) -> ReportData:
    """Load every results file the report is built from.

    Raises FileNotFoundError naming the missing file, so a half-finished project
    gets a clear message instead of a KeyError further down.
    """
    report_config = config["report"]
    paths = {
        "eval_summary": Path(report_config["eval_summary"]),
        "ragas": Path(report_config["ragas_summary"]),
        "ablations": Path(report_config["ablations_summary"]),
        "classifier": Path(report_config["classifier_metrics"]),
        "vector": Path(report_config["vector_benchmark"]),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found at {path}; run the section that produces it first"
            )
    return ReportData(**{name: load_json(path) for name, path in paths.items()})


def format_number(value: Optional[float], digits: int = 3) -> str:
    """Format a metric for a markdown table, rendering a missing value as an em dash."""
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def format_signed(value: Optional[float], digits: int = 3) -> str:
    """Format a difference with an explicit sign, so a zero reads as neutral."""
    if value is None:
        return "—"
    return f"{value:+.{digits}f}"


def label_for(key: str, labels: Dict[str, str]) -> str:
    """Return the display label for a dataset or system key, falling back to the key."""
    return labels.get(key, key)


def metric(data: ReportData, dataset: str, system: str, name: str) -> Optional[float]:
    """Read one metric for one dataset/system pair from the evaluation summary.

    Returns None when the dataset, the system or the metric is absent, so a report
    can be built from a partial evaluation run.
    """
    return data.eval_summary.get(dataset, {}).get(system, {}).get(name)


def confidence_interval(
    data: ReportData, dataset: str, system: str
) -> Optional[Tuple[float, float]]:
    """Return the 95% confidence interval on relaxed accuracy, or None if absent."""
    interval = metric(data, dataset, system, "relaxed_accuracy_ci95")
    if not interval:
        return None
    return float(interval[0]), float(interval[1])


# --------------------------------------------------------------------------- tables


def main_comparison_table(data: ReportData, config: Dict[str, Any]) -> str:
    """Build the headline table: relaxed accuracy per dataset per system.

    Datasets are rows and systems are columns; nothing is averaged across datasets,
    because the three corpora measure different things.
    """
    report_config = config["report"]
    datasets: Sequence[str] = report_config["datasets"]
    systems: Sequence[str] = report_config["systems"]
    labels: Dict[str, str] = report_config["labels"]

    header = ["dataset", "n"] + [label_for(system, labels) for system in systems]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for dataset in datasets:
        count = metric(data, dataset, systems[0], "n")
        row = [label_for(dataset, labels), str(count if count is not None else "—")]
        for system in systems:
            accuracy = metric(data, dataset, system, "relaxed_accuracy")
            interval = confidence_interval(data, dataset, system)
            cell = format_number(accuracy)
            if interval is not None:
                cell += f" [{interval[0]:.3f}, {interval[1]:.3f}]"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def cost_table(data: ReportData, config: Dict[str, Any]) -> str:
    """Build the latency and call-count table that sits beside the quality numbers."""
    report_config = config["report"]
    datasets: Sequence[str] = report_config["datasets"]
    systems: Sequence[str] = report_config["systems"]
    labels: Dict[str, str] = report_config["labels"]

    header = ["dataset", "system", "model s", "wait s", "lite calls", "local tokens"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for dataset in datasets:
        for system in systems:
            if data.eval_summary.get(dataset, {}).get(system) is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        label_for(dataset, labels),
                        label_for(system, labels),
                        format_number(metric(data, dataset, system, "mean_model_seconds"), 1),
                        format_number(metric(data, dataset, system, "mean_wait_seconds"), 1),
                        format_number(metric(data, dataset, system, "mean_lite_calls"), 2),
                        format_number(metric(data, dataset, system, "mean_tokens_local"), 0),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def groundedness_table(data: ReportData, config: Dict[str, Any]) -> str:
    """Build the table of the metrics the agentic pipeline is actually meant to improve."""
    report_config = config["report"]
    datasets: Sequence[str] = report_config["datasets"]
    systems: Sequence[str] = report_config["grounded_systems"]
    labels: Dict[str, str] = report_config["labels"]

    header = [
        "dataset",
        "system",
        "unsupported claims",
        "contradicted claims",
        "null abstention",
        "false abstention",
        "skip rate",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for dataset in datasets:
        for system in systems:
            if data.eval_summary.get(dataset, {}).get(system) is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        label_for(dataset, labels),
                        label_for(system, labels),
                        format_number(metric(data, dataset, system, "unsupported_claim_rate"), 2),
                        format_number(metric(data, dataset, system, "contradicted_claim_rate"), 2),
                        format_number(metric(data, dataset, system, "null_abstention_rate"), 2),
                        format_number(metric(data, dataset, system, "false_abstention_rate"), 2),
                        format_number(metric(data, dataset, system, "skip_rate"), 2),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def ablation_rows(data: ReportData) -> List[Dict[str, Any]]:
    """Return one row per ablation configuration, baseline first then worst to best.

    The paired difference against the full system is the honest comparison, because
    every configuration answered exactly the same questions.
    """
    rows = []
    for name, summary in data.ablations.items():
        paired = summary.get("paired_vs_full") or {}
        rows.append(
            {
                "name": name,
                "description": summary.get("description", ""),
                "n": summary.get("n"),
                "relaxed_accuracy": summary.get("relaxed_accuracy"),
                "difference": paired.get("difference"),
                "ci95": paired.get("ci95"),
                "wins": paired.get("wins"),
                "losses": paired.get("losses"),
                "mean_model_seconds": summary.get("mean_model_seconds"),
                "mean_lite_calls": summary.get("mean_lite_calls"),
            }
        )
    baseline = [row for row in rows if row["difference"] is None]
    compared = sorted(
        (row for row in rows if row["difference"] is not None),
        key=lambda row: row["difference"],
    )
    return baseline + compared


def significant(row: Dict[str, Any]) -> bool:
    """True when a configuration's paired confidence interval excludes zero."""
    interval = row.get("ci95")
    if not interval:
        return False
    return interval[0] > 0.0 or interval[1] < 0.0


def ablation_table(data: ReportData) -> str:
    """Build the ablation table, marking the configurations whose effect is real."""
    header = [
        "configuration",
        "n",
        "relaxed acc",
        "vs full",
        "95% CI",
        "significant",
        "model s",
        "lite calls",
        "what changed",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in ablation_rows(data):
        interval = row.get("ci95")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['name']}`",
                    str(row["n"] if row["n"] is not None else "—"),
                    format_number(row["relaxed_accuracy"]),
                    "(baseline)" if row["difference"] is None else format_signed(row["difference"]),
                    "—" if not interval else f"[{interval[0]:+.3f}, {interval[1]:+.3f}]",
                    "—" if row["difference"] is None else ("**yes**" if significant(row) else "no"),
                    format_number(row["mean_model_seconds"], 1),
                    format_number(row["mean_lite_calls"], 2),
                    row["description"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def classifier_table(data: ReportData) -> str:
    """Build the block 1 table: the classifier against the always-retrieve baseline."""
    test = data.classifier.get("test", {}).get("metrics", {})
    intervals = data.classifier.get("test", {}).get("confidence_intervals_95", {})
    challenge = data.classifier.get("challenge", {}).get("metrics", {})
    baseline = data.classifier.get("baseline_always_retrieve_test", {})

    accuracy_interval = intervals.get("accuracy", {})
    interval_text = (
        f" [{accuracy_interval['low']:.3f}, {accuracy_interval['high']:.3f}]"
        if accuracy_interval
        else ""
    )

    header = ["split / system", "accuracy", "macro F1", "retrieve recall", "skip rate", "n"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.append(
        "| "
        + " | ".join(
            [
                "classifier, test",
                format_number(test.get("accuracy")) + interval_text,
                format_number(test.get("macro_f1")),
                format_number(test.get("retrieve_recall")),
                format_number(test.get("skip_rate"), 2),
                str(test.get("count", "—")),
            ]
        )
        + " |"
    )
    lines.append(
        "| "
        + " | ".join(
            [
                "always retrieve, test",
                format_number(baseline.get("accuracy")),
                format_number(baseline.get("macro_f1")),
                format_number(baseline.get("retrieve_recall")),
                format_number(baseline.get("skip_rate"), 2),
                str(test.get("count", "—")),
            ]
        )
        + " |"
    )
    lines.append(
        "| "
        + " | ".join(
            [
                "classifier, challenge set",
                format_number(challenge.get("accuracy")),
                format_number(challenge.get("macro_f1")),
                format_number(challenge.get("retrieve_recall")),
                format_number(challenge.get("skip_rate"), 2),
                str(challenge.get("count", "—")),
            ]
        )
        + " |"
    )
    return "\n".join(lines)


def vector_rows(data: ReportData) -> List[Dict[str, Any]]:
    """Return one row per vector-store configuration, exact search first."""
    rows: List[Dict[str, Any]] = []
    exact = data.vector.get("exact")
    if exact:
        rows.append(
            {
                "name": f"{exact.get('engine', 'exact')} (exact)",
                "quantization": "none",
                "hnsw_ef": None,
                "recall_at_k": exact.get("recall_at_k"),
                "p50_ms": exact.get("p50_ms"),
                "server_p50_ms": None,
                "disk_mb": None,
            }
        )
    for name, entry in (data.vector.get("approximate") or {}).items():
        rows.append(
            {
                "name": name,
                "quantization": entry.get("quantization"),
                "hnsw_ef": entry.get("hnsw_ef"),
                "recall_at_k": entry.get("recall_at_k"),
                "p50_ms": entry.get("p50_ms"),
                "server_p50_ms": entry.get("server_p50_ms"),
                "disk_mb": entry.get("disk_mb"),
            }
        )
    return rows


def vector_table(data: ReportData) -> str:
    """Build the FAISS-versus-HNSW table, including the server-side search time."""
    settings = data.vector.get("settings", {})
    header = ["configuration", "recall@k", "client p50 ms", "server p50 ms", "disk MB"]
    lines = []
    if settings:
        lines.append(
            f"{settings.get('n_vectors', '?')} vectors, {settings.get('n_queries', '?')} queries, "
            f"top-k {settings.get('top_k', '?')}; recall is measured against exact search.\n"
        )
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in vector_rows(data):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['name']}`",
                    format_number(row["recall_at_k"]),
                    format_number(row["p50_ms"], 2),
                    format_number(row["server_p50_ms"], 2),
                    format_number(row["disk_mb"], 0),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def ragas_table(data: ReportData, config: Dict[str, Any]) -> str:
    """Build the RAGAS table (faithfulness and answer relevancy on a judged sample)."""
    labels: Dict[str, str] = config["report"]["labels"]
    header = ["dataset", "system", "n judged", "faithfulness", "answer relevancy"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for dataset, systems in data.ragas.items():
        for system, scores in systems.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        label_for(dataset, labels),
                        label_for(system, labels),
                        str(scores.get("n", "—")),
                        format_number(scores.get("faithfulness")),
                        format_number(scores.get("answer_relevancy")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


# -------------------------------------------------------------------------- figures


def _style_axes(axes: Any, theme: Dict[str, Any], y_grid: bool = True) -> None:
    """Make the axes recessive: hairline solid grid, no top/right spines, muted ticks."""
    axes.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(theme["axis"])
        axes.spines[side].set_linewidth(0.8)
    axes.tick_params(colors=theme["muted"], labelsize=9, length=3, width=0.8)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_color(theme["secondary_ink"])
    if y_grid:
        axes.grid(axis="y", color=theme["grid"], linewidth=0.8, linestyle="-")
        axes.set_axisbelow(True)


def _new_figure(theme: Dict[str, Any], size: Tuple[float, float]) -> Tuple[Any, Any]:
    """Create a figure and one axes on the theme's surface, in the system sans."""
    _use_font_stack()
    figure, axes = plt.subplots(figsize=size)
    figure.patch.set_facecolor(theme["surface"])
    _style_axes(axes, theme)
    return figure, axes


def _title(axes: Any, theme: Dict[str, Any], title: str, subtitle: str = "") -> None:
    """Put a left-aligned title, and an optional quieter subtitle, above the plot."""
    axes.set_title(title, color=theme["ink"], fontsize=12, fontweight="bold", loc="left", pad=18 if subtitle else 10)
    if subtitle:
        axes.text(
            0.0,
            1.02,
            subtitle,
            transform=axes.transAxes,
            color=theme["secondary_ink"],
            fontsize=9,
            va="bottom",
            ha="left",
        )


def _legend(axes: Any, theme: Dict[str, Any], **kwargs: Any) -> None:
    """Add a legend with no frame; identity is never carried by color alone."""
    legend = axes.legend(frameon=False, fontsize=9, **kwargs)
    for text in legend.get_texts():
        text.set_color(theme["secondary_ink"])


def _save(figure: Any, path: Path, theme: Dict[str, Any], dpi: int) -> Path:
    """Write the figure to disk, creating the folder, and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(figure)
    return path


def figure_main_comparison(
    data: ReportData, config: Dict[str, Any], theme_name: str, path: Path
) -> Path:
    """Grouped bars: relaxed accuracy per dataset for each system, with 95% CI whiskers.

    Every bar carries its value as a direct label, which is also what the palette's
    light-mode contrast warning requires (a label, not color alone, carries the read).
    """
    theme = THEMES[theme_name]
    report_config = config["report"]
    datasets: Sequence[str] = report_config["datasets"]
    systems: Sequence[str] = report_config["systems"]
    labels: Dict[str, str] = report_config["labels"]

    figure, axes = _new_figure(theme, (7.6, 4.2))
    group_width = 0.76
    bar_width = group_width / len(systems)

    for index, system in enumerate(systems):
        positions = []
        values = []
        errors_low = []
        errors_high = []
        label_heights = []
        for group, dataset in enumerate(datasets):
            accuracy = metric(data, dataset, system, "relaxed_accuracy") or 0.0
            interval = confidence_interval(data, dataset, system)
            # A 2px-equivalent gap between neighbouring bars, rather than a border.
            offset = -group_width / 2 + index * bar_width + bar_width / 2
            positions.append(group + offset)
            values.append(accuracy)
            errors_low.append(max(0.0, accuracy - interval[0]) if interval else 0.0)
            errors_high.append(max(0.0, interval[1] - accuracy) if interval else 0.0)
            # The label clears the whisker cap, never the bar top, or the error bar
            # would strike through it.
            label_heights.append((interval[1] if interval else accuracy) + 0.035)
        bars = axes.bar(
            positions,
            values,
            width=bar_width * 0.88,
            label=label_for(system, labels),
            color=theme["series"][index % len(theme["series"])],
            zorder=3,
        )
        axes.errorbar(
            positions,
            values,
            yerr=[errors_low, errors_high],
            fmt="none",
            ecolor=theme["secondary_ink"],
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            zorder=4,
        )
        for bar, value, height in zip(bars, values, label_heights):
            axes.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=theme["secondary_ink"],
                zorder=5,
            )

    axes.set_xticks(range(len(datasets)))
    axes.set_xticklabels([label_for(dataset, labels) for dataset in datasets], fontsize=10)
    axes.set_ylim(0, 1.12)
    axes.set_ylabel("relaxed accuracy", color=theme["secondary_ink"], fontsize=9)
    _title(
        axes,
        theme,
        "Answer accuracy by dataset",
        "whiskers are 95% confidence intervals; the arXiv column is optimistic (see the caveat)",
    )
    _legend(axes, theme, loc="upper right", ncol=len(systems))
    return _save(figure, path, theme, report_config["figure_dpi"])


def figure_ablation_effects(
    data: ReportData, config: Dict[str, Any], theme_name: str, path: Path
) -> Path:
    """Dot plot of each ablation's paired effect against the full system, with CI bars.

    The sign is carried by position relative to the zero rule and by the printed
    value, so the diverging colour is a second channel rather than the only one.
    """
    theme = THEMES[theme_name]
    report_config = config["report"]
    rows = [row for row in ablation_rows(data) if row["difference"] is not None]

    figure, axes = _new_figure(theme, (7.8, 0.42 * len(rows) + 1.6))
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", color=theme["grid"], linewidth=0.8, linestyle="-")
    axes.set_axisbelow(True)

    for index, row in enumerate(rows):
        colour = theme["negative"] if row["difference"] < 0 else theme["positive"]
        interval = row.get("ci95")
        if interval:
            axes.plot(
                [interval[0], interval[1]],
                [index, index],
                color=colour,
                linewidth=2.0,
                solid_capstyle="round",
                zorder=3,
            )
        axes.plot(
            [row["difference"]],
            [index],
            marker="o",
            markersize=9,
            color=colour,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.5,
            zorder=4,
        )
        axes.text(
            0.995,
            index,
            f"{row['difference']:+.3f}" + ("  *" if significant(row) else ""),
            transform=axes.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8.5,
            color=theme["secondary_ink"],
            zorder=5,
        )

    axes.axvline(0.0, color=theme["axis"], linewidth=1.0, zorder=2)
    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels([row["name"] for row in rows], fontsize=9)
    axes.invert_yaxis()
    axes.set_xlim(-0.34, 0.14)
    axes.set_xlabel(
        "change in relaxed accuracy vs the full system", color=theme["secondary_ink"], fontsize=9
    )
    _title(
        axes,
        theme,
        "What each block is worth",
        "paired on the same 103 arXiv questions; * marks a 95% CI that excludes zero",
    )
    return _save(figure, path, theme, report_config["figure_dpi"])


def figure_vector_tradeoff(
    data: ReportData, config: Dict[str, Any], theme_name: str, path: Path
) -> Path:
    """Recall against latency for exact search and for HNSW at several ef values."""
    theme = THEMES[theme_name]
    report_config = config["report"]

    figure, axes = _new_figure(theme, (7.0, 4.4))
    axes.grid(axis="x", color=theme["grid"], linewidth=0.8, linestyle="-")

    series: Dict[str, List[Dict[str, Any]]] = {"fp32": [], "int8": []}
    for row in vector_rows(data):
        if row["hnsw_ef"] is None:
            continue
        key = "int8" if row["quantization"] not in (None, "none") else "fp32"
        series[key].append(row)

    display = {"fp32": "Qdrant HNSW, fp32", "int8": "Qdrant HNSW, int8"}
    for index, (key, rows) in enumerate(series.items()):
        # Ordered by the swept parameter, not by latency: ef is what the curve traces,
        # and sorting by latency would reorder the two int8 points whose timings are
        # within noise of each other and draw a kink that means nothing.
        rows = sorted(rows, key=lambda row: row["hnsw_ef"])
        if not rows:
            continue
        colour = theme["series"][index]
        axes.plot(
            [row["p50_ms"] for row in rows],
            [row["recall_at_k"] for row in rows],
            marker="o",
            markersize=8,
            linewidth=2.0,
            color=colour,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.2,
            label=display[key],
            zorder=3,
        )
        # Only the ends of each curve are labelled: a label on every point collides
        # with the curves and goes unread. The ef values between them are in the table.
        for row, offset, align in ((rows[0], (0, -16), "center"), (rows[-1], (8, 2), "left")):
            axes.annotate(
                f"ef={row['hnsw_ef']}",
                (row["p50_ms"], row["recall_at_k"]),
                textcoords="offset points",
                xytext=offset,
                ha=align,
                fontsize=8,
                color=theme["muted"],
            )

    exact = data.vector.get("exact")
    if exact:
        axes.plot(
            [exact["p50_ms"]],
            [exact["recall_at_k"]],
            marker="D",
            markersize=9,
            color=theme["series"][2],
            markeredgecolor=theme["surface"],
            markeredgewidth=1.2,
            linestyle="none",
            label="FAISS flat, exact",
            zorder=3,
        )
        axes.annotate(
            "exact",
            (exact["p50_ms"], exact["recall_at_k"]),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            fontsize=8,
            color=theme["muted"],
        )

    settings = data.vector.get("settings", {})
    # Log latency: the HNSW points sit inside a 2 ms band while exact search is ten times
    # slower, so a linear axis squeezes every approximate configuration into one corner.
    axes.set_xscale("log")
    axes.set_xticks([2, 3, 5, 10, 20])
    axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes.set_xlim(1.6, 30)
    axes.set_xlabel(
        "median search latency, ms (client side, log scale)",
        color=theme["secondary_ink"],
        fontsize=9,
    )
    axes.set_ylabel("recall@10 vs exact search", color=theme["secondary_ink"], fontsize=9)
    axes.set_ylim(0.82, 1.03)
    ef_values = sorted({row["hnsw_ef"] for row in vector_rows(data) if row["hnsw_ef"]})
    _title(
        axes,
        theme,
        "Recall against latency on the scale corpus",
        f"{settings.get('n_vectors', 0):,} chunks, {settings.get('n_queries', '?')} queries; "
        f"points along each curve are ef = {', '.join(str(value) for value in ef_values)}"
        if settings
        else "",
    )
    _legend(axes, theme, loc="lower right")
    return _save(figure, path, theme, report_config["figure_dpi"])


def figure_groundedness(
    data: ReportData, config: Dict[str, Any], theme_name: str, path: Path
) -> Path:
    """Two panels on one rate scale: unsupported claims, and abstention on null questions.

    Both panels show the same two systems in the same two colours, so the figure reads
    as small multiples rather than two unrelated charts. No second y-axis is used.
    """
    theme = THEMES[theme_name]
    report_config = config["report"]
    datasets: Sequence[str] = report_config["datasets"]
    systems: Sequence[str] = report_config["grounded_systems"]
    labels: Dict[str, str] = report_config["labels"]
    null_dataset: str = report_config["null_dataset"]

    _use_font_stack()
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(9.2, 4.2), gridspec_kw={"width_ratios": [3, 1.15]}
    )
    figure.patch.set_facecolor(theme["surface"])
    for axes in (left, right):
        _style_axes(axes, theme)
        axes.set_ylim(0, 1.0)

    def draw(
        axes: Any,
        groups: Sequence[str],
        values_for: Any,
        show_labels: bool,
        group_width: float = 0.62,
    ) -> None:
        """Draw one panel of grouped bars with direct value labels."""
        bar_width = group_width / len(systems)
        for index, system in enumerate(systems):
            positions = []
            values = []
            for group, key in enumerate(groups):
                value = values_for(key, system)
                offset = -group_width / 2 + index * bar_width + bar_width / 2
                positions.append(group + offset)
                values.append(0.0 if value is None else value)
            bars = axes.bar(
                positions,
                values,
                width=bar_width * 0.86,
                color=theme["series"][index],
                label=label_for(system, labels) if show_labels else None,
                zorder=3,
            )
            for bar, value in zip(bars, values):
                axes.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.03,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=theme["secondary_ink"],
                    zorder=5,
                )
        axes.set_xticks(range(len(groups)))
        axes.set_xticklabels([label_for(key, labels) for key in groups], fontsize=9.5)

    draw(
        left,
        datasets,
        lambda dataset, system: metric(data, dataset, system, "unsupported_claim_rate"),
        show_labels=True,
    )
    left.set_ylabel("rate", color=theme["secondary_ink"], fontsize=9)
    _title(left, theme, "Unsupported claims (lower is better)", "share of answer claims the NLI verifier cannot support")
    _legend(left, theme, loc="upper left")

    draw(
        right,
        [null_dataset],
        lambda dataset, system: metric(data, dataset, system, "null_abstention_rate"),
        show_labels=False,
        group_width=0.42,  # one group only, so narrower bars keep the panel in proportion
    )
    right.set_xlim(-0.5, 0.5)
    _title(right, theme, "Abstention when the answer is absent", "higher is better")

    figure.tight_layout()
    return _save(figure, path, theme, report_config["figure_dpi"])


FIGURES = {
    "main_comparison": figure_main_comparison,
    "ablation_effects": figure_ablation_effects,
    "vector_tradeoff": figure_vector_tradeoff,
    "groundedness": figure_groundedness,
}


def render_figures(data: ReportData, config: Dict[str, Any]) -> Dict[str, Dict[str, Path]]:
    """Render every figure once per theme and return the written paths by name and theme."""
    report_config = config["report"]
    figures_dir = Path(report_config["figures_dir"])
    written: Dict[str, Dict[str, Path]] = {}
    for name, builder in FIGURES.items():
        written[name] = {}
        for theme_name in report_config["themes"]:
            path = figures_dir / f"{name}_{theme_name}.png"
            written[name][theme_name] = builder(data, config, theme_name, path)
    return written


def picture_tag(name: str, alt: str, figures_dir: Path, document_dir: Path) -> str:
    """Build a <picture> element that serves the dark figure to a dark-themed reader."""
    try:
        relative = figures_dir.relative_to(document_dir).as_posix()
    except ValueError:
        relative = figures_dir.as_posix()
    return (
        "<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" srcset="{relative}/{name}_dark.png">\n'
        f'  <img alt="{alt}" src="{relative}/{name}_light.png" width="820">\n'
        "</picture>"
    )
