"""Section 12: build the report assets (figures and `docs/REPORT.md`) from the results.

Reads only the JSON files that earlier sections wrote into ``results/``. It makes no LLM
calls, needs no quota, does not need Qdrant or Ollama running, and spends no GPU time,
so it is safe to run on battery.

    python scripts/make_report_assets.py                  # figures + the report document
    python scripts/make_report_assets.py --figures-only
    python scripts/make_report_assets.py --print-tables   # tables to stdout, write nothing
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.eval.report import (  # noqa: E402
    ablation_table,
    classifier_table,
    cost_table,
    groundedness_table,
    load_report_data,
    main_comparison_table,
    ragas_table,
    render_figures,
    resolve_paths,
    vector_table,
)
from src.eval.report_document import build_report_markdown  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description="Build the section 12 report assets.")
    parser.add_argument(
        "--figures-only", action="store_true", help="Render the figures but do not write the report."
    )
    parser.add_argument(
        "--report-only", action="store_true", help="Write the report but do not re-render figures."
    )
    parser.add_argument(
        "--print-tables", action="store_true", help="Print every table to stdout and write nothing."
    )
    return parser.parse_args()


def print_tables(data, config) -> None:
    """Print each generated table, for a quick check without opening the report."""
    tables = {
        "Main comparison": main_comparison_table(data, config),
        "Groundedness": groundedness_table(data, config),
        "RAGAS": ragas_table(data, config),
        "Ablations": ablation_table(data),
        "Classifier": classifier_table(data),
        "Vector store": vector_table(data),
        "Latency and cost": cost_table(data, config),
    }
    for title, table in tables.items():
        print(f"\n### {title}\n")
        print(table)


def main() -> None:
    """Build whichever assets were asked for and report what was written."""
    args = parse_args()
    config = resolve_paths(load_config(), PROJECT_ROOT)
    data = load_report_data(config)

    if args.print_tables:
        print_tables(data, config)
        return

    if not args.report_only:
        written = render_figures(data, config)
        for name, by_theme in written.items():
            paths = ", ".join(path.name for path in by_theme.values())
            print(f"figure {name}: {paths}")

    if not args.figures_only:
        output = Path(config["report"]["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(build_report_markdown(data, config), encoding="utf-8")
        print(f"report: {output}")


if __name__ == "__main__":
    main()
