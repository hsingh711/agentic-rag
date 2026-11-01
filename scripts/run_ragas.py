"""Section 10: RAGAS on a sample of the evaluation answers (judge = Gemini lite via role ragas_judge).

Run after scripts/run_eval.py has produced the answers. Resumable; stops cleanly at the daily limit.

    python scripts/run_ragas.py --limit 2          # pilot
    python scripts/run_ragas.py
    python scripts/run_ragas.py --summary-only
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.eval.harness import DATASETS, results_dir  # noqa: E402
from src.eval.ragas_eval import run_ragas, summarize_ragas  # noqa: E402
from src.llm import DailyLimitReached  # noqa: E402


def main() -> None:
    """Score the sampled answers and print the per-dataset means."""
    parser = argparse.ArgumentParser(description="Run RAGAS on a sample of the evaluation answers.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--limit", type=int, default=None, help="At most this many NEW samples per dataset and system.")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config()
    if not args.summary_only:
        try:
            for dataset in args.datasets:
                for system in config["ragas"]["systems"]:
                    new = run_ragas(config, dataset, system, args.limit)
                    print(f"{dataset} / {system}: {new} new samples scored")
        except DailyLimitReached as error:
            print(f"\nStopped: {error}\nProgress is saved; re-run after midnight Pacific Time to continue.")

    summary = summarize_ragas(config)
    for dataset, systems in summary.items():
        print(f"\n== {dataset}")
        for system, values in systems.items():
            metrics = "  ".join(
                f"{name} {values[name]:.3f} (n={values[name + '_n']})" for name in config["ragas"]["metrics"] if values[name] is not None
            )
            print(f"  {system:10s} samples {values['n']:3d}  {metrics}")
    path = results_dir(config) / "ragas_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
