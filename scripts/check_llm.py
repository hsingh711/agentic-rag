"""Ping every configured model once with a tiny prompt and report which ones work.

Each ping counts as one request against that model's daily quota (the shared counter in
data/gemini_usage.json is updated), so run this sparingly.

Run from the project root:
    python scripts/check_llm.py
    python scripts/check_llm.py --models gemma_26b flash_25
"""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.llm import LLMClient  # noqa: E402

PING_PROMPT = "Reply with the single word: pong"


def main() -> None:
    """Ping the chosen models (default: all) and print one line per model."""
    parser = argparse.ArgumentParser(description="Ping configured LLMs once each.")
    parser.add_argument("--models", nargs="*", default=None, help="Model keys to ping (default: all).")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config()
    model_keys = args.models or list(config["llm"]["models"])

    failures = []
    for model_key in model_keys:
        # A temporary one-model role lets the normal wrapper (limits, counter, log) do the call.
        ping_config = copy.deepcopy(config)
        ping_config["llm"]["roles"] = {"ping": model_key}
        client = LLMClient(ping_config)
        try:
            result = client.call(PING_PROMPT, role="ping")
            reply = result.text.strip().replace("\n", " ")[:60]
            print(
                f"OK    {model_key:15s} {result.model_id:26s} "
                f"{result.latency_seconds:5.2f}s  tokens {result.input_tokens}+{result.output_tokens}  reply: {reply!r}"
            )
        except Exception as error:  # report every kind of failure per model, do not stop the loop
            failures.append(model_key)
            print(f"FAIL  {model_key:15s} {type(error).__name__}: {str(error)[:200]}")

    print(f"\n{len(model_keys) - len(failures)} of {len(model_keys)} models OK.")
    if failures:
        print("Failed:", ", ".join(failures))


if __name__ == "__main__":
    main()
