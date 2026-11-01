"""Service checks and small formatting helpers for the demo app.

Kept free of Streamlit so the logic can be tested without a browser or a running
service. Every check returns a `ServiceStatus` rather than raising, because the app
should explain what is missing instead of showing a stack trace.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class ServiceStatus:
    """Whether one dependency is usable, and what to do when it is not."""

    name: str
    ok: bool
    detail: str
    remedy: str = ""


def check_qdrant(config: Dict[str, Any], timeout: float = 2.0) -> ServiceStatus:
    """Check that the Qdrant server answers and holds the pipeline's collection."""
    vector_config = config["vector_store"]
    collection = config["graph"]["collection"]
    url = vector_config["url"].rstrip("/")
    remedy = "Start it with: powershell -File scripts/start_qdrant.ps1"
    try:
        response = requests.get(f"{url}/collections", timeout=timeout)
        response.raise_for_status()
        names = [item["name"] for item in response.json()["result"]["collections"]]
    except Exception as error:  # noqa: BLE001 - any failure means "not usable"
        return ServiceStatus("Qdrant", False, f"no answer at {url} ({type(error).__name__})", remedy)
    if collection not in names:
        return ServiceStatus(
            "Qdrant",
            False,
            f"running, but the collection '{collection}' is missing",
            "Build it with: python scripts/build_index.py",
        )
    return ServiceStatus("Qdrant", True, f"running at {url}, collection '{collection}' present")


def check_ollama(config: Dict[str, Any], timeout: float = 2.0) -> ServiceStatus:
    """Check that Ollama answers and has the generator model pulled."""
    model_config = config["llm"]["models"]["ollama_qwen"]
    base_url = model_config["base_url"].rstrip("/")
    model_id = model_config["model_id"]
    remedy = f"Start Ollama, then: ollama pull {model_id}"
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        response.raise_for_status()
        names = [item["name"] for item in response.json().get("models", [])]
    except Exception as error:  # noqa: BLE001
        return ServiceStatus("Ollama", False, f"no answer at {base_url} ({type(error).__name__})", remedy)
    if not any(name.startswith(model_id.split(":")[0]) for name in names):
        return ServiceStatus("Ollama", False, f"running, but '{model_id}' is not pulled", remedy)
    return ServiceStatus("Ollama", True, f"running at {base_url}, '{model_id}' available")


def check_gemini_key(environment: Optional[Dict[str, str]] = None) -> ServiceStatus:
    """Check that a Gemini API key is present (its validity is only known on first use)."""
    environment = environment if environment is not None else os.environ
    key = environment.get("GOOGLE_API_KEY", "").strip()
    if not key:
        return ServiceStatus(
            "Gemini key",
            False,
            "GOOGLE_API_KEY is not set",
            "Copy .env.example to .env and put a free AI Studio key in it",
        )
    return ServiceStatus("Gemini key", True, f"set (ends with ...{key[-4:]})")


def check_all(config: Dict[str, Any]) -> List[ServiceStatus]:
    """Run every dependency check, in the order the pipeline needs them."""
    return [check_qdrant(config), check_ollama(config), check_gemini_key()]


def blocking(statuses: List[ServiceStatus]) -> List[ServiceStatus]:
    """Return the failed checks that stop a question from being answered at all."""
    return [status for status in statuses if not status.ok]


def count_lite_calls(trace: List[Dict[str, Any]]) -> int:
    """Count the Gemini lite calls in a trace (one per reformulator or critic step)."""
    return sum(
        1 for entry in trace if entry.get("step") in ("reformulate", "critique") and entry.get("model")
    )


def retrieval_rounds(trace: List[Dict[str, Any]]) -> int:
    """Count how many times the pipeline searched the index."""
    return sum(1 for entry in trace if entry.get("step") == "retrieve")


def step_of(trace: List[Dict[str, Any]], step: str) -> Optional[Dict[str, Any]]:
    """Return the first trace entry for a step, or None when the step never ran."""
    return next((entry for entry in trace if entry.get("step") == step), None)


def verdict_sequence(trace: List[Dict[str, Any]]) -> List[str]:
    """Every critic verdict in order, so the retry loop can be summarised in one line."""
    return [entry["verdict"] for entry in trace if entry.get("step") == "critique"]


def summarise(state: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a finished pipeline state to the handful of numbers the app shows."""
    trace = state.get("trace", [])
    classify = step_of(trace, "classify")
    return {
        "decision": (classify or {}).get("decision", "retrieve"),
        "confidence": (classify or {}).get("confidence"),
        "retries": state.get("retries", 0),
        "retrieval_rounds": retrieval_rounds(trace),
        "lite_calls": count_lite_calls(trace),
        "total_seconds": state.get("total_seconds"),
        "verdicts": verdict_sequence(trace),
        "stop_reason": state.get("stop_reason"),
        "chunks": len(state.get("contexts", [])),
    }


CLAIM_ICONS = {"supported": "✅", "unsupported": "⚠️", "contradicted": "❌"}


def claim_icon(label: str) -> str:
    """An icon for a verified claim's label; identity never rests on colour alone."""
    return CLAIM_ICONS.get(label, "•")
