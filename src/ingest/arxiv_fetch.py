"""Fetch paper metadata and abstracts from the arXiv API.

Two modes:
  * ``fetch_papers``  - small development corpus (newest papers, ~500 abstracts).
  * ``fetch_scale``   - large corpus, fetched in date windows with a state file so it
                        can be interrupted and resumed.

Both modes append to a JSONL file as they go and skip papers already saved, so
re-running never duplicates data.
"""

import json
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Set

import arxiv

from src.ingest.io_utils import PathLike, append_jsonl, load_ids

logger = logging.getLogger(__name__)

SAVE_EVERY = 100  # write to disk after this many new papers, so a crash loses little


def build_category_query(categories: List[str]) -> str:
    """Build an arXiv query such as ``cat:cs.CL OR cat:cs.IR OR cat:cs.LG``."""
    return " OR ".join(f"cat:{category}" for category in categories)


def clean_text(text: str) -> str:
    """Collapse all runs of whitespace (including newlines) into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def strip_version(short_id: str) -> str:
    """Remove a trailing version suffix: ``2401.00001v2`` -> ``2401.00001``."""
    return re.sub(r"v\d+$", "", short_id)


def paper_from_result(result: arxiv.Result) -> Dict[str, Any]:
    """Convert an ``arxiv.Result`` into the plain dictionary we store on disk."""
    return {
        "arxiv_id": strip_version(result.get_short_id()),
        "title": clean_text(result.title),
        "authors": [author.name for author in result.authors],
        "abstract": clean_text(result.summary),
        "published": result.published.isoformat(),
        "pdf_url": result.pdf_url,
        "categories": list(result.categories),
    }


def make_client(page_size: int, sleep_seconds: float) -> arxiv.Client:
    """Create an arXiv client that sleeps between requests and retries on failures."""
    return arxiv.Client(page_size=page_size, delay_seconds=sleep_seconds, num_retries=5)


def save_results(
    results,
    papers_path: PathLike,
    seen_ids: Set[str],
    limit: int,
) -> int:
    """Save new papers from an iterator of arXiv results, stopping after ``limit`` new ones.

    Papers are written in small batches. ``seen_ids`` is updated in place.

    Returns:
        The number of new papers saved.
    """
    saved = 0
    pending: List[Dict[str, Any]] = []
    try:
        for result in results:
            paper = paper_from_result(result)
            if paper["arxiv_id"] in seen_ids:
                continue
            seen_ids.add(paper["arxiv_id"])
            pending.append(paper)
            saved += 1
            if len(pending) >= SAVE_EVERY:
                append_jsonl(papers_path, pending)
                pending = []
            if saved >= limit:
                break
    finally:
        # Always flush what we have, even if the request loop raised an error.
        if pending:
            append_jsonl(papers_path, pending)
    return saved


def fetch_papers(config: Dict[str, Any], papers_path: PathLike, max_results: int) -> int:
    """Fetch up to ``max_results`` newest papers (abstracts only) into ``papers_path``.

    Resumes if the file exists: papers already saved are kept and counted, and the
    search continues from that position. If new papers were submitted in the meantime
    the search may return a few already-saved papers; they are skipped, so the file can
    end slightly below ``max_results`` - simply run again to top it up.

    Returns:
        The number of new papers saved in this call.
    """
    ingest_config = config["ingest"]
    seen_ids = load_ids(papers_path, "arxiv_id")
    already_saved = len(seen_ids)
    if already_saved >= max_results:
        logger.info("Already have %d papers (target %d); nothing to fetch.", already_saved, max_results)
        return 0

    client = make_client(ingest_config["page_size"], ingest_config["request_sleep_seconds"])
    search = arxiv.Search(
        query=build_category_query(ingest_config["categories"]),
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    logger.info("Fetching papers %d..%d for categories %s", already_saved, max_results, ingest_config["categories"])
    results = client.results(search, offset=already_saved)
    saved = save_results(results, papers_path, seen_ids, limit=max_results - already_saved)
    logger.info("Saved %d new papers (total now %d).", saved, len(seen_ids))
    return saved


# ---------------------------------------------------------------------------
# Large corpus (--scale)
# ---------------------------------------------------------------------------


def make_windows(earliest: date, window_days: int, today: date) -> List[Dict[str, str]]:
    """List date windows from ``today`` back to ``earliest``, newest first.

    Each window is ``{"name": "20260901-20260907", "start": "202609010000", "end": "202609072359"}``
    where start/end use the ``submittedDate`` format of the arXiv API.
    """
    windows: List[Dict[str, str]] = []
    window_end = today
    while window_end >= earliest:
        window_start = max(window_end - timedelta(days=window_days - 1), earliest)
        windows.append(
            {
                "name": f"{window_start:%Y%m%d}-{window_end:%Y%m%d}",
                "start": f"{window_start:%Y%m%d}0000",
                "end": f"{window_end:%Y%m%d}2359",
            }
        )
        window_end = window_start - timedelta(days=1)
    return windows


def load_completed_windows(state_path: PathLike) -> Set[str]:
    """Read the names of windows that were fully fetched in earlier runs."""
    state_path = Path(state_path)
    if not state_path.is_file():
        return set()
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("State file %s is unreadable; starting windows from scratch.", state_path)
        return set()
    return set(state.get("completed_windows", []))


def save_completed_windows(state_path: PathLike, completed: Set[str]) -> None:
    """Write the completed-window list to the state file."""
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"completed_windows": sorted(completed)}, indent=2), encoding="utf-8")


def fetch_scale(config: Dict[str, Any], papers_path: PathLike, state_path: PathLike) -> int:
    """Build the large abstract corpus window by window, newest first.

    Each window (default 7 days) is one arXiv query. After a window finishes it is
    recorded in ``state_path``. If the run is interrupted, the next run skips finished
    windows and redoes only the unfinished one (already-saved papers are skipped).

    Stops when ``scale.target_abstracts`` papers are saved or windows run out.

    Returns:
        The number of new papers saved in this call.
    """
    ingest_config = config["ingest"]
    scale_config = config["scale"]
    target = scale_config["target_abstracts"]

    seen_ids = load_ids(papers_path, "arxiv_id")
    completed = load_completed_windows(state_path)
    category_query = build_category_query(ingest_config["categories"])
    client = make_client(scale_config["page_size"], ingest_config["request_sleep_seconds"])

    earliest = datetime.strptime(str(scale_config["earliest_date"]), "%Y-%m-%d").date()
    windows = make_windows(earliest, scale_config["window_days"], date.today())

    total_saved = 0
    for window in windows:
        if len(seen_ids) >= target:
            break
        if window["name"] in completed:
            continue

        query = f"({category_query}) AND submittedDate:[{window['start']} TO {window['end']}]"
        search = arxiv.Search(
            query=query,
            max_results=None,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        logger.info("Window %s (have %d / %d papers)", window["name"], len(seen_ids), target)
        saved = save_results(client.results(search), papers_path, seen_ids, limit=target - len(seen_ids))
        total_saved += saved

        # Only mark the window finished if it was not cut short by reaching the target.
        if len(seen_ids) < target:
            completed.add(window["name"])
            save_completed_windows(state_path, completed)

    logger.info("Scale fetch done: %d new papers, %d total.", total_saved, len(seen_ids))
    return total_saved
