"""Small helpers for reading and writing JSON Lines files safely."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Set, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def read_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    """Yield one dictionary per valid line of a JSONL file.

    A missing file yields nothing. A malformed line (for example a half-written
    last line after a crash) is skipped with a warning instead of stopping the run.
    """
    path = Path(path)
    if not path.is_file():
        return
    with open(path, "r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %d in %s", line_number, path)


def append_jsonl(path: PathLike, records: Iterable[Dict[str, Any]]) -> int:
    """Append records to a JSONL file (created if missing). Returns how many were written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "a", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
        jsonl_file.flush()
    return written


def load_ids(path: PathLike, key: str) -> Set[str]:
    """Return the set of values stored under ``key`` in a JSONL file."""
    return {record[key] for record in read_jsonl(path) if key in record}


def replace_file_atomically(temp_path: PathLike, final_path: PathLike) -> None:
    """Move a finished temporary file over the final path in one step."""
    os.replace(str(temp_path), str(final_path))
