"""Tests for the methods bibliography that feeds the papers workbook.

The bibliography is hand-curated data rather than generated output, so these tests
guard the things a human editing a JSON file gets wrong: a missing field, a duplicate
entry, an area that no longer groups, or an arXiv id that disagrees with its own
reference string. They read the real data file, because that file is the artefact.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT

BIBLIOGRAPHY_PATH = PROJECT_ROOT / "data" / "methods_bibliography.json"
EXPORT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "export_papers.py"

REQUIRED_FIELDS = ("area", "title", "authors", "reference", "used_for", "source")
KNOWN_SOURCES = ("arXiv API", "hand-entered")


def load_export_module():
    """Import scripts/export_papers.py, which is a script rather than a package module."""
    spec = importlib.util.spec_from_file_location("export_papers", EXPORT_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bibliography():
    """The bibliography as stored on disk."""
    return json.loads(BIBLIOGRAPHY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def entries(bibliography):
    """Just the entry list."""
    return bibliography["entries"]


def test_bibliography_file_exists():
    """The workbook export depends on this file, so its absence is a failure."""
    assert BIBLIOGRAPHY_PATH.exists()


def test_has_entries_and_provenance(bibliography):
    """The file carries entries and says where its metadata came from."""
    assert bibliography["entries"]
    assert bibliography["arxiv_metadata_verified"]
    assert bibliography["verification_note"]


def test_every_entry_has_the_required_fields(entries):
    """A blank field would render as an empty spreadsheet cell."""
    for entry in entries:
        for field in REQUIRED_FIELDS:
            assert entry.get(field), f"{entry.get('title')!r} is missing {field}"


def test_every_entry_explains_why_it_is_cited(entries):
    """The 'used_for' line is the whole point of the sheet: a reading list without it
    is indistinguishable from a generic one."""
    for entry in entries:
        assert len(entry["used_for"]) > 20, entry["title"]


def test_authors_are_non_empty_strings(entries):
    """Authors drive both the 'first author' and 'all authors' columns."""
    for entry in entries:
        assert isinstance(entry["authors"], list)
        for author in entry["authors"]:
            assert author == author.strip()
            assert author


def test_titles_are_unique(entries):
    """A duplicate means the same work was added twice under different areas."""
    titles = [entry["title"] for entry in entries]
    assert len(titles) == len(set(titles))


def test_arxiv_ids_are_unique_and_well_formed(entries):
    """An id is NNNN.NNNNN with no version suffix, and no id appears twice."""
    ids = [entry["arxiv_id"] for entry in entries if entry["arxiv_id"]]
    assert len(ids) == len(set(ids))
    for arxiv_id in ids:
        head, _, tail = arxiv_id.partition(".")
        assert len(head) == 4 and head.isdigit(), arxiv_id
        assert tail.isdigit() and len(tail) in (4, 5), arxiv_id


def test_arxiv_entries_agree_with_their_reference_and_url(entries):
    """The reference string and link are derived from the id, so they must match it."""
    for entry in entries:
        if not entry["arxiv_id"]:
            continue
        assert entry["reference"] == f"arXiv:{entry['arxiv_id']}"
        assert entry["url"].endswith(entry["arxiv_id"])
        assert entry["source"] == "arXiv API"


def test_years_are_plausible_or_blank_for_software(entries):
    """Papers carry a year; libraries with no publication carry none."""
    for entry in entries:
        year = entry.get("year", "")
        if year == "":
            assert entry["source"] == "hand-entered", entry["title"]
            continue
        assert isinstance(year, int)
        assert 1990 <= year <= 2030, entry["title"]


def test_sources_are_known(entries):
    """Provenance is either the API or a hand entry; nothing else."""
    for entry in entries:
        assert entry["source"] in KNOWN_SOURCES


def test_areas_are_contiguous(entries):
    """Each area must appear as one unbroken run, so the sheet reads top to bottom by
    pipeline block as well as being filterable."""
    seen = []
    for entry in entries:
        if not seen or seen[-1] != entry["area"]:
            seen.append(entry["area"])
    assert len(seen) == len(set(seen)), f"an area is split across the file: {seen}"


def test_every_pipeline_block_is_represented(entries):
    """All six blocks have literature behind them; a missing one means a gap."""
    areas = " | ".join(entry["area"] for entry in entries)
    for block in ("Block 1", "Block 2", "Blocks 3 and 4", "Block 5", "Block 6"):
        assert block in areas


def test_row_builder_matches_the_column_definition():
    """The sheet writer zips rows against BIBLIO_COLUMNS, so the lengths must agree."""
    module = load_export_module()
    entry = {
        "area": "Foundations",
        "title": "A Paper",
        "year": 2020,
        "authors": ["First Author", "Second Author"],
        "reference": "arXiv:2005.11401",
        "url": "https://arxiv.org/abs/2005.11401",
        "used_for": "Something specific.",
    }
    row = module.bibliography_row(7, entry)
    assert len(row) == len(module.BIBLIO_COLUMNS)
    assert row[0] == 7
    assert row[4] == "First Author"
    assert row[5] == "First Author; Second Author"


def test_row_builder_survives_a_missing_author_list():
    """An entry with no authors must still produce a full-width row."""
    module = load_export_module()
    row = module.bibliography_row(1, {"title": "No Authors"})
    assert len(row) == len(module.BIBLIO_COLUMNS)
    assert row[4] == ""
    assert row[5] == ""


def test_loader_reads_the_real_file():
    """load_bibliography is what the export calls; point it at the real data."""
    module = load_export_module()
    loaded = module.load_bibliography(BIBLIOGRAPHY_PATH)
    assert loaded["entries"]
