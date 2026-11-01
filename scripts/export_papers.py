"""Export the papers behind this project to an Excel workbook.

Two different things are called "the papers we used", so the workbook keeps them apart:

- the **methods bibliography** (`data/methods_bibliography.json`): the literature and
  software the method itself rests on, each entry mapped to the block that uses it;
- the **corpora** (`data/raw/*.jsonl`): the arXiv papers the system indexes and
  retrieves from. The dev corpus is what every quality number is measured on, while the
  scale corpus exists only for the vector-store benchmark.

Conflating the two would misread the evaluation, so they are never merged into one list.

    python scripts/export_papers.py
    python scripts/export_papers.py --skip-scale    # leave out the 50,000-paper corpus
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.ingest.io_utils import read_jsonl  # noqa: E402

FONT_NAME = "Arial"
HEADER_FILL = PatternFill("solid", fgColor="DDEBF7")

COLUMNS = [
    ("Title", 72),
    ("Year", 8),
    ("First author", 24),
    ("All authors", 60),
    ("arXiv ID", 14),
    ("PDF URL", 40),
]

BIBLIO_COLUMNS = [
    ("#", 5),
    ("Area", 34),
    ("Title", 66),
    ("Year", 7),
    ("First author", 22),
    ("All authors", 50),
    ("Why it is here", 64),
    ("Reference", 22),
    ("Link", 32),
]

LINK_FONT_COLOR = "0563C1"

# Columns of the bibliography sheet whose text should wrap rather than run off the page.
BIBLIO_WRAP_COLUMNS = (2, 3, 6, 7)


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description="Export the project's arXiv papers to Excel.")
    parser.add_argument(
        "--skip-scale",
        action="store_true",
        help="Export only the dev corpus, leaving out the 50,000-paper benchmark corpus.",
    )
    parser.add_argument(
        "--output", default=None, help="Output path (default: papers/papers_used.xlsx)."
    )
    return parser.parse_args()


def year_of(paper: Dict[str, Any]) -> Any:
    """The publication year as an integer, or an empty string when the date is unusable."""
    published = str(paper.get("published", ""))
    if len(published) >= 4 and published[:4].isdigit():
        return int(published[:4])
    return ""


def paper_row(paper: Dict[str, Any]) -> List[Any]:
    """One paper as a spreadsheet row, in the order of COLUMNS."""
    authors = paper.get("authors") or []
    return [
        paper.get("title", ""),
        year_of(paper),
        authors[0] if authors else "",
        "; ".join(authors),
        paper.get("arxiv_id", ""),
        paper.get("pdf_url", ""),
    ]


def load_papers(path: Path) -> List[List[Any]]:
    """Read a papers JSONL file and return its rows, newest first then by title."""
    rows = [paper_row(paper) for paper in read_jsonl(path)]
    rows.sort(key=lambda row: (-(row[1] or 0), str(row[0]).lower()))
    return rows


def write_sheet(workbook: Workbook, title: str, rows: List[List[Any]], note: str) -> Any:
    """Write one sheet: a note line, a header row, then the papers."""
    sheet = workbook.create_sheet(title)

    sheet.cell(row=1, column=1, value=note).font = Font(name=FONT_NAME, italic=True, size=10)
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(COLUMNS))

    for index, (heading, width) in enumerate(COLUMNS, start=1):
        cell = sheet.cell(row=2, column=index, value=heading)
        cell.font = Font(name=FONT_NAME, bold=True)
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(vertical="center")
        sheet.column_dimensions[get_column_letter(index)].width = width

    body_font = Font(name=FONT_NAME)
    for offset, row in enumerate(rows, start=3):
        for index, value in enumerate(row, start=1):
            cell = sheet.cell(row=offset, column=index, value=value)
            cell.font = body_font
            if index == 2:
                cell.number_format = "0"  # a year is never 2,026
                cell.alignment = Alignment(horizontal="center")
            else:
                cell.alignment = Alignment(vertical="top")

    sheet.freeze_panes = "A3"
    sheet.auto_filter.ref = f"A2:{get_column_letter(len(COLUMNS))}{len(rows) + 2}"
    return sheet


def load_bibliography(path: Path) -> Dict[str, Any]:
    """Read the hand-curated methods bibliography."""
    return json.loads(path.read_text(encoding="utf-8"))


def bibliography_row(number: int, entry: Dict[str, Any]) -> List[Any]:
    """One bibliography entry as a spreadsheet row, in the order of BIBLIO_COLUMNS."""
    authors = entry.get("authors") or []
    return [
        number,
        entry.get("area", ""),
        entry.get("title", ""),
        entry.get("year", ""),
        authors[0] if authors else "",
        "; ".join(authors),
        entry.get("used_for", ""),
        entry.get("reference", ""),
        entry.get("url", ""),
    ]


def write_bibliography_sheet(workbook: Workbook, bibliography: Dict[str, Any], index: int) -> Any:
    """Write the methods bibliography: what the method rests on, and where each piece is used.

    Placed at `index` so it sits beside the corpus sheets without being mistaken for
    one. Entries arrive already ordered by pipeline block, so the sheet is readable
    top to bottom as well as filterable by area.
    """
    sheet = workbook.create_sheet("Methods bibliography", index)

    note = (
        "The literature and software this project's method rests on, each entry mapped to the "
        "block or component that uses it. This is the project's reading list - NOT the corpus "
        "the system retrieves from, which is on the corpus sheets."
    )
    sheet.cell(row=1, column=1, value=note).font = Font(name=FONT_NAME, italic=True, size=10)
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(BIBLIO_COLUMNS))

    for column, (heading, width) in enumerate(BIBLIO_COLUMNS, start=1):
        cell = sheet.cell(row=2, column=column, value=heading)
        cell.font = Font(name=FONT_NAME, bold=True)
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        sheet.column_dimensions[get_column_letter(column)].width = width

    body_font = Font(name=FONT_NAME)
    link_font = Font(name=FONT_NAME, color=LINK_FONT_COLOR, underline="single")
    entries = bibliography["entries"]

    for offset, entry in enumerate(entries):
        row_number = offset + 3
        for column, value in enumerate(bibliography_row(offset + 1, entry), start=1):
            cell = sheet.cell(row=row_number, column=column, value=value)
            if column == len(BIBLIO_COLUMNS) and value:
                cell.font = link_font
                cell.hyperlink = value
            else:
                cell.font = body_font
            if column == 4:
                cell.number_format = "0"  # a year is never 2,024
                cell.alignment = Alignment(horizontal="center", vertical="top")
            else:
                cell.alignment = Alignment(
                    vertical="top", wrap_text=column in BIBLIO_WRAP_COLUMNS
                )

    sheet.freeze_panes = "C3"  # keep the number and area visible while scrolling right
    sheet.auto_filter.ref = f"A2:{get_column_letter(len(BIBLIO_COLUMNS))}{len(entries) + 2}"

    footer = len(entries) + 4
    sheet.cell(
        row=footer,
        column=1,
        value=bibliography["verification_note"],
    ).font = Font(name=FONT_NAME, italic=True, size=10)
    sheet.merge_cells(
        start_row=footer, start_column=1, end_row=footer, end_column=len(BIBLIO_COLUMNS)
    )
    return sheet


def write_about(workbook: Workbook, counts: Dict[str, int], bibliography: Dict[str, Any]) -> None:
    """Write the sheet that says where the data came from and what each sheet is."""
    sheet = workbook.create_sheet("About", 0)
    verified = bibliography.get("arxiv_metadata_verified", "an earlier date")
    lines = [
        ("Papers used in the Agentic RAG project", True),
        ("", False),
        (f"Generated by scripts/export_papers.py on {date.today().isoformat()}.", False),
        ("", False),
        ('Two different things are called "the papers we used", so they are kept apart:', False),
        ("    - the methods bibliography: the literature and software the method rests on;", False),
        ("    - the corpus sheets: the arXiv papers the system indexes and retrieves from.", False),
        ("Merging them would misread the evaluation, so they are never one list.", False),
        ("", False),
        (f"Sources: data/methods_bibliography.json (hand-curated; every arXiv entry's title,", False),
        (f"year and authors fetched from the arXiv API on {verified} rather than typed from", False),
        ("memory), and data/raw/papers.jsonl + data/raw/papers_scale.jsonl, saved at ingestion", False),
        ("from the arXiv API (categories cs.CL, cs.IR, cs.LG). The counts below are the row", False),
        ("counts of that snapshot; re-run the script to refresh them.", False),
        ("", False),
        ("Sheets", True),
    ]
    for number, (label, bold) in enumerate(lines, start=1):
        if not label:
            continue  # a spacer row: left empty, but it still advances the row number
        cell = sheet.cell(row=number, column=1, value=label)
        cell.font = Font(
            name=FONT_NAME, bold=bold, size=12 if bold and label.startswith("Papers") else 11
        )
    next_row = len(lines) + 1

    descriptions = [
        (
            "Methods bibliography",
            len(bibliography["entries"]),
            "The literature and software this project's method rests on, each entry mapped to "
            "the block or component that uses it. Ordered by pipeline block, filterable by area.",
        ),
        (
            "Dev corpus",
            counts.get("dev", 0),
            "The corpus the pipeline actually answers from. Chunked into 528 passages, indexed "
            "in Qdrant, and the basis of every accuracy number in docs/REPORT.md.",
        ),
        (
            "Scale corpus",
            counts.get("scale", 0),
            "Used only for the vector-store benchmark (FAISS vs HNSW, fp32 vs int8) at 343,235 "
            "chunks. No answer quality is measured on it.",
        ),
    ]
    start = next_row
    headings = [("Sheet", 22), ("Entries", 10), ("What it is", 90)]
    for index, (heading, width) in enumerate(headings, start=1):
        cell = sheet.cell(row=start, column=index, value=heading)
        cell.font = Font(name=FONT_NAME, bold=True)
        cell.fill = HEADER_FILL
        sheet.column_dimensions[get_column_letter(index)].width = width
    for offset, (name, count, description) in enumerate(descriptions, start=start + 1):
        if count == 0:
            continue
        sheet.cell(row=offset, column=1, value=name).font = Font(name=FONT_NAME)
        count_cell = sheet.cell(row=offset, column=2, value=count)
        count_cell.font = Font(name=FONT_NAME)
        count_cell.number_format = "#,##0"
        note = sheet.cell(row=offset, column=3, value=description)
        note.font = Font(name=FONT_NAME)
        note.alignment = Alignment(wrap_text=True, vertical="top")
        sheet.row_dimensions[offset].height = 30

    footer = start + len([d for d in descriptions if d[1]]) + 2
    sheet.cell(
        row=footer,
        column=1,
        value="HotpotQA and MultiHop-RAG are also evaluated against, but they are question sets "
        "built from Wikipedia paragraphs and news articles, not arXiv papers, so their documents "
        "appear on neither corpus sheet. The two dataset papers themselves are in the methods "
        "bibliography, under \"Evaluation datasets\".",
    ).font = Font(name=FONT_NAME, italic=True, size=10)


def main() -> None:
    """Build the workbook and report what was written."""
    args = parse_args()
    config = load_config()
    raw_dir = PROJECT_ROOT / config["paths"]["raw_dir"]

    dev_path = raw_dir / config["ingest"]["papers_file"]
    scale_path = raw_dir / config["scale"]["papers_file"]
    bibliography_path = PROJECT_ROOT / "data" / "methods_bibliography.json"

    if not dev_path.exists():
        raise SystemExit(f"{dev_path} not found; run scripts/run_ingest.py first")
    if not bibliography_path.exists():
        raise SystemExit(f"{bibliography_path} not found")

    bibliography = load_bibliography(bibliography_path)

    workbook = Workbook()
    workbook.remove(workbook.active)

    counts: Dict[str, int] = {}

    dev_rows = load_papers(dev_path)
    counts["dev"] = len(dev_rows)
    write_sheet(
        workbook,
        "Dev corpus",
        dev_rows,
        "The corpus the agentic RAG pipeline retrieves from; every accuracy number in "
        "docs/REPORT.md is measured on questions about these papers.",
    )

    if not args.skip_scale and scale_path.exists():
        scale_rows = load_papers(scale_path)
        counts["scale"] = len(scale_rows)
        write_sheet(
            workbook,
            "Scale corpus",
            scale_rows,
            "Used only for the vector-store benchmark (FAISS vs Qdrant HNSW, fp32 vs int8). "
            "No answer quality is measured on these papers.",
        )

    # About first, then the bibliography second: the reading list before the raw corpora.
    write_about(workbook, counts, bibliography)
    write_bibliography_sheet(workbook, bibliography, index=1)

    output = Path(args.output) if args.output else PROJECT_ROOT / "papers" / "papers_used.xlsx"
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)

    print(f"wrote {output}")
    print(f"  methods bibliography: {len(bibliography['entries'])} entries")
    for name, count in counts.items():
        print(f"  {name}: {count:,} papers")


if __name__ == "__main__":
    main()
