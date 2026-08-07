"""Item/section-boundary chunker for SEC 10-K/10-Q filings.

Real filings are large, Item-sectioned documents (Item 1. Business, Item 1A.
Risk Factors, Item 7. MD&A, Item 8. Financial Statements, ...) that mix HTML
tables with prose. Naive token-count chunking would cut a table in half or mix
two Items into one chunk, destroying section-scoped retrieval and — worse —
truncating a numeric table row so the number Phase 3 needs to verify never
survives into a retrievable chunk.

Strategy:
  1. Flatten every <table> into a pipe-delimited text block *in place*, so a
     later plain-text render keeps every cell (numbers included) verbatim and
     in its original document position.
  2. Render the whole document to text and split it into lines.
  3. Detect "Item N[letter]." heading lines (SEC's standard heading format)
     and cut the document into per-Item sections at those boundaries — this is
     the section-boundary chunking Phase 1 requires, not a token-count split.
  4. Within each Item section, sub-chunk by a bounded word window (still
     needed — Item 7/8 can run to tens of thousands of words) but a sub-chunk
     never crosses an Item boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

import config

# Some EDGAR primary documents (esp. inline-XBRL 8-K cover pages) are XML that
# the HTML parser can still walk safely for our text-extraction purposes; we
# don't need XBRL-specific parsing, so suppress the informational warning.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_ITEM_HEADING_RE = re.compile(
    r"^\s*ITEM\s+(\d{1,2}[A-Z]?)\.?\s*[-–—:.]?\s*(.{0,120})?$",
    re.IGNORECASE,
)
_MAX_HEADING_LINE_LEN = 160


@dataclass
class FilingSection:
    item: str          # e.g. "Item 1A" or "FULL_DOCUMENT" if no headings were found
    title: str
    text: str


@dataclass
class FilingChunk:
    item: str
    title: str
    text: str
    chunk_num: int


def _table_to_text(table) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _flatten_tables_in_place(soup: BeautifulSoup) -> None:
    """Replace every <table> with its pipe-delimited text, preserving position."""
    for table in soup.find_all("table"):
        text = _table_to_text(table)
        table.replace_with("\n[TABLE]\n" + text + "\n[/TABLE]\n" if text else "")


def html_to_lines(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    _flatten_tables_in_place(soup)
    raw = soup.get_text("\n")
    raw = raw.replace("\xa0", " ")
    lines = [line.strip() for line in raw.split("\n")]
    return [line for line in lines if line]


def _match_heading(line: str) -> tuple[str, str] | None:
    if len(line) > _MAX_HEADING_LINE_LEN:
        return None
    m = _ITEM_HEADING_RE.match(line)
    if not m:
        return None
    number = m.group(1).upper()
    title = (m.group(2) or "").strip(" .-–—:")
    return f"Item {number}", title


def split_into_sections(html: str) -> list[FilingSection]:
    """Split filing HTML into Item-boundary sections (never by raw token count)."""
    lines = html_to_lines(html)
    boundaries: list[tuple[int, str, str]] = []
    for idx, line in enumerate(lines):
        match = _match_heading(line)
        if match:
            item, title = match
            boundaries.append((idx, item, title))

    if not boundaries:
        return [FilingSection("FULL_DOCUMENT", "", "\n".join(lines))]

    # Collapse consecutive duplicate headings for the same Item (TOC entries +
    # the real heading both match); keep the LAST occurrence of each Item
    # number seen before a different Item appears, since SEC filings commonly
    # list a table of contents with the same "Item N." labels before the body.
    deduped: list[tuple[int, str, str]] = []
    seen_items: dict[str, int] = {}
    for idx, item, title in boundaries:
        seen_items[item] = idx  # remember most recent occurrence
    # Rebuild boundaries keeping only the final occurrence position per item,
    # but preserve first-seen ORDER of distinct items for section ordering.
    order: list[str] = []
    for _, item, _ in boundaries:
        if item not in order:
            order.append(item)
    for item in order:
        idx = seen_items[item]
        title = next(t for i, it, t in boundaries if it == item and i == idx)
        deduped.append((idx, item, title))
    deduped.sort(key=lambda x: x[0])

    sections: list[FilingSection] = []
    preamble = "\n".join(lines[: deduped[0][0]]).strip()
    if preamble:
        sections.append(FilingSection("PREAMBLE", "", preamble))
    for i, (idx, item, title) in enumerate(deduped):
        end = deduped[i + 1][0] if i + 1 < len(deduped) else len(lines)
        body = "\n".join(lines[idx + 1 : end]).strip()
        sections.append(FilingSection(item, title, body))
    return sections


def _word_chunk(text: str, chunk_words: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    step = max(1, chunk_words - overlap)
    while i < len(words):
        window = words[i : i + chunk_words]
        chunk = " ".join(window).strip()
        if len(chunk) > 40:
            chunks.append(chunk)
        i += step
    return chunks


def chunk_filing(
    html: str,
    chunk_words: int | None = None,
    overlap: int | None = None,
) -> list[FilingChunk]:
    """Item-boundary chunking with a bounded word-window sub-split per section."""
    chunk_words = chunk_words or config.EDGAR_ITEM_CHUNK_WORDS
    overlap = overlap or config.EDGAR_ITEM_CHUNK_OVERLAP
    sections = split_into_sections(html)
    out: list[FilingChunk] = []
    for section in sections:
        if not section.text.strip():
            continue
        sub_chunks = _word_chunk(section.text, chunk_words, overlap)
        for n, sub in enumerate(sub_chunks):
            out.append(FilingChunk(item=section.item, title=section.title, text=sub, chunk_num=n))
    return out
