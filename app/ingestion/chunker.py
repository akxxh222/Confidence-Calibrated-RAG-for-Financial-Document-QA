"""
Section-aware chunking for 10-K / 10-Q filings.

Design decision (per Section 5/7 of the system design doc): naive fixed-size
chunking splits tables and financial statements mid-sentence, which directly
degrades retrieval-confidence quality later on. Instead we first split on
known Item/section boundaries, then apply a secondary size-based split only
*within* a section so no chunk crosses a section boundary.
"""
import re
from dataclasses import dataclass

# Regexes matching common 10-K/10-Q Item headers. Real filings are messy;
# this is a starting point to refine against actual downloaded HTML/text.
# Known real-data issues to check on the first filing: table-of-contents lines
# match the same patterns (creating short spurious sections), inline
# cross-references to "notes to consolidated financial statements" can split
# sections mid-body, and EDGAR HTML sometimes uses "Part II, Item 7" wording.
SECTION_PATTERNS = {
    # 10-K MD&A is Item 7; the 10-Q equivalent is Part I Item 2.
    "MD&A": re.compile(r"item\s+(?:2|7)\.?\s+management'?s discussion", re.I),
    # 10-K financial statements are Item 8; 10-Q Part I Item 1.
    "financial_statements": re.compile(r"item\s+(?:1|8)\.?\s+financial statements", re.I),
    "footnotes": re.compile(r"notes to (the )?consolidated financial statements", re.I),
    # 10-K only; without this its text gets absorbed into the MD&A section.
    "market_risk": re.compile(r"item\s+7a\.?\s+quantitative", re.I),
    "qa_segment": re.compile(r"question-and-answer|q&a session", re.I),
}

MAX_CHUNK_TOKENS = 400   # rough word-count proxy; swap for a real tokenizer later
OVERLAP_TOKENS = 50


@dataclass
class Chunk:
    section: str
    text: str
    page_ref: str | None = None


def split_into_sections(full_text: str) -> list[tuple[str, str]]:
    """Returns [(section_label, section_text), ...] by finding header offsets."""
    matches = []
    for label, pattern in SECTION_PATTERNS.items():
        for m in pattern.finditer(full_text):
            matches.append((m.start(), label))
    matches.sort()

    if not matches:
        return [("unlabeled", full_text)]

    # Retain everything before the first matched header instead of dropping it
    # (business description / cover pages still hold retrievable content).
    if matches[0][0] > 0:
        matches.insert(0, (0, "unlabeled"))

    sections = []
    for i, (start, label) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(full_text)
        sections.append((label, full_text[start:end]))
    return sections


def _word_chunks(text: str, max_tokens: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + max_tokens]))
        i += max_tokens - overlap
    return chunks


def chunk_document(full_text: str) -> list[Chunk]:
    chunks = []
    for label, section_text in split_into_sections(full_text):
        for piece in _word_chunks(section_text, MAX_CHUNK_TOKENS, OVERLAP_TOKENS):
            if piece.strip():
                chunks.append(Chunk(section=label, text=piece.strip()))
    return chunks
