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


# EDGAR sometimes inserts a newline inside a rendered word at a page boundary.
def _broken_word(word: str) -> str:
    return r"\s*".join(re.escape(char) for char in word)


_FINANCIAL = _broken_word("financial")
_STATEMENTS = _broken_word("statements")
_REAL_HEADER = r"^[ \t]*(?![^\r\n]*\|)"


# Match body headers at the start of a line. The negative lookahead excludes
# pipe-delimited table-of-contents rows found in the downloaded AAPL/MSFT
# filings, while the broken-word forms handle Microsoft page boundaries such
# as "FINA\nNCIAL STATEMENTS".
SECTION_PATTERNS = {
    # 10-K MD&A is Item 7; the 10-Q equivalent is Part I Item 2.
    "MD&A": re.compile(
        _REAL_HEADER
        + r"item[ \t]+(?:2|7)\.?[ \t]+management(?:['’\ufffd]?s)?[ \t]+discussion",
        re.I | re.M,
    ),
    # 10-K financial statements are Item 8; 10-Q Part I Item 1.
    "financial_statements": re.compile(
        _REAL_HEADER
        + rf"item[ \t]+(?:1|8)\.?[ \t]+{_FINANCIAL}\s+{_STATEMENTS}",
        re.I | re.M,
    ),
    "footnotes": re.compile(
        _REAL_HEADER
        + rf"notes[ \t]+to[ \t]+(?:the[ \t]+)?(?:consolidated[ \t]+)?"
          rf"{_FINANCIAL}\s+{_STATEMENTS}[ \t]*$",
        re.I | re.M,
    ),
    # 10-K only; without this its text gets absorbed into the MD&A section.
    "market_risk": re.compile(
        _REAL_HEADER + r"item[ \t]+7a\.?[ \t]+quantitative", re.I | re.M
    ),
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
