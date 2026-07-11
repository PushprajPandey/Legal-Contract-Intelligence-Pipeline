from __future__ import annotations

import re
import unicodedata
from collections import Counter

_LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb06": "st",
    "\ufb05": "st",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2014": " - ",
    "\u2013": " - ",
    "\u2026": "...",
    "\u00a0": " ",
    "\u00ad": "",
}

_LIGATURE_RE = re.compile("|".join(re.escape(k) for k in _LIGATURE_MAP))

def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _LIGATURE_RE.sub(lambda m: _LIGATURE_MAP[m.group(0)], text)
    return text

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

def _remove_control_chars(text: str) -> str:
    return _CONTROL_CHAR_RE.sub("", text)

_PAGE_NUMBER_RE = re.compile(
    r"^\s*"
    r"("
    r"page\s+\d+(\s+of\s+\d+)?"      # "Page 3" / "Page 3 of 10"
    r"|\d+\s*$"                        # line that is purely a number
    r"|[-–—]\s*\d+\s*[-–—]"           # "- 3 -"
    r"|\d+\s*/\s*\d+"                  # "3/10"
    r")"
    r"\s*$",
    re.IGNORECASE,
)

def _remove_page_artifacts(lines: list[str]) -> list[str]:
    short_line_counts: Counter = Counter()
    for line in lines:
        stripped = line.strip()
        if 0 < len(stripped) <= 120:
            short_line_counts[stripped] += 1

    boilerplate: set[str] = {
        line for line, cnt in short_line_counts.items() if cnt >= 3
    }

    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if _PAGE_NUMBER_RE.match(stripped):
            continue
        if stripped in boilerplate:
            continue
        cleaned.append(line)

    return cleaned

_LINE_END_PUNCT_RE = re.compile(r"[.?!:;,)\]}\'\"\u201d]\s*$")

_NEXT_LINE_UPPER_RE = re.compile(r"^\s*([A-Z\d(•\-–])")

def _join_broken_lines(lines: list[str]) -> list[str]:
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            result.append(line)
            i += 1
            continue

        if i + 1 < len(lines):
            next_line = lines[i + 1]
            next_stripped = next_line.strip()

            can_join = (
                len(line.strip()) >= 20
                and not _LINE_END_PUNCT_RE.search(line)
                and bool(next_stripped)
                and not _NEXT_LINE_UPPER_RE.match(next_stripped)
            )

            if can_join:
                result.append(line.rstrip() + " " + next_stripped)
                i += 2
                continue

        result.append(line)
        i += 1

    return result

_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")

def _normalize_whitespace(text: str) -> str:
    text = text.replace("\t", " ")
    lines = text.split("\n")
    lines = [_MULTI_SPACE_RE.sub(" ", line).rstrip() for line in lines]
    text = "\n".join(lines)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()

def clean_text(raw_text: str) -> str:
    text = _normalize_unicode(raw_text)
    text = _remove_control_chars(text)

    lines = text.splitlines()
    lines = _remove_page_artifacts(lines)
    lines = _join_broken_lines(lines)
    text = "\n".join(lines)

    text = _normalize_whitespace(text)
    return text

if __name__ == "__main__":
    sample = (
        "Page 1\n\n"
        "AFFILIATE AGREEMENT\n\n"
        "This Agreement is entered into as of the 1st day of\n"
        'January, 2007 between CreditCards.com Inc. (\u201cCompany\u201d)\n'
        "and the Affiliate.\n\n"
        "1\n\n"
        "1.  DEFINITIONS\n\n"
        '"Affiliate" means any entity that directly or indirectly\n'
        "controls, is controlled by, or is under common control with\n"
        "the Company.\n\n"
        "Page 2\n\n"
    )
    print("=== RAW ===")
    print(repr(sample))
    print("\n=== CLEANED ===")
    print(clean_text(sample))
