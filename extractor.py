from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

import pdfplumber
import fitz  # PyMuPDF

from config import MIN_CHARS_THRESHOLD, TXT_MATCH_RATIO_THRESHOLD, TXT_ROOT

def extract_with_pdfplumber(pdf_path: Path) -> str:
    pages: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if text:
                    pages.append(text)
    except Exception:
        return ""
    return "\n".join(pages)

def extract_with_pymupdf(pdf_path: Path) -> str:
    pages: list[str] = []
    try:
        doc = fitz.open(str(pdf_path))
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
    except Exception:
        return ""
    return "\n".join(pages)

def extract_text(pdf_path: Path) -> tuple[str, str]:
    text = extract_with_pdfplumber(pdf_path)

    if len(text.strip()) >= MIN_CHARS_THRESHOLD:
        return text, "pdfplumber"

    fallback = extract_with_pymupdf(pdf_path)
    if len(fallback.strip()) >= len(text.strip()):
        return fallback, "pymupdf"

    return (text if len(text) >= len(fallback) else fallback), "pymupdf"

def find_txt_counterpart(contract_id: str, txt_root: Path = TXT_ROOT) -> Optional[Path]:
    if not txt_root.exists():
        return None

    exact = txt_root / (contract_id + ".txt")
    if exact.exists():
        return exact

    for txt_file in txt_root.glob("*.txt"):
        if txt_file.stem.startswith(contract_id):
            return txt_file

    cid_lower = contract_id.lower()
    for txt_file in txt_root.glob("*.txt"):
        if cid_lower in txt_file.stem.lower():
            return txt_file

    return None

def quality_check(
    raw_text: str,
    contract_id: str,
    txt_root: Path = TXT_ROOT,
) -> dict:
    pdf_chars = len(raw_text.strip())
    result = {
        "txt_path_found": False,
        "txt_char_count": 0,
        "pdf_char_count": pdf_chars,
        "ratio": None,
        "flagged": False,
        "flag_reason": "",
    }

    if pdf_chars < MIN_CHARS_THRESHOLD:
        result["flagged"] = True
        result["flag_reason"] = f"Extracted only {pdf_chars} chars (< {MIN_CHARS_THRESHOLD})"
        return result

    txt_path = find_txt_counterpart(contract_id, txt_root)
    if txt_path is None:
        return result  # can't compare, not flagged on length alone

    result["txt_path_found"] = True
    try:
        txt_content = txt_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return result

    txt_chars = len(txt_content.strip())
    result["txt_char_count"] = txt_chars

    if txt_chars == 0:
        return result

    ratio = pdf_chars / txt_chars
    result["ratio"] = round(ratio, 4)

    if ratio < TXT_MATCH_RATIO_THRESHOLD:
        result["flagged"] = True
        result["flag_reason"] = (
            f"PDF extraction ({pdf_chars} chars) is only "
            f"{ratio:.1%} of the .txt reference ({txt_chars} chars)"
        )

    return result

if __name__ == "__main__":
    from config import PDF_ROOT

    pdfs = list(PDF_ROOT.rglob("*.pdf")) + list(PDF_ROOT.rglob("*.PDF"))
    if pdfs:
        p = pdfs[0]
        print(f"Testing: {p.name}")
        text, method = extract_text(p)
        print(f"Method : {method}")
        print(f"Chars  : {len(text)}")
        qc = quality_check(text, p.stem)
        print(f"QC     : {qc}")
        print("---")
        print(text[:500])
