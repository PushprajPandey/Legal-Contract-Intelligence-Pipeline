from __future__ import annotations

import re
from typing import Callable

from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))

    _TOKENIZER = "tiktoken"

except ImportError:  # pragma: no cover
    def count_tokens(text: str) -> int:
        return int(len(text.split()) / 0.75)

    _TOKENIZER = "word_approx"

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]

def _make_chunks(
    units: list[str],
    size: int,
    overlap: int,
    token_fn: Callable[[str], int],
) -> list[str]:
    chunks: list[str] = []
    current_units: list[str] = []
    current_tokens: int = 0

    for unit in units:
        unit_tokens = token_fn(unit)

        if unit_tokens > size:
            if current_units:
                chunks.append("\n\n".join(current_units))
                current_units = []
                current_tokens = 0
            chunks.append(unit)
            continue

        if current_tokens + unit_tokens > size and current_units:
            chunks.append("\n\n".join(current_units))

            overlap_units: list[str] = []
            overlap_tokens = 0
            for u in reversed(current_units):
                t = token_fn(u)
                if overlap_tokens + t > overlap:
                    break
                overlap_units.insert(0, u)
                overlap_tokens += t

            current_units = overlap_units
            current_tokens = overlap_tokens

        current_units.append(unit)
        current_tokens += unit_tokens

    if current_units:
        chunks.append("\n\n".join(current_units))

    return chunks

def split_contract(
    text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    if not text or not text.strip():
        return [""]

    total_tokens = count_tokens(text)

    if total_tokens <= chunk_size:
        return [text.strip()]

    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    units: list[str] = []
    for para in paragraphs:
        if count_tokens(para) > chunk_size:
            units.extend(_split_sentences(para))
        else:
            units.append(para)

    chunks = _make_chunks(units, chunk_size, overlap, count_tokens)

    return chunks if chunks else [text.strip()]

def chunk_summary(text: str) -> dict:
    chunks = split_contract(text)
    return {
        "total_tokens": count_tokens(text),
        "n_chunks": len(chunks),
        "tokenizer": _TOKENIZER,
        "chunk_tokens": [count_tokens(c) for c in chunks],
    }

if __name__ == "__main__":
    import json
    from pathlib import Path
    from config import PREPROCESSED_OUTPUT_PATH

    with open(PREPROCESSED_OUTPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    contracts = data["contracts"]
    print(f"Tokenizer: {_TOKENIZER}")
    print(f"{'contract_id':<50} {'chars':>8} {'tokens':>8} {'chunks':>7}")
    print("-" * 76)
    for c in contracts:
        info = chunk_summary(c["cleaned_text"])
        print(
            f"{c['contract_id'][:50]:<50} "
            f"{len(c['cleaned_text']):>8,} "
            f"{info['total_tokens']:>8,} "
            f"{info['n_chunks']:>7}"
        )
