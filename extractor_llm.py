from __future__ import annotations

import json
import re
from typing import Optional

from chunker import split_contract
from ollama_client import generate
from config import MODEL_NAME, NUM_CTX, CLAUSE_MERGE_STRATEGY

CLAUSE_TYPES = ["termination_clause", "confidentiality_clause", "liability_clause"]

_CLAUSE_KEYWORDS: dict[str, list[str]] = {
    "termination_clause": [
        r"\bterminat\w*",
        r"\bcancel\w*",
        r"\bexpir\w*\s+(of\s+)?(this\s+)?agreement",
        r"\bearly\s+termination\b",
        r"\bnotice\s+of\s+termination\b",
    ],
    "confidentiality_clause": [
        r"\bconfidential\w*\s+information\b",
        r"\bnon-?disclos\w*",
        r"\bnda\b",
        r"\btrade\s+secrets?\b",
        r"\bproprietary\s+information\b",
    ],
    "liability_clause": [
        r"\blimitation\s+of\s+liability\b",
        r"\bliab\w*",
        r"\bindemnif\w*",
        r"\bhold\s+harmless\b",
        r"\bconsequential\s+damages\b",
    ],
}

_MIN_SUBSTANTIVE_CHARS = 150

TOP_K_CHUNKS_PER_CLAUSE = 2

def _score_chunk(chunk: str, patterns: list[str]) -> int:
    return sum(len(re.findall(p, chunk, re.IGNORECASE)) for p in patterns)

def rank_chunks_for_clause(chunks: list[str], clause_type: str) -> list[int]:
    patterns = _CLAUSE_KEYWORDS[clause_type]
    scored: list[tuple[int, int, int]] = []  # (index, hit_count, length)
    for i, chunk in enumerate(chunks):
        hits = _score_chunk(chunk, patterns)
        if hits > 0:
            scored.append((i, hits, len(chunk)))

    scored.sort(
        key=lambda t: (t[1], t[2] >= _MIN_SUBSTANTIVE_CHARS, t[2]),
        reverse=True,
    )
    return [idx for idx, _, _ in scored]

_CLAUSE_LABELS = {
    "termination_clause": "termination or cancellation clause (conditions under which the agreement can be ended)",
    "confidentiality_clause": "confidentiality or non-disclosure clause (obligations to keep information secret)",
    "liability_clause": "liability, indemnification, or limitation-of-liability clause",
}

_EXTRACTION_PROMPT = """\
You are a precise legal-text extractor. Read the CONTRACT EXCERPT below.

TASK: Find the {clause_label}, if one is present in this excerpt.

RULES:
- Copy the EXACT sentences from the excerpt that ARE the {clause_label}.
- The excerpt may contain MULTIPLE unrelated sections (addresses, notices, \
definitions, other clause types). Extract ONLY the {clause_label} sentences — \
do NOT copy addresses, contact details, phone/fax numbers, or unrelated clauses.
- If this excerpt does not contain that clause, respond with null.
- Do NOT explain, paraphrase, summarize, or add any commentary.
- Output ONLY this JSON object, nothing else:
{{"clause": "exact copied sentences or null"}}

CONTRACT EXCERPT:
{chunk}

JSON:"""

_EXTRACTION_STRICT_PROMPT = """\
From the contract excerpt below, copy ONLY the exact sentences that form \
the {clause_label}. Do not copy addresses, phone numbers, or other clauses.
If the clause is absent, use null.
Output ONLY: {{"clause": "text or null"}}

CONTRACT EXCERPT:
{chunk}

JSON:"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_PLACEHOLDER_PATTERNS = [
    "or null",
    "exact text",
    "exact copied text",
    "verbatim text of",
    "copy the exact text",
    "clause_label",
    "is null",
    "clause is null",
    "clause is not",
]

_MIN_CLAUSE_CHARS = 40  # anything shorter is a heading/fragment, not a real clause

def _looks_like_echo(value: str) -> bool:
    lowered = value.lower()
    return any(p in lowered for p in _PLACEHOLDER_PATTERNS)

def _is_header_only(value: str) -> bool:
    stripped = value.strip()
    if len(stripped) < _MIN_CLAUSE_CHARS:
        return True
    if re.match(r'^[\d.()\sa-zA-Z]{0,8}\)?\s*\.?\s*[A-Z][A-Za-z\s]{1,40}[.:]?\s*$', stripped):
        return True
    return False

def _looks_like_echo(value: str) -> bool:
    lowered = value.lower()
    return any(p in lowered for p in _PLACEHOLDER_PATTERNS)

def _parse_json_response(raw: str) -> Optional[dict]:
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    fence_match = _JSON_FENCE_RE.search(raw)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    return None

def _safe_str(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in ("null", "none", "n/a", "not found", "not present", ""):
        return ""
    if _looks_like_echo(s):
        return ""  # model echoed instructional text
    if _is_header_only(s):
        return ""  # model grabbed section heading, not clause body
    return s

def _extract_clause_from_chunk(
    chunk: str, clause_type: str, strict: bool = False
) -> str:
    max_chars = 4800
    if len(chunk) > max_chars:
        chunk = chunk[:max_chars] + "\n[truncated]"

    clause_label = _CLAUSE_LABELS[clause_type]
    prompt_tpl = _EXTRACTION_STRICT_PROMPT if strict else _EXTRACTION_PROMPT
    prompt = prompt_tpl.format(chunk=chunk, clause_label=clause_label)

    raw = generate(
        prompt,
        model=MODEL_NAME,
        num_ctx=NUM_CTX,
        temperature=0.0,
        use_json_format=True,
    )
    parsed = _parse_json_response(raw)

    if parsed is None:
        if not strict:
            return _extract_clause_from_chunk(chunk, clause_type, strict=True)
        return ""

    return _safe_str(parsed.get("clause"))

def _merge_longest(candidates: list[str]) -> str:
    non_empty = [c for c in candidates if c.strip()]
    if not non_empty:
        return ""
    return max(non_empty, key=len)

def _dedupe_across_clause_types(merged: dict[str, str]) -> dict[str, str]:
    seen: dict[str, str] = {}
    result = dict(merged)

    for clause_type in CLAUSE_TYPES:
        text = result.get(clause_type, "")
        if not text or text == "Not found":
            continue

        if text in seen:
            scores = {
                ct: _score_chunk(text, _CLAUSE_KEYWORDS[ct])
                for ct in CLAUSE_TYPES
            }
            winner = max(scores, key=scores.get)
            for ct in CLAUSE_TYPES:
                if ct != winner and result.get(ct) == text:
                    result[ct] = "Not found"
        else:
            seen[text] = clause_type

    return result

def extract_clauses(
    contract_id: str,
    cleaned_text: str,
    merge_strategy: str = CLAUSE_MERGE_STRATEGY,
    top_k: int = TOP_K_CHUNKS_PER_CLAUSE,
) -> dict:
    chunks = split_contract(cleaned_text)
    chunk_results: list[dict] = []
    merged: dict[str, str] = {}

    for clause_type in CLAUSE_TYPES:
        ranked_indices = rank_chunks_for_clause(chunks, clause_type)

        if not ranked_indices:
            merged[clause_type] = "Not found"
            continue

        top_indices = ranked_indices[:top_k]
        candidates: list[str] = []

        for idx in top_indices:
            extracted = _extract_clause_from_chunk(chunks[idx], clause_type)
            chunk_results.append(
                {
                    "chunk_index": idx,
                    "clause_type": clause_type,
                    "extracted": extracted,
                }
            )
            if extracted:
                candidates.append(extracted)

        merged_value = _merge_longest(candidates)
        merged[clause_type] = merged_value if merged_value else "Not found"

    merged = _dedupe_across_clause_types(merged)

    return {
        **merged,
        "chunk_results": chunk_results,
        "n_chunks": len(chunks),
    }
