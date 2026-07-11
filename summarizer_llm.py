from __future__ import annotations

import re

from ollama_client import generate
from config import MODEL_NAME, NUM_CTX

_MIN_WORDS = 80
_MAX_WORDS = 170

_SUMMARY_PROMPT = """\
Write a 100-150 word summary of this contract. Cover: (1) purpose, (2) key obligations, (3) notable risks.
Write ONLY the summary. No headings. No bullet points. Stop after 150 words.

CONTRACT:
{text}

Summary:"""

_RETRY_PROMPT = """\
Rewrite in exactly 100-150 words. Same meaning, correct length. Output ONLY the paragraph.

Original:
{previous}

Rewritten:"""

def _count_words(text: str) -> int:
    return len(text.split())

def _strip_preamble(text: str) -> str:
    text = text.strip()
    text = re.sub(
        r'^(?:here(?:\s+is)?(?:\s+a)?\s+)?(?:summary|overview)\s*[:\-]\s*',
        '', text, flags=re.IGNORECASE
    )
    return text.strip()

SUMMARY_INPUT_CHARS = 3000

def summarize_contract(contract_id: str, cleaned_text: str) -> dict:
    text = cleaned_text[:SUMMARY_INPUT_CHARS]
    if len(cleaned_text) > SUMMARY_INPUT_CHARS:
        text += "\n[...contract continues...]"

    prompt = _SUMMARY_PROMPT.format(text=text)
    raw = generate(
        prompt, model=MODEL_NAME, num_ctx=NUM_CTX,
        temperature=0.1, num_predict=250, use_json_format=False,
    )
    summary = _strip_preamble(raw)
    word_count = _count_words(summary)

    retried = False
    if not (_MIN_WORDS <= word_count <= _MAX_WORDS):
        retry_prompt = _RETRY_PROMPT.format(
            actual_words=word_count, previous=summary
        )
        retry_raw = generate(
            retry_prompt, model=MODEL_NAME, num_ctx=NUM_CTX,
            temperature=0.1, num_predict=250, use_json_format=False,
        )
        summary = _strip_preamble(retry_raw)
        word_count = _count_words(summary)
        retried = True

    return {
        "summary": summary,
        "chunk_summaries": [],
        "word_count": word_count,
        "retried": retried,
    }
