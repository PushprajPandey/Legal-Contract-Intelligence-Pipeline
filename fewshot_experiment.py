from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from config import PREPROCESSED_OUTPUT_PATH
from ollama_client import generate, check_models_available, OllamaError
from chunker import split_contract
from extractor_llm import _parse_json_response, _safe_str, CLAUSE_TYPES

_FEW_SHOT_EXAMPLES = """\
EXAMPLE 1 INPUT:
\"\"\"
Either party may terminate this Agreement upon thirty (30) days written notice
to the other party.  Upon termination, all licenses granted hereunder shall
immediately cease, and each party shall return or destroy Confidential
Information belonging to the other.
\"\"\"
EXAMPLE 1 OUTPUT:
{
  "termination_clause": "Either party may terminate this Agreement upon thirty (30) days written notice to the other party.",
  "confidentiality_clause": "each party shall return or destroy Confidential Information belonging to the other.",
  "liability_clause": null
}

EXAMPLE 2 INPUT:
\"\"\"
IN NO EVENT SHALL EITHER PARTY BE LIABLE TO THE OTHER FOR ANY INDIRECT,
INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
ARISING OUT OF OR RELATED TO THIS AGREEMENT, EVEN IF SUCH PARTY HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.  Each party's total liability
shall not exceed the fees paid in the twelve (12) months preceding the claim.
\"\"\"
EXAMPLE 2 OUTPUT:
{
  "termination_clause": null,
  "confidentiality_clause": null,
  "liability_clause": "IN NO EVENT SHALL EITHER PARTY BE LIABLE TO THE OTHER FOR ANY INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF OR RELATED TO THIS AGREEMENT, EVEN IF SUCH PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. Each party's total liability shall not exceed the fees paid in the twelve (12) months preceding the claim."
}"""

_ZERO_SHOT_PROMPT = """\
You are a legal contract analyst.  Read the following contract excerpt and \
extract VERBATIM text for each clause type IF present:

  1. termination_clause
  2. confidentiality_clause
  3. liability_clause

Use null for any clause not found.
Respond with ONLY a valid JSON object — no markdown, no explanation.

CONTRACT EXCERPT:
\"\"\"
{chunk}
\"\"\"

JSON response:"""

_FEW_SHOT_PROMPT = """\
You are a legal contract analyst.  Read the following contract excerpt and \
extract VERBATIM text for each clause type IF present:

  1. termination_clause
  2. confidentiality_clause
  3. liability_clause

Use null for any clause not found.
Respond with ONLY a valid JSON object — no markdown, no explanation.

Here are two examples of correct extractions:

{examples}

Now extract from this contract excerpt:

CONTRACT EXCERPT:
\"\"\"
{chunk}
\"\"\"

JSON response:"""

from config import MODEL_NAME, NUM_CTX

def _run_prompt(prompt: str) -> dict[str, str]:
    raw = generate(prompt, model=MODEL_NAME, num_ctx=NUM_CTX, temperature=0.0)
    parsed = _parse_json_response(raw)
    if parsed is None:
        return {k: "" for k in CLAUSE_TYPES}
    return {k: _safe_str(parsed.get(k)) for k in CLAUSE_TYPES}

def _zero_shot(chunk: str) -> dict[str, str]:
    return _run_prompt(_ZERO_SHOT_PROMPT.format(chunk=chunk))

def _few_shot(chunk: str) -> dict[str, str]:
    return _run_prompt(
        _FEW_SHOT_PROMPT.format(examples=_FEW_SHOT_EXAMPLES, chunk=chunk)
    )

def _compare_contract(contract: dict) -> dict:
    text = contract["cleaned_text"]
    chunks = split_contract(text)
    chunk = chunks[0]  # Compare on the same first chunk for fairness

    zs = _zero_shot(chunk)
    fs = _few_shot(chunk)

    return {
        "contract_id":  contract["contract_id"],
        "category":     contract["category"],
        "chunk_chars":  len(chunk),
        "zero_shot":    zs,
        "few_shot":     fs,
    }

def _clause_cell(text: str, max_len: int = 200) -> str:
    if not text or text == "Not found":
        return "*(not found)*"
    trimmed = text[:max_len]
    if len(text) > max_len:
        trimmed += " …"
    return trimmed.replace("\n", " ").replace("|", "\\|")

def _word_count(text: str) -> int:
    return len(text.split()) if text and text != "Not found" else 0

def write_markdown_report(comparisons: list[dict], output_path: Path) -> None:
    lines = [
        "# Zero-Shot vs Few-Shot Clause Extraction — Comparison\n",
        "## Methodology\n",
        "- **Zero-shot**: prompt asks for clause extraction with no examples.\n",
        "- **Few-shot**: same prompt + 2 hardcoded example chunk→extraction pairs.\n",
        "- Comparison is on the **first chunk** of each contract (same input for both).\n",
        "- Metric: presence/absence of each clause, approximate length (word count), "
        "and qualitative note.\n\n",
    ]

    for comp in comparisons:
        cid = comp["contract_id"]
        cat = comp["category"]
        lines.append(f"---\n\n### {cid[:70]}\n**Category:** {cat}\n\n")

        header = "| Clause Type | Zero-Shot (words) | Few-Shot (words) | Notes |\n"
        sep    = "|---|---|---|---|\n"
        lines.append(header)
        lines.append(sep)

        for clause in CLAUSE_TYPES:
            zs_text = comp["zero_shot"].get(clause, "")
            fs_text = comp["few_shot"].get(clause, "")
            zs_words = _word_count(zs_text)
            fs_words = _word_count(fs_text)

            if not zs_text and not fs_text:
                note = "Neither found — likely absent in this chunk"
            elif zs_text and not fs_text:
                note = "Zero-shot found it; few-shot missed"
            elif not zs_text and fs_text:
                note = "Few-shot found it; zero-shot missed"
            elif abs(zs_words - fs_words) <= 5:
                note = "Both found similar length output"
            elif fs_words > zs_words:
                note = f"Few-shot more complete (+{fs_words - zs_words} words)"
            else:
                note = f"Zero-shot more complete (+{zs_words - fs_words} words)"

            zs_cell = f"{_clause_cell(zs_text)} *({zs_words}w)*" if zs_text else "*(not found)*"
            fs_cell = f"{_clause_cell(fs_text)} *({fs_words}w)*" if fs_text else "*(not found)*"
            lines.append(f"| `{clause}` | {zs_cell} | {fs_cell} | {note} |\n")

        lines.append("\n")

    lines.append("---\n\n## Summary Across All 5 Contracts\n\n")
    lines.append("| Clause Type | Zero-Shot Found | Few-Shot Found |\n")
    lines.append("|---|---|---|\n")
    for clause in CLAUSE_TYPES:
        zs_found = sum(1 for c in comparisons if c["zero_shot"].get(clause, ""))
        fs_found = sum(1 for c in comparisons if c["few_shot"].get(clause, ""))
        lines.append(f"| `{clause}` | {zs_found}/5 | {fs_found}/5 |\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Report written → {output_path}")

def main() -> None:
    print("Checking Ollama …")
    try:
        check_models_available()
    except OllamaError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    with open(PREPROCESSED_OUTPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    contracts = data["contracts"]

    target_cats = [
        "Franchise", "Collaboration", "Affiliate_Agreements",
        "Transportation", "License_Agreements",
    ]
    selected: list[dict] = []
    used_cats: set[str] = set()
    for c in contracts:
        if c["category"] in target_cats and c["category"] not in used_cats:
            selected.append(c)
            used_cats.add(c["category"])
        if len(selected) == 5:
            break

    for c in contracts:
        if len(selected) >= 5:
            break
        if c["contract_id"] not in {s["contract_id"] for s in selected}:
            selected.append(c)

    print(f"Running comparison on {len(selected)} contracts …\n")

    comparisons: list[dict] = []
    for c in selected:
        print(f"  Processing: {c['contract_id'][:60]}")
        comp = _compare_contract(c)
        comparisons.append(comp)

        for clause in CLAUSE_TYPES:
            zs = "✓" if comp["zero_shot"].get(clause) else "✗"
            fs = "✓" if comp["few_shot"].get(clause) else "✗"
            print(f"    {clause:<30} zero-shot:{zs}  few-shot:{fs}")
        print()

    output_path = Path(__file__).parent / "fewshot_comparison.md"
    write_markdown_report(comparisons, output_path)

    raw_path = Path(__file__).parent / "fewshot_raw_results.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(comparisons, f, ensure_ascii=False, indent=2)
    print(f"Raw results saved → {raw_path}")

if __name__ == "__main__":
    main()
