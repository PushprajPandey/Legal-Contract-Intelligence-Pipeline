from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from config import (
    PREPROCESSED_OUTPUT_PATH,
    SAMPLE_SIZE,
    SELECTED_OUTPUT_DIR,
)
from scanner import build_inventory, print_inventory_summary
from sampler import stratified_sample, copy_selected_contracts
from extractor import extract_text, quality_check
from cleaner import clean_text

SCHEMA_DOCS = {
    "contract_id": (
        "Filename stem (no extension) — unique identifier derived from the "
        "original PDF filename."
    ),
    "category": (
        "Normalised contract category (near-duplicate folder names across Parts "
        "are collapsed to one canonical label — e.g. 'Affiliate Agreement' and "
        "'Affiliate_Agreements' both become 'Affiliate_Agreements')."
    ),
    "category_raw": (
        "Original folder name on disk before normalisation. Useful for tracing "
        "back to the exact source subfolder."
    ),
    "part": "Dataset part the contract belongs to: 'Part_I', 'Part_II', or 'Part_III'.",
    "original_pdf_path": "Absolute path to the source PDF in the CUAD dataset.",
    "file_size_kb": "File size of the source PDF in kilobytes.",
    "extraction_method": (
        "Which library was used for final text extraction: 'pdfplumber' or 'pymupdf'."
    ),
    "char_count": "Number of characters in cleaned_text.",
    "raw_char_count": "Number of characters in raw_text (before cleaning).",
    "txt_reference_found": (
        "Whether a matching .txt reference file was found in full_contract_txt/."
    ),
    "txt_char_count": "Number of characters in the reference .txt file (0 if not found).",
    "extraction_ratio": (
        "Ratio of PDF extraction char count to .txt reference char count. "
        "Values well below 1.0 may indicate extraction problems."
    ),
    "quality_flagged": (
        "True if extraction looks suspicious (very short output, or ratio < 0.3 "
        "versus the .txt reference)."
    ),
    "quality_flag_reason": "Human-readable explanation of why quality_flagged is True (empty string if not flagged).",
    "raw_text": (
        "Full text extracted directly from the PDF, after unicode normalization "
        "but before any structural cleaning."
    ),
    "cleaned_text": (
        "Fully normalized text: control chars removed, page headers/footers stripped, "
        "broken PDF line-breaks joined, whitespace collapsed. "
        "This is the field Task 2 (LLM extraction) should consume."
    ),
}

def process_contract(row: pd.Series) -> dict:
    pdf_path = Path(row["file_path"])

    raw_text, method = extract_text(pdf_path)

    qc = quality_check(raw_text, row["contract_id"])

    cleaned = clean_text(raw_text)

    return {
        "contract_id": row["contract_id"],
        "category": row["category"],             # normalised canonical name
        "category_raw": row.get("category_raw", row["category"]),  # original folder
        "part": row["part"],
        "original_pdf_path": row["file_path"],
        "file_size_kb": row["file_size_kb"],
        "extraction_method": method,
        "char_count": len(cleaned),
        "raw_char_count": len(raw_text),
        "txt_reference_found": qc["txt_path_found"],
        "txt_char_count": qc["txt_char_count"],
        "extraction_ratio": qc["ratio"],
        "quality_flagged": qc["flagged"],
        "quality_flag_reason": qc["flag_reason"],
        "raw_text": raw_text,
        "cleaned_text": cleaned,
    }

def run_pipeline() -> None:
    t0 = time.time()

    print("\n[1/5] Building contract inventory …")
    inventory = build_inventory()
    print_inventory_summary(inventory)

    print(f"\n[2/5] Sampling {SAMPLE_SIZE} contracts (stratified, seed=42) …")
    sample = stratified_sample(inventory)
    print(f"  Sampled {len(sample)} contracts across "
          f"{sample['category'].nunique()} categories")
    print("\n  Sample distribution by category:")
    for cat, n in sample["category"].value_counts().items():
        print(f"    {cat:<45} : {n}")

    print(f"\n[3/5] Copying selected PDFs → {SELECTED_OUTPUT_DIR} …")
    copy_selected_contracts(sample)

    print(f"\n[4/5] Extracting and cleaning text from {len(sample)} PDFs …")
    records: list[dict] = []
    flagged_contracts: list[str] = []

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        print(f"  [{i:>2}/{len(sample)}] {row['contract_id'][:60]} …", end=" ")
        record = process_contract(row)
        records.append(record)

        flag = "⚠ FLAGGED" if record["quality_flagged"] else f"{record['char_count']:,} chars"
        method_short = record["extraction_method"][:3].upper()
        print(f"[{method_short}] {flag}")

        if record["quality_flagged"]:
            flagged_contracts.append(
                f"  {row['contract_id']}: {record['quality_flag_reason']}"
            )

    print(f"\n[5/5] Saving output → {PREPROCESSED_OUTPUT_PATH} …")
    output = {
        "__schema__": SCHEMA_DOCS,
        "metadata": {
            "total_contracts": len(records),
            "sample_size": SAMPLE_SIZE,
            "random_seed": 42,
            "categories_represented": sorted(sample["category"].unique().tolist()),
        },
        "contracts": records,
    }

    PREPROCESSED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PREPROCESSED_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    char_counts = [r["char_count"] for r in records]
    flagged_n = sum(1 for r in records if r["quality_flagged"])

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Contracts processed : {len(records)}")
    print(f"  Time elapsed        : {elapsed:.1f}s")
    print(f"  Output file         : {PREPROCESSED_OUTPUT_PATH}")
    print(f"  Avg char count      : {sum(char_counts)/len(char_counts):,.0f}")
    print(f"  Min char count      : {min(char_counts):,}")
    print(f"  Max char count      : {max(char_counts):,}")
    print(f"  Quality flagged     : {flagged_n}")

    if flagged_contracts:
        print("\n  Flagged contracts:")
        for msg in flagged_contracts:
            print(msg)
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
