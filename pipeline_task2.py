from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    PREPROCESSED_OUTPUT_PATH,
    TASK2_CSV_PATH,
    TASK2_JSON_PATH,
    TASK2_FULL_JSON_PATH,
    TASK2_CHECKPOINT_PATH,
    CHECKPOINT_EVERY,
)
from ollama_client import check_models_available, OllamaError
from extractor_llm import extract_clauses
from summarizer_llm import summarize_contract

DELIVERABLE_FIELDS = [
    "contract_id",
    "summary",
    "termination_clause",
    "confidentiality_clause",
    "liability_clause",
]

def _load_checkpoint() -> dict[str, dict]:
    if not TASK2_CHECKPOINT_PATH.exists():
        return {}
    try:
        with open(TASK2_CHECKPOINT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Resuming from checkpoint: {len(data)} contracts already done.")
        return data
    except (json.JSONDecodeError, OSError):
        print("  Checkpoint file corrupt — starting fresh.")
        return {}

def _save_checkpoint(results: dict[str, dict]) -> None:
    with open(TASK2_CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def _write_deliverables(results: dict[str, dict]) -> None:
    rows = []
    for cid, r in results.items():
        rows.append({
            "contract_id":           cid,
            "summary":               r.get("summary", ""),
            "termination_clause":    r.get("termination_clause", "Not found"),
            "confidentiality_clause": r.get("confidentiality_clause", "Not found"),
            "liability_clause":      r.get("liability_clause", "Not found"),
        })

    df = pd.DataFrame(rows, columns=DELIVERABLE_FIELDS)
    df.to_csv(TASK2_CSV_PATH, index=False, encoding="utf-8")

    with open(TASK2_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

def _write_full_results(results: dict[str, dict]) -> None:
    with open(TASK2_FULL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(list(results.values()), f, ensure_ascii=False, indent=2)

def process_contract(contract: dict) -> dict:
    cid = contract["contract_id"]
    text = contract["cleaned_text"]

    extraction = extract_clauses(cid, text)

    summarization = summarize_contract(cid, text)

    return {
        "contract_id":            cid,
        "summary":                summarization["summary"],
        "termination_clause":     extraction["termination_clause"],
        "confidentiality_clause": extraction["confidentiality_clause"],
        "liability_clause":       extraction["liability_clause"],
        "category":               contract.get("category", ""),
        "part":                   contract.get("part", ""),
        "char_count":             contract.get("char_count", 0),
        "n_chunks":               extraction["n_chunks"],
        "summary_word_count":     summarization["word_count"],
        "summary_retried":        summarization["retried"],
        "chunk_results":          extraction["chunk_results"],
        "chunk_summaries":        summarization["chunk_summaries"],
    }

def run_pipeline() -> None:
    t0 = time.time()

    print("Checking Ollama connection and models …")
    try:
        check_models_available()
        print("  ✓ Ollama running, models available.\n")
    except OllamaError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print(f"Loading contracts from {PREPROCESSED_OUTPUT_PATH} …")
    with open(PREPROCESSED_OUTPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    contracts = data["contracts"]
    print(f"  {len(contracts)} contracts loaded.\n")

    results: dict[str, dict] = _load_checkpoint()
    already_done = set(results.keys())
    todo = [c for c in contracts if c["contract_id"] not in already_done]

    if not todo:
        print("All contracts already processed. Writing final outputs …")
    else:
        print(f"Processing {len(todo)} contracts "
              f"({len(already_done)} already done from checkpoint) …\n")

    errors: list[str] = []
    since_last_checkpoint = 0

    for contract in tqdm(todo, desc="Contracts", unit="contract"):
        cid = contract["contract_id"]
        try:
            record = process_contract(contract)
            results[cid] = record
            since_last_checkpoint += 1
        except OllamaError as e:
            tqdm.write(f"  [SKIP] {cid[:60]} — Ollama error: {e}")
            errors.append(cid)
            continue
        except Exception as e:
            tqdm.write(f"  [SKIP] {cid[:60]} — Unexpected error: {e}")
            errors.append(cid)
            continue

        if since_last_checkpoint >= CHECKPOINT_EVERY:
            _save_checkpoint(results)
            since_last_checkpoint = 0
            tqdm.write(f"  ✓ Checkpoint saved ({len(results)} contracts done)")

    _save_checkpoint(results)

    print("\nWriting deliverables …")
    _write_deliverables(results)
    _write_full_results(results)

    elapsed = time.time() - t0

    success = len(results) - len(already_done)
    found_counts = {k: 0 for k in ["termination_clause", "confidentiality_clause", "liability_clause"]}
    for r in results.values():
        for k in found_counts:
            if r.get(k, "Not found") != "Not found":
                found_counts[k] += 1

    print("\n" + "=" * 60)
    print("TASK 2 PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Contracts processed  : {success}")
    print(f"  Errors / skipped     : {len(errors)}")
    print(f"  Time elapsed         : {elapsed / 60:.1f} min")
    print(f"  Termination found    : {found_counts['termination_clause']}/50")
    print(f"  Confidentiality found: {found_counts['confidentiality_clause']}/50")
    print(f"  Liability found      : {found_counts['liability_clause']}/50")
    print()
    print(f"  Deliverables:")
    print(f"    {TASK2_CSV_PATH}")
    print(f"    {TASK2_JSON_PATH}")
    print(f"    {TASK2_FULL_JSON_PATH}")
    print("=" * 60)

    if errors:
        print(f"\nSkipped contracts ({len(errors)}):")
        for cid in errors:
            print(f"  {cid}")

if __name__ == "__main__":
    run_pipeline()
