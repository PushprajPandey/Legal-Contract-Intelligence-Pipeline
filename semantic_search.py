from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from config import (
    TASK2_JSON_PATH,
    SEMANTIC_STORE_PATH,
    EMBED_MODEL_NAME,
)
from ollama_client import embed, OllamaError

_CLAUSE_TYPES = [
    "termination_clause",
    "confidentiality_clause",
    "liability_clause",
]

def build_store(
    results_path: Path = TASK2_JSON_PATH,
    store_path: Path = SEMANTIC_STORE_PATH,
) -> None:
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run pipeline_task2.py first.")
        sys.exit(1)

    with open(results_path, encoding="utf-8") as f:
        records = json.load(f)

    vectors: list[list[float]] = []
    metadata: list[dict] = []

    print(f"Building semantic store from {len(records)} contracts …")
    for rec in records:
        cid = rec["contract_id"]
        for clause_type in _CLAUSE_TYPES:
            text = rec.get(clause_type, "")
            if not text or text.strip().lower() in ("not found", ""):
                continue
            try:
                vec = embed(text, model=EMBED_MODEL_NAME)
                vectors.append(vec)
                metadata.append({
                    "contract_id": cid,
                    "clause_type": clause_type,
                    "text": text,
                })
                print(f"  Embedded {cid[:50]} [{clause_type}]")
            except OllamaError as e:
                print(f"  [SKIP] {cid} [{clause_type}] — {e}")

    if not vectors:
        print("No clauses were embedded. Store not written.")
        return

    matrix = np.array(vectors, dtype=np.float32)

    np.savez(
        store_path,
        vectors=matrix,
        metadata=np.array([json.dumps(m) for m in metadata]),
    )
    print(f"\nStore saved → {store_path}")
    print(f"  {len(metadata)} clauses embedded across {len(records)} contracts.")

def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    normed = matrix / norms
    return normed @ q

def load_store(store_path: Path = SEMANTIC_STORE_PATH) -> tuple[np.ndarray, list[dict]]:
    if not store_path.exists():
        raise FileNotFoundError(
            f"Semantic store not found at {store_path}.\n"
            "Run:  python semantic_search.py build"
        )
    data = np.load(store_path, allow_pickle=True)
    matrix = data["vectors"].astype(np.float32)
    metadata = [json.loads(s) for s in data["metadata"]]
    return matrix, metadata

def search(
    query: str,
    top_k: int = 5,
    store_path: Path = SEMANTIC_STORE_PATH,
) -> list[dict]:
    matrix, metadata = load_store(store_path)

    query_vec = np.array(embed(query, model=EMBED_MODEL_NAME), dtype=np.float32)
    scores = _cosine_similarity(query_vec, matrix)

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices, 1):
        m = metadata[idx]
        results.append({
            "rank":        rank,
            "score":       float(scores[idx]),
            "contract_id": m["contract_id"],
            "clause_type": m["clause_type"],
            "text":        m["text"],
        })
    return results

def _print_results(results: list[dict]) -> None:
    if not results:
        print("No results found.")
        return
    for r in results:
        print(f"\n{'─' * 60}")
        print(f"Rank {r['rank']}  |  Score: {r['score']:.4f}")
        print(f"Contract : {r['contract_id'][:70]}")
        print(f"Clause   : {r['clause_type']}")
        print(f"Text     : {r['text'][:300]}{'…' if len(r['text']) > 300 else ''}")
    print(f"\n{'─' * 60}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic search over CUAD extracted clauses."
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("build", help="Build the embedding store from task2_results.json")

    search_p = sub.add_parser("search", help="Search the clause store")
    search_p.add_argument("query", type=str, help="Natural language search query")
    search_p.add_argument("--top-k", type=int, default=5, help="Number of results (default 5)")

    args = parser.parse_args()

    if args.command == "build":
        build_store()
    elif args.command == "search":
        try:
            results = search(args.query, top_k=args.top_k)
            _print_results(results)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        except OllamaError as e:
            print(f"Ollama error: {e}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
