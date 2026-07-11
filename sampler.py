from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from config import RANDOM_SEED, SAMPLE_SIZE, SELECTED_OUTPUT_DIR

def stratified_sample(
    df: pd.DataFrame,
    n: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    total = len(df)
    if total < n:
        raise ValueError(
            f"Inventory has only {total} contracts; cannot sample {n}."
        )

    category_counts = df["category"].value_counts()
    n_categories = len(category_counts)

    base_alloc: dict[str, int] = {cat: 1 for cat in category_counts.index}
    remaining = n - n_categories

    if remaining < 0:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    proportions = category_counts / category_counts.sum()
    extra = (proportions * remaining).astype(int)

    shortfall = remaining - extra.sum()
    remainders = (proportions * remaining) - extra
    top_cats = remainders.nlargest(int(shortfall)).index
    for cat in top_cats:
        extra[cat] += 1

    final_alloc: dict[str, int] = {}
    for cat in category_counts.index:
        available = int(category_counts[cat])
        wanted = base_alloc[cat] + int(extra.get(cat, 0))
        final_alloc[cat] = min(wanted, available)

    actual_total = sum(final_alloc.values())
    deficit = n - actual_total
    if deficit > 0:
        sorted_cats = category_counts.sort_values(ascending=False).index
        for cat in sorted_cats:
            if deficit == 0:
                break
            available = int(category_counts[cat])
            can_add = available - final_alloc[cat]
            add = min(can_add, deficit)
            final_alloc[cat] += add
            deficit -= add

    rng = pd.core.common  # just to confirm import; actual use below
    sampled_parts: list[pd.DataFrame] = []
    for cat, k in final_alloc.items():
        cat_df = df[df["category"] == cat]
        sampled_parts.append(cat_df.sample(n=k, random_state=seed))

    sample = pd.concat(sampled_parts).sample(frac=1, random_state=seed)
    return sample.reset_index(drop=True)

def copy_selected_contracts(
    sample: pd.DataFrame,
    output_dir: Path = SELECTED_OUTPUT_DIR,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for _, row in sample.iterrows():
        src = Path(row["file_path"])
        dst = output_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1

    manifest = sample[["contract_id", "file_path", "category", "category_raw", "part", "file_size_kb"]].copy()
    manifest = manifest.rename(columns={"file_path": "original_path"})
    manifest_path = output_dir / "selected_contracts.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"Copied {copied} PDFs → {output_dir}")
    print(f"Manifest written  → {manifest_path}")
    return sample

if __name__ == "__main__":
    from scanner import build_inventory, print_inventory_summary

    inventory = build_inventory()
    print_inventory_summary(inventory)

    sample = stratified_sample(inventory)
    print(f"\nSample size      : {len(sample)}")
    print("Sample by category:")
    print(sample["category"].value_counts().to_string())

    copy_selected_contracts(sample)
