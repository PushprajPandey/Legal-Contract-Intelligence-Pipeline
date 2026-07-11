from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import PDF_ROOT

CATEGORY_NORMALISATION: dict[str, str] = {
    "Affiliate Agreement":        "Affiliate_Agreements",   # Part_III (singular)
    "Affiliate_Agreements":       "Affiliate_Agreements",   # Part_I

    "Endorsement":                "Endorsement",            # Part_I / Part_II
    "Endorsement Agreement":      "Endorsement",            # Part_III

    "Joint Venture":              "Joint_Venture",          # Part_I
    "Joint Venture _ Filing":     "Joint_Venture",          # Part_III

    "Agency Agreements":          "Agency_Agreements",
    "Co_Branding":                "Co_Branding",
    "Consulting Agreements":      "Consulting_Agreements",
    "License_Agreements":         "License_Agreements",
    "Non_Compete_Non_Solicit":    "Non_Compete_Non_Solicit",
    "Strategic Alliance":         "Strategic_Alliance",
}

def normalise_category(raw: str) -> str:
    return CATEGORY_NORMALISATION.get(raw, raw)

def build_inventory(pdf_root: Path = PDF_ROOT) -> pd.DataFrame:
    records: list[dict] = []

    for part_dir in sorted(pdf_root.iterdir()):
        if not part_dir.is_dir() or not part_dir.name.startswith("Part_"):
            continue
        part_name = part_dir.name  # "Part_I" / "Part_II" / "Part_III"

        for category_dir in sorted(part_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            raw_category = category_dir.name
            canonical_category = normalise_category(raw_category)

            for pdf_file in sorted(category_dir.iterdir()):
                if pdf_file.suffix.lower() != ".pdf":
                    continue

                size_kb = round(pdf_file.stat().st_size / 1024, 2)

                records.append(
                    {
                        "contract_id":   pdf_file.stem,
                        "file_path":     str(pdf_file),
                        "category":      canonical_category,   # normalised
                        "category_raw":  raw_category,         # original folder name
                        "part":          part_name,
                        "file_size_kb":  size_kb,
                    }
                )

    return pd.DataFrame(records)

def print_inventory_summary(df: pd.DataFrame) -> None:
    print("=" * 62)
    print("CUAD CONTRACT INVENTORY SUMMARY")
    print("=" * 62)
    print(f"Total contracts found : {len(df)}")
    print(f"Unique categories     : {df['category'].nunique()}  "
          f"(raw folders: {df['category_raw'].nunique()})")
    print()

    print("── By Part ────────────────────────────────────────")
    for part, count in df["part"].value_counts().sort_index().items():
        print(f"  {part:<12} : {count:>4} contracts")
    print()

    print("── By Category (normalised) ───────────────────────")
    for cat, count in df["category"].value_counts().sort_values(ascending=False).items():
        print(f"  {cat:<45} : {count:>4}")
    print("=" * 62)

if __name__ == "__main__":
    inventory = build_inventory()
    print_inventory_summary(inventory)
