import bz2
from pathlib import Path

import pandas as pd

RAW_FILE = "data/raw/tokens-with-entities.tsv.bz2"
OUTPUT_FILE = "data/processed/entity_linking_dataset.csv"

def build_dataset():

    dataset_rows = []

    with bz2.open(RAW_FILE, "rt", encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            parts = line.strip().split("\t")

            if len(parts) != 5:
                continue

            doc_id, token, flag, entity_name, wikidata_id = parts

            if wikidata_id == "_":
                continue

            dataset_rows.append(
                {
                    "doc_id": doc_id,
                    "mention": entity_name,
                    "wikidata_id": wikidata_id
                }
            )

            if line_number % 1_000_000 == 0:
                print(f"Processed {line_number:,} rows")

    df = pd.DataFrame(dataset_rows)

    df = df.drop_duplicates()

    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")

if __name__ == "__main__":
    build_dataset()