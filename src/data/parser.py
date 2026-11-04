import bz2
from collections import Counter
from pathlib import Path

import pandas as pd

RAW_FILE = "data/raw/tokens-with-entities.tsv.bz2"
OUTPUT_FILE = "data/processed/entity_mentions.csv"

def extract_mentions():

    mention_entity_counter = Counter()

    with bz2.open(RAW_FILE, "rt", encoding="utf-8") as f:
        for line_number, line in enumerate(f):

            parts = line.strip().split("\t")

            if len(parts) != 5:
                continue

            doc_id, token, flag, entity_name, wikidata_id = parts

            if wikidata_id == "_":
                continue

            mention_entity_counter[(entity_name, wikidata_id)] += 1

            if line_number % 1_000_000 == 0:
                print(f"Processed {line_number:,} rows")

    rows = []

    for (mention, wikidata_id), count in mention_entity_counter.items():
        rows.append(
            {
                "mention": mention,
                "wikidata_id": wikidata_id,
                "count": count
            }
        )

    df = pd.DataFrame(rows)

    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"Rows: {len(df)}")

if __name__ == "__main__":
    extract_mentions()