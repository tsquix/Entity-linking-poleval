import bz2
from pathlib import Path

import pandas as pd

RAW_FILE = "data/raw/tokens-with-entities.tsv.bz2"
OUTPUT_FILE = "data/processed/context_dataset.csv"

MAX_DOCUMENTS = 50_000

def process_document(doc_id, tokens, entities):

    if not tokens:
        return []

    context = " ".join(tokens)

    rows = []
    seen = set()

    for mention, wikidata_id in entities:

        key = (mention, wikidata_id)

        if key in seen:
            continue

        seen.add(key)

        rows.append(
            {
                "doc_id": doc_id,
                "context": context,
                "mention": mention,
                "wikidata_id": wikidata_id
            }
        )

    return rows


def build_context_dataset():

    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True
    )

    all_rows = []

    current_doc_id = None
    current_tokens = []
    current_entities = []

    processed_docs = 0

    print("Building context dataset...")

    with bz2.open(RAW_FILE, "rt", encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            parts = line.strip().split("\t")

            if len(parts) != 5:
                continue

            doc_id, token, _, entity_name, wikidata_id = parts

            if current_doc_id is None:
                current_doc_id = doc_id

            if doc_id != current_doc_id:

                all_rows.extend(
                    process_document(
                        current_doc_id,
                        current_tokens,
                        current_entities
                    )
                )

                processed_docs += 1

                if processed_docs >= MAX_DOCUMENTS:
                    print(f"\nReached limit of {MAX_DOCUMENTS:,}.")
                    break

                current_doc_id = doc_id
                current_tokens = []
                current_entities = []

            current_tokens.append(token)

            if wikidata_id != "_":
                current_entities.append(
                    (
                        entity_name,
                        wikidata_id
                    )
                )

            if line_number % 1_000_000 == 0:
                print(f"Processed {line_number:,} rows")

        if current_doc_id is not None:
            all_rows.extend(
                process_document(
                    current_doc_id,
                    current_tokens,
                    current_entities,
                )
            )
    
        df = pd.DataFrame(all_rows)

        df = df.drop_duplicates(
            subset=[
                "doc_id",
                "mention",
                "wikidata_id",
            ]
        )

        df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(f"\nSaved: {OUTPUT_FILE}")
        print(f"Documents processed: {processed_docs:,}")
        print(f"Rows: {len(df):,}")

if __name__ == "__main__":
    build_context_dataset()
