import bz2
import json
from collections import defaultdict

import pandas as pd


ENTITY_FILE = "data/raw/entities.jsonl.bz2"
CONTEXT_FILE = "data/processed/context_dataset.csv"

OUTPUT_FILE = "data/processed/candidate_dataset.csv"


def build_candidate_index():

    candidate_index = defaultdict(list)

    with bz2.open(
        ENTITY_FILE,
        "rt",
        encoding="utf-8"
    ) as f:

        for line in f:

            entity = json.loads(line)

            entity_id = entity["id"]

            label = (
                entity.get("labels", {})
                .get("pl")
            )

            if label:

                candidate_index[label.lower()].append(entity_id)

    return candidate_index


def export_dataset():

    print("Building candidate index...")

    candidate_index = (
        build_candidate_index()
    )

    print("Loading context dataset...")

    first_chunk = True

    for chunk in pd.read_csv(
        CONTEXT_FILE,
        chunksize=10_000
    ):

        export_rows = []

        for _, row in chunk.iterrows():

            mention = row["mention"]

            if pd.isna(mention):
                mention = ""

            mention = str(mention).strip()

            candidates = (
                candidate_index.get(
                    mention.lower(),
                    []
                )
            )

            export_rows.append(
                {
                    "doc_id": row["doc_id"],
                    "context": row["context"],
                    "mention": row["mention"],
                    "candidate_ids": "|".join(candidates),
                    "true_wikidata_id": row["wikidata_id"]
                }
            )

        export_df = (
            pd.DataFrame(export_rows)
        )

        export_df.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

        print(f"Saved: {OUTPUT_FILE}")
        print(f"Rows: {len(export_df):,}")

if __name__ == "__main__":
    export_dataset()