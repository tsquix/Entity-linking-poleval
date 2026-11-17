import bz2
import json

from rank_bm25 import BM25Okapi

ENTITY_FILE = "data/raw/entities.jsonl.bz2"


def build_entity_documents():

    entity_ids = []
    documents = []

    with bz2.open(
        ENTITY_FILE,
        "rt",
        encoding="utf-8"
    ) as f:

        for line in f:

            entity = json.loads(line)

            entity_id = entity["id"]

            labels = entity.get("labels", {})
            wiki = entity.get("wiki", {})

            text = (
                f"{labels.get('pl', '')} "
                f"{wiki.get('pl', '')}"
            ).strip()

            if not text:
                continue

            entity_ids.append(
                entity_id
            )

            documents.append(
                text.split()
            )

    return entity_ids, documents


def build_bm25():

    entity_ids, documents = (
        build_entity_documents()
    )

    bm25 = BM25Okapi(documents)

    return bm25, entity_ids

if __name__ == "__main__":

    bm25, entity_ids = build_bm25()

    query = "Polska"

    scores = bm25.get_scores(
        query.split()
    )

    best_idx = scores.argmax()

    print(f"Query: {query}")
    print(f"Best entity: ")
    print(f"{entity_ids[best_idx]}")

