import bz2
import json

ENTITY_FILE = "data/raw/entities.jsonl.bz2"


def build_entity_documents():

    entity_documents = {}

    with bz2.open(
        ENTITY_FILE,
        "rt",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f):

            entity = json.loads(line)

            entity_id = entity["id"]

            labels = entity.get("labels", {})

            wiki = entity.get("wiki", {})

            label_pl = labels.get("pl", "")

            wiki_pl = wiki.get("pl", "")

            text = (
                f"{label_pl} "
                f"{wiki_pl}"
            ).strip()

            entity_documents[
                entity_id
            ] = text

            if line_number % 500_000 == 0:
                print(f"Processed: {line_number:,}")

    print("\nEntity documents: ")
    print(f"{len(entity_documents):,}")

    return entity_documents

if __name__ == "__main__":

    entity_documents = (
        build_entity_documents()
    )

    examples = [
        "Q31",
        "Q23",
        "Q36"
    ]

    for entity_id in examples:

        print(f"\n{entity_id}:")

        print(entity_documents.get(entity_id))