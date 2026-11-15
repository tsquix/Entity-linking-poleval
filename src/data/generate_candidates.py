import bz2
import json
from collections import defaultdict

ENTITY_FILE = "data/raw/entities.jsonl.bz2"

def build_candidate_index():

    candidate_index = defaultdict(list)

    with bz2.open(
        ENTITY_FILE,
        "rt",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f):

            entity = json.loads(line)

            entity_id = entity.get("id")

            labels = entity.get("labels", {})

            polish_label = labels.get("pl")

            if polish_label:

                candidate_index[
                    polish_label.lower()
                ].append(
                    entity_id
                )

            if line_number % 500_000 == 0:
                print(f"Processed {line_number:,} entities")

    return candidate_index

if __name__ == "__main__":

    candidate_index = (
        build_candidate_index()
    )

    print(f"\nUnique mentions: ")
    print(f"{len(candidate_index):,}")

    examples = [
        "belgia",
        "george washington",
        "polska"
    ]

    for mention in examples:

        print(f"\n{mention} -> " f"{candidate_index.get(mention)}")