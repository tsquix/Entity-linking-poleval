import bz2
import json
from collections import defaultdict

import pandas as pd
from rank_bm25 import BM25Okapi


ENTITY_FILE = "data/raw/entities.jsonl.bz2"
CONTEXT_FILE = "data/processed/context_dataset.csv"


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

def build_entity_documents():

    entity_documents = {}

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
                .get("pl", "")
            )

            wiki = (
                entity.get("wiki", {})
                .get("pl", "")
            )

            entity_documents[entity_id] = f"{label} {wiki}"

    return entity_documents


def predict_simple(
        context,
        mention,
        candidate_index,
        entity_documents
):

    candidates = candidate_index.get(mention.lower(), [])

    if not candidates:
        return None

    candidate_texts = []

    for candidate_id in candidates:

        text = entity_documents.get(
            candidate_id,
            ""
        )

        candidate_texts.append(
            text.split()
        )

    bm25 = BM25Okapi(
        candidate_texts
    )

    scores = bm25.get_scores(
        context.split()
    )

    best_idx = scores.argmax()

    return candidates[best_idx]


def evaluate_sample():

    print("Building candidate index...")

    candidate_index = (
        build_candidate_index()
    )

    print("Build entity documents...")

    entity_documents = (
        build_entity_documents()
    )

    print("Loading context dataset...")

    df = pd.read_csv(
        CONTEXT_FILE
    )

    sample = df.head(1000)

    correct = 0

    for _, row in sample.iterrows():

        prediction = predict_simple(
            row["context"],
            row["mention"],
            candidate_index,
            entity_documents
        )

        if prediction == row["wikidata_id"]:
            correct += 1

    accuracy = correct / len(sample)

    print("\nSample accuracy:")
    print(f"{accuracy:.4f}")

if __name__ == "__main__":
    evaluate_sample()