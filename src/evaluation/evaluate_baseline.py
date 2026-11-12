import pandas as pd

from src.evaluation.metrics import (
    accuracy,
    coverage,
    mrr_at_1
)

TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"


def build_baseline_mapping():

    train_df = pd.read_csv(TRAIN_FILE)

    mention_popularity = (
        train_df.groupby(
            ["mention", "wikidata_id"]
        )
        .size()
        .reset_index(name="count")
    )

    baseline_mapping = (
        mention_popularity
        .sort_values(
            "count",
            ascending=False
        )
        .drop_duplicates(
            subset=["mention"]
        )
    )

    return dict(
        zip(
            baseline_mapping["mention"],
            baseline_mapping["wikidata_id"]
        )
    )


def evaluate():

    print("Building baseline...")

    baseline = build_baseline_mapping()

    print("Loading test set...")

    test_df = pd.read_csv(TEST_FILE)

    y_true = []
    y_pred = []

    for _, row in test_df.iterrows():

        mention = row["mention"]

        prediction = baseline.get(
            mention,
            None
        )

        y_true.append(
            row["wikidata_id"]
        )

        y_pred.append(
            prediction
        )

    print("\nResults")
    print("Accuracy:", accuracy(y_true, y_pred))
    print("Coverage:", coverage(y_pred))
    print("MRR@1:", mrr_at_1(y_true, y_pred))

if __name__ == "__main__":
    evaluate()