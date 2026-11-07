import pandas as pd

TRAIN_FILE = "data/processed/train.csv"

def build_popularity_baseline():

    print("Loading training data...")

    df = pd.read_csv(TRAIN_FILE)

    print(f"Loaded {len(df):,} training examples")

    mention_popularity = (
        df.groupby(['mention', 'wikidata_id'])
        .size()
        .reset_index(name="count")
    )

    baseline_mapping = (
        mention_popularity
        .sort_values(
            "count",
            ascending=False
        )
        .drop_duplicates(subset=['mention'])
    )

    print(f"\nBaseline mappings: {len(baseline_mapping):,}")

    print("\nTop baseline mappings:")

    print(baseline_mapping.head(20))

    return baseline_mapping

if __name__ == "__main__":
    build_popularity_baseline()