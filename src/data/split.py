from pathlib import Path

import pandas as pd 
from sklearn.model_selection import train_test_split

SEED = 1234

INPUT_FILE = "data/processed/entity_linking_dataset.csv"
OUTPUT_DIR = "data/processed"

def create_split():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows: {len(df):,}")

    train_df, temp_df, = train_test_split(
        df,
        test_size=0.4,
        random_state=SEED,
        shuffle=True
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=SEED,
        shuffle=True
    )

    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        f"{OUTPUT_DIR}/train.csv",
        index=False
    )

    val_df.to_csv(
        f"{OUTPUT_DIR}/val.csv",
        index=False
    )

    test_df.to_csv(
        f"{OUTPUT_DIR}/test.csv",
        index=False
    )

    print("\nSplit completed")

    print(f"Train: {len(train_df):,}")
    print(f"Validation: {len(val_df):,}")
    print(f"Test: {len(test_df):,}")

if __name__ == "__main__":
    create_split()