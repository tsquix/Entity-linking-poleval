from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_FILE = "data/processed/entity_mentions.csv"
OUTPUT_DIR = "reports/figures"

def run_eda():
    df = pd.read_csv(DATA_FILE)

    print("||=== DATASET INFO ===||")
    print(f"Liczba rekordów: {len(df)}")
    print(f"Liczba unikalnych encji: {df['wikidata_id'].nunique()}")

    print("\n||=== TOP 10 ENCJI ===||")
    print(df.sort_values("count", ascending=False).head(10))

    print("\n||=== STATYSTYKI COUNT ===||")
    print(df["count"].describe())

    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    top20 = (
        df.sort_values("count", ascending=False).head(20)
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        top20['mention'],
        top20['count']
    )

    plt.xticks(
        rotation=90
    )

    plt.title("Top 20 najczęstszych encji")
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/top20_entities.png"
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    plt.hist(
        df['count'],
        bins=50,
        log=True
    )

    plt.title("Rozkład liczby wystąpień encji")

    plt.xlabel("Liczba wystąpień")
    plt.ylabel("Liczba encji (skala logarytmiczna)")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/entity_frequency_distribution.png"
    )

    plt.close()

    print("\nWykresy zapisane w reports/figures/")

if __name__ == "__main__":
    run_eda()