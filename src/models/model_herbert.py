import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import json

MODEL_NAME = "allegro/herbert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
    return embedding


def extract_window(context, mention, window_size=250):

    context_lower = context.lower()
    mention_lower = mention.lower()

    start = context_lower.find(mention_lower)

    if start == -1:
        return context[:500]

    end = start + len(mention)
    left = max(0, start - window_size)
    right = end + window_size

    return context[left:right]


def parse_candidates(raw_candidates):
    if pd.isna(raw_candidates) or raw_candidates == "":
        return []

    return raw_candidates.split("|")


def load_entity_labels(jsonl_path, needed_ids):
    result = {}

    with open(jsonl_path, encoding="utf-8") as entities:
        for line in entities:
            entity = json.loads(line)
            if entity["id"] in needed_ids:
                result[entity["id"]] = entity["labels"].get("pl")

    return result


def rank_candidates(context_text, candidate_ids, id_to_label):
    results = []
    emb1 = get_embedding(context_text)
    for candidate_id in candidate_ids:
        label = id_to_label.get(candidate_id)
        if label:
            emb2 = get_embedding(label)
            results.append(
                (
                    candidate_id,
                    torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item(),
                )
            )
    return sorted(results, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/sample_candidates.csv")

    for raw_candidates in df["candidate_ids"]:
        parsed = parse_candidates(raw_candidates)

    # test_labels = load_entity_labels("data/entities.jsonl", {"Q31"})
    # print(test_labels)

    sample_df = df.head(50)
    needed_ids = set()
    for raw in sample_df["candidate_ids"]:
        needed_ids.update(parse_candidates(raw))

    id_to_label = load_entity_labels("data/entities.jsonl", needed_ids)

    correct = 0
    total = 0

    for _, row in sample_df.iterrows():
        candidates = parse_candidates(row["candidate_ids"])
        window = extract_window(row["context"], row["mention"])
        ranked = rank_candidates(window, candidates, id_to_label)

        if not ranked:
            top_1_id = None
            top_1_score = None
            top_1_label = None
        else:
            top_1_id = ranked[0][0]
            top_1_score = ranked[0][1]
            top_1_label = id_to_label.get(top_1_id)

        true_id = row["true_wikidata_id"]

        print(row["mention"], top_1_id, top_1_label, top_1_score, true_id)

        total += 1
        if top_1_id == true_id:
            correct += 1

    accuracy = 100 * correct / total
    print(f"Accuracy {accuracy:.2f}% ({correct}/{total})")
