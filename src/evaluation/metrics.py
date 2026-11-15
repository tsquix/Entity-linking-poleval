def _extract_top1(pred):
    if isinstance(pred, list):
        return pred[0] if pred else None
    return pred

def accuracy(y_true, y_pred):
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Invalid lengths")

    correct = 0
    for true_label, pred in zip(y_true, y_pred):
        top1 = _extract_top1(pred)
        if top1 is not None and top1 == true_label:
            correct += 1

    return correct / len(y_true)

def coverage(y_true, y_pred=None):
    preds = y_pred if y_pred is not None else y_true
    if not preds:
        raise ValueError("Empty predictions")

    predicted = 0
    for pred in preds:
        top1 = _extract_top1(pred)
        if top1 is not None and top1 != "":
            predicted += 1

    return predicted / len(preds)

def mrr_at_1(y_true, y_pred):
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Invalid lengths")

    scores = []
    for true_label, pred in zip(y_true, y_pred):
        top1 = _extract_top1(pred)
        scores.append(1.0 if (top1 is not None and top1 == true_label) else 0.0)

    return sum(scores) / len(scores)

if __name__ == "__main__":
    y_true = ["Q36", "Q270", "Q142", "Q183"]
    y_pred_flat = ["Q36", "Q270", None, "Q999"]
    y_pred_ranked = [["Q36", "Q1"], ["Q270"], [], ["Q999"]]

    print("Flat -> ACC:", accuracy(y_true, y_pred_flat), "COV:", coverage(y_true, y_pred_flat), "MRR@1:", mrr_at_1(y_true, y_pred_flat))
    print("Rank -> ACC:", accuracy(y_true, y_pred_ranked), "COV:", coverage(y_true, y_pred_ranked), "MRR@1:", mrr_at_1(y_true, y_pred_ranked))