def accuracy(y_true, y_pred):

    correct = 0

    for true_label, predicted_label in zip(
        y_true,
        y_pred
    ):
        if true_label == predicted_label:
            correct += 1

    return correct / len(y_true)


def coverage(y_pred):

    predicted = 0

    for prediction in y_pred:

        if prediction is not None:
            predicted += 1

    return predicted / len(y_pred)

def mrr_at_1(y_true, y_pred):

    scores = []

    for true_label, predicted_label in zip(
        y_true,
        y_pred
    ):
        if true_label == predicted_label:
            scores.append(1.0)
        else:
            scores.append(0.0)

    return sum(scores) / len(scores)

if __name__ == "__main__":

    y_true = [
        "Q36",
        "Q270",
        "Q142",
        "Q183"
    ]

    y_pred = [
        "Q36",
        "Q270",
        None,
        "Q999"
    ]

    print("Accuracy:", accuracy(y_true, y_pred))
    print("Coverage:", coverage(y_pred))
    print("MRR@1:", mrr_at_1(y_true, y_pred))