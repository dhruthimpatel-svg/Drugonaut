
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "BBBP_prepared.csv"
MODEL_PATH = ROOT / "models" / "drugonaut_numpy.npz"

FEATURES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "RingCount",
]


def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1 / (1 + np.exp(-z))


def main():
    df = pd.read_csv(DATA_PATH)
    saved = np.load(MODEL_PATH)

    # Re-create the exact same test split used during training.
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["p_np"].to_numpy(dtype=int)

    rng = np.random.default_rng(42)
    train_parts = []
    test_parts = []

    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        split = int(len(indices) * 0.8)
        train_parts.append(indices[:split])
        test_parts.append(indices[split:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_test = X[test_idx]
    y_test = y[test_idx]

    # Use scaling learned from the training set.
    mean = saved["mean"]
    std = saved["std"]
    X_test = (X_test - mean) / std

    probabilities = sigmoid(X_test @ saved["weights"] + float(saved["bias"]))
    predictions = (probabilities >= 0.5).astype(int)

    # Confusion matrix: rows = actual, columns = predicted.
    cm = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y_test, predictions):
        cm[actual, predicted] += 1

    tn, fp = cm[0]
    fn, tp = cm[1]

    def divide(a, b):
        return a / b if b else 0.0

    accuracy = divide(tp + tn, len(y_test))
    precision_positive = divide(tp, tp + fp)
    recall_positive = divide(tp, tp + fn)
    f1_positive = divide(
        2 * precision_positive * recall_positive,
        precision_positive + recall_positive,
    )

    precision_negative = divide(tn, tn + fn)
    recall_negative = divide(tn, tn + fp)
    f1_negative = divide(
        2 * precision_negative * recall_negative,
        precision_negative + recall_negative,
    )

    balanced_accuracy = (recall_positive + recall_negative) / 2

    # A simple baseline that always predicts the majority class.
    majority_prediction = int(np.bincount(y[test_idx]).argmax())
    majority_accuracy = np.mean(y_test == majority_prediction)

    # ROC-AUC from scratch, using positive-class probabilities.
    positives = probabilities[y_test == 1]
    negatives = probabilities[y_test == 0]

    if len(positives) and len(negatives):
        comparisons = (
            positives[:, None] > negatives[None, :]
        ).mean()
        ties = (
            positives[:, None] == negatives[None, :]
        ).mean()
        roc_auc = comparisons + 0.5 * ties
    else:
        roc_auc = float("nan")

    print("DRUGONAUT: BASELINE EVALUATION")
    print("=" * 40)
    print(f"Test molecules: {len(y_test)}")
    print(f"Accuracy:          {accuracy:.3f}")
    print(f"Balanced accuracy: {balanced_accuracy:.3f}")
    print(f"ROC-AUC:           {roc_auc:.3f}")
    print(f"Majority baseline accuracy: {majority_accuracy:.3f}")

    print("\nClass-specific metrics")
    print("-" * 40)
    print(
        f"BBB-negative | Precision: {precision_negative:.3f} "
        f"| Recall: {recall_negative:.3f} | F1: {f1_negative:.3f}"
    )
    print(
        f"BBB-positive | Precision: {precision_positive:.3f} "
        f"| Recall: {recall_positive:.3f} | F1: {f1_positive:.3f}"
    )

    print("\nConfusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
