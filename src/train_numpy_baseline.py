
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
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["p_np"].to_numpy(dtype=float)

    # Create a reproducible, stratified 80/20 split.
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

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Standardise using training data only.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Give both classes equal total training weight.
    n = len(y_train)
    class_counts = np.bincount(y_train.astype(int))
    sample_weights = np.where(
        y_train == 1,
        n / (2 * class_counts[1]),
        n / (2 * class_counts[0]),
    )

    # Logistic regression trained with gradient descent.
    weights = np.zeros(X_train.shape[1])
    bias = 0.0
    learning_rate = 0.05
    epochs = 3000

    for _ in range(epochs):
        probabilities = sigmoid(X_train @ weights + bias)
        errors = (probabilities - y_train) * sample_weights

        grad_w = X_train.T @ errors / n + 0.001 * weights
        grad_b = errors.mean()

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    probabilities = sigmoid(X_test @ weights + bias)
    predictions = (probabilities >= 0.5).astype(int)

    cm = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y_test.astype(int), predictions):
        cm[actual, predicted] += 1

    recalls = []
    for label in (0, 1):
        mask = y_test == label
        recalls.append(np.mean(predictions[mask] == label))

    accuracy = np.mean(predictions == y_test)
    balanced_accuracy = np.mean(recalls)

    print("DRUGONAUT: NUMPY BASELINE")
    print("-" * 35)
    print(f"Training molecules: {len(y_train)}")
    print(f"Test molecules: {len(y_test)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Balanced accuracy: {balanced_accuracy:.3f}")
    print("\nConfusion matrix (actual rows, predicted columns):")
    print(cm)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        MODEL_PATH,
        weights=weights,
        bias=bias,
        mean=mean,
        std=std,
        features=np.array(FEATURES),
    )
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
