
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "BBBP_prepared.csv"
MODEL_PATH = ROOT / "models" / "drugonaut_fingerprint.npz"

FINGERPRINT_BITS = 1024
RADIUS = 2


def main():
    df = pd.read_csv(DATA_PATH)

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=FINGERPRINT_BITS,
    )

    fingerprints = []
    labels = []
    skipped = 0

    print("Generating molecular fingerprints...")

    for _, row in df.iterrows():
        molecule = Chem.MolFromSmiles(str(row["smiles"]))

        if molecule is None or pd.isna(row["p_np"]):
            skipped += 1
            continue

        fingerprint = generator.GetFingerprint(molecule)

        array = np.zeros(FINGERPRINT_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)

        fingerprints.append(array)
        labels.append(int(row["p_np"]))

    X = np.asarray(fingerprints, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    # Reproduce the same stratified split as the descriptor baseline.
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

    # Balanced logistic regression using NumPy gradient descent.
    class_counts = np.bincount(y_train, minlength=2)
    sample_weights = np.where(
        y_train == 1,
        len(y_train) / (2 * class_counts[1]),
        len(y_train) / (2 * class_counts[0]),
    )

    weights = np.zeros(FINGERPRINT_BITS, dtype=np.float64)
    bias = 0.0
    learning_rate = 0.1
    epochs = 1500
    regularization = 0.01

    def sigmoid(z):
        z = np.clip(z, -40, 40)
        return 1 / (1 + np.exp(-z))

    print("Training fingerprint model...")

    for _ in range(epochs):
        probabilities = sigmoid(X_train @ weights + bias)
        errors = (probabilities - y_train) * sample_weights

        grad_w = X_train.T @ errors / len(y_train)
        grad_w += regularization * weights
        grad_b = errors.mean()

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    probabilities = sigmoid(X_test @ weights + bias)
    predictions = (probabilities >= 0.5).astype(int)

    cm = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y_test, predictions):
        cm[actual, predicted] += 1

    tn, fp = cm[0]
    fn, tp = cm[1]

    def divide(a, b):
        return a / b if b else 0.0

    recall_negative = divide(tn, tn + fp)
    recall_positive = divide(tp, tp + fn)
    balanced_accuracy = (recall_negative + recall_positive) / 2
    accuracy = np.mean(predictions == y_test)

    positives = probabilities[y_test == 1]
    negatives = probabilities[y_test == 0]

    if len(positives) and len(negatives):
        greater = (positives[:, None] > negatives[None, :]).mean()
        tied = (positives[:, None] == negatives[None, :]).mean()
        roc_auc = greater + 0.5 * tied
    else:
        roc_auc = float("nan")

    print("\nDRUGONAUT: FINGERPRINT BASELINE")
    print("=" * 42)
    print(f"Molecules used: {len(y)}")
    print(f"Invalid/missing rows skipped: {skipped}")
    print(f"Fingerprint size: {FINGERPRINT_BITS} bits")
    print(f"Training molecules: {len(y_train)}")
    print(f"Test molecules: {len(y_test)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Balanced accuracy: {balanced_accuracy:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print("\nConfusion matrix (actual rows, predicted columns):")
    print(cm)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        MODEL_PATH,
        weights=weights,
        bias=bias,
        fingerprint_bits=FINGERPRINT_BITS,
        radius=RADIUS,
    )
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
