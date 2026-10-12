
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "BBBP_prepared.csv"
MODEL_PATH = ROOT / "models" / "drugonaut_scaffold.npz"

FINGERPRINT_BITS = 1024
RADIUS = 2
TEST_FRACTION = 0.20
RANDOM_SEED = 42
EPOCHS = 1500
LEARNING_RATE = 0.1
REGULARIZATION = 0.01


def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def get_scaffold_key(molecule):
    """Return a key used to keep scaffold groups together."""
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)

    if scaffold is not None and scaffold.GetNumAtoms() > 0:
        return "SCAFFOLD:" + Chem.MolToSmiles(scaffold)

    # Acyclic molecules have no Murcko ring framework.
    # Group identical canonical structures together.
    return "ACYCLIC:" + Chem.MolToSmiles(molecule)


def calculate_auc(y_true, scores):
    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]

    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")

    greater = (positives[:, None] > negatives[None, :]).mean()
    tied = (positives[:, None] == negatives[None, :]).mean()

    return float(greater + 0.5 * tied)


def calculate_metrics(y_true, scores):
    predictions = (scores >= 0.5).astype(int)

    tn = int(np.sum((y_true == 0) & (predictions == 0)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))
    tp = int(np.sum((y_true == 1) & (predictions == 1)))

    accuracy = float(np.mean(predictions == y_true))

    recall_0 = tn / (tn + fp) if tn + fp else 0.0
    recall_1 = tp / (tp + fn) if tp + fn else 0.0
    balanced_accuracy = (recall_0 + recall_1) / 2

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": calculate_auc(y_true, scores),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if not {"smiles", "p_np"}.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'smiles' and 'p_np' columns."
        )

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=FINGERPRINT_BITS,
    )

    fingerprints = []
    labels = []
    scaffold_keys = []
    skipped = 0

    print("Generating fingerprints and scaffold groups...")

    for _, row in df.iterrows():
        if pd.isna(row["smiles"]) or pd.isna(row["p_np"]):
            skipped += 1
            continue

        molecule = Chem.MolFromSmiles(str(row["smiles"]))

        if molecule is None:
            skipped += 1
            continue

        try:
            label = int(row["p_np"])
        except (ValueError, TypeError):
            skipped += 1
            continue

        if label not in (0, 1):
            skipped += 1
            continue

        fingerprint = generator.GetFingerprint(molecule)

        array = np.zeros(FINGERPRINT_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)

        fingerprints.append(array)
        labels.append(label)
        scaffold_keys.append(get_scaffold_key(molecule))

    X = np.asarray(fingerprints, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    scaffold_keys = np.asarray(scaffold_keys, dtype=object)

    if len(y) < 2:
        raise ValueError("Not enough valid molecules to train.")

    # Each scaffold group must stay entirely in train or test.
    unique_groups = list(dict.fromkeys(scaffold_keys.tolist()))

    if len(unique_groups) < 2:
        raise ValueError("Not enough distinct scaffold groups.")

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(unique_groups)

    group_indices = {
        group: np.flatnonzero(scaffold_keys == group)
        for group in unique_groups
    }

    # Greedily add whole groups to the test set until approximately
    # 20% of the molecules are held out. Never split a group.
    target_test_size = int(round(len(y) * TEST_FRACTION))
    test_groups = []
    test_count = 0

    for group in unique_groups:
        if test_count >= target_test_size:
            break

        test_groups.append(group)
        test_count += len(group_indices[group])

    test_group_set = set(test_groups)

    test_mask = np.array([
        key in test_group_set for key in scaffold_keys
    ])

    test_idx = np.flatnonzero(test_mask)
    train_idx = np.flatnonzero(~test_mask)

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "The scaffold split produced an empty train or test set."
        )

    y_train = y[train_idx]
    y_test = y[test_idx]
    X_train = X[train_idx]
    X_test = X[test_idx]

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "Training split contains only one class. "
            "A reliable binary classifier cannot be trained."
        )

    if len(np.unique(y_test)) < 2:
        raise ValueError(
            "Test split contains only one class, so ROC-AUC and "
            "balanced evaluation may be uninformative. "
            "We need to adjust the group-selection strategy."
        )

    # Class-balanced logistic regression using NumPy only.
    class_counts = np.bincount(y_train, minlength=2)

    sample_weights = np.where(
        y_train == 1,
        len(y_train) / (2 * class_counts[1]),
        len(y_train) / (2 * class_counts[0]),
    )

    weights = np.zeros(FINGERPRINT_BITS, dtype=np.float64)
    bias = 0.0

    print("Training on scaffold-separated molecules...")

    for _ in range(EPOCHS):
        probabilities = sigmoid(X_train @ weights + bias)
        errors = (probabilities - y_train) * sample_weights

        grad_w = X_train.T @ errors / len(y_train)
        grad_w += REGULARIZATION * weights
        grad_b = errors.mean()

        weights -= LEARNING_RATE * grad_w
        bias -= LEARNING_RATE * grad_b

    test_scores = sigmoid(X_test @ weights + bias)
    metrics = calculate_metrics(y_test, test_scores)

    # Check that no scaffold group appears on both sides.
    train_groups = set(scaffold_keys[train_idx].tolist())
    test_groups = set(scaffold_keys[test_idx].tolist())
    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise RuntimeError(
            "Scaffold leakage detected: a group appears in both splits."
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        MODEL_PATH,
        weights=weights,
        bias=bias,
        fingerprint_bits=FINGERPRINT_BITS,
        radius=RADIUS,
        test_fraction=TEST_FRACTION,
        random_seed=RANDOM_SEED,
        train_indices=train_idx,
        test_indices=test_idx,
        y_test=y_test,
        test_scores=test_scores,
        test_predictions=(test_scores >= 0.5).astype(int),
        scaffold_test_groups=np.asarray(
            sorted(test_group_set), dtype=str
        ),
        accuracy=metrics["accuracy"],
        balanced_accuracy=metrics["balanced_accuracy"],
        roc_auc=metrics["roc_auc"],
        tn=metrics["tn"],
        fp=metrics["fp"],
        fn=metrics["fn"],
        tp=metrics["tp"],
    )

    print("\nDRUGONAUT: SCAFFOLD SPLIT EVALUATION")
    print("=" * 44)
    print(f"Valid molecules: {len(y)}")
    print(f"Skipped rows: {skipped}")
    print(f"Distinct scaffold groups: {len(unique_groups)}")
    print(f"Training molecules: {len(train_idx)}")
    print(f"Test molecules: {len(test_idx)}")
    print(f"Training scaffold groups: {len(train_groups)}")
    print(f"Test scaffold groups: {len(test_groups)}")
    print(f"Scaffold overlap: {len(overlap)}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print("Confusion matrix (actual rows, predicted columns):")
    print(np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]],
    ]))
    print(f"\nNew model saved to: {MODEL_PATH}")
    print("The existing random-split model was not overwritten.")


if __name__ == "__main__":
    main()
