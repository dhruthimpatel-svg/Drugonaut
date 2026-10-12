
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "BBBP_prepared.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "drugonaut_baseline.joblib"

FEATURES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "RingCount",
]


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df["p_np"]

    # Keep the class proportions similar in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("Training Drugonaut's first model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nBASELINE MODEL RESULTS")
    print("-" * 40)
    print(f"Training molecules: {len(X_train)}")
    print(f"Test molecules: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_test, predictions):.3f}"
    )
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["BBB-negative", "BBB-positive"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURES},
        MODEL_PATH,
    )
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
