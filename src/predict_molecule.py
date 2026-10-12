
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "drugonaut_fingerprint.npz"

FINGERPRINT_BITS = 1024
RADIUS = 2


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Train the fingerprint model first.")
        return

    model = np.load(MODEL_PATH)

    weights = model["weights"]
    bias = float(model["bias"])

    smiles = input("Enter a molecule's SMILES string: ").strip()

    if not smiles:
        print("Please enter a SMILES string.")
        return

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        print("Invalid SMILES string. Please check the structure.")
        return

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=FINGERPRINT_BITS,
    )

    fingerprint = generator.GetFingerprint(molecule)

    features = np.zeros(FINGERPRINT_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, features)

    score = float(features @ weights + bias)
    score = float(np.clip(score, -40, 40))
    model_score = 1 / (1 + np.exp(-score))

    prediction = int(model_score >= 0.5)

    print("\n" + "=" * 40)
    print("         DRUGONAUT")
    print("    BBBP MOLECULE PREDICTOR")
    print("=" * 40)

    print(f"SMILES: {smiles}")
    print(f"Molecular weight: {rdMolDescriptors.CalcExactMolWt(molecule):.2f} Da")

    print(f"\nModel score for class 1: {model_score:.3f}")

    if prediction == 1:
        print("Predicted dataset class: 1 (BBBP-positive)")
    else:
        print("Predicted dataset class: 0 (BBBP-negative)")

    print("\nIMPORTANT:")
    print("This is an experimental model prediction,")
    print("not proof of BBB permeability or drug safety.")
    print("Experimental validation is still required.")


if __name__ == "__main__":
    main()
