
from pathlib import Path
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "BBBP_prepared.csv"
MODEL_PATH = ROOT / "models" / "drugonaut_fingerprint.npz"

RADIUS = 2
FINGERPRINT_BITS = 1024
TOP_K = 5


def load_model():
    if not MODEL_PATH.exists():
        print("Model not found:", MODEL_PATH)
        return None

    with np.load(MODEL_PATH) as saved:
        required = {"weights", "bias"}
        if not required.issubset(saved.files):
            print("Model file is missing required weights or bias.")
            print("Available keys:", saved.files)
            return None

        return {
            "weights": saved["weights"].copy(),
            "bias": float(saved["bias"]),
        }


def predict_bbb(molecule, model, generator):
    fingerprint = generator.GetFingerprint(molecule)
    features = np.zeros(FINGERPRINT_BITS, dtype=np.float64)
    DataStructs.ConvertToNumpyArray(fingerprint, features)

    if features.shape != model["weights"].shape:
        raise ValueError(
            "Fingerprint size does not match the trained model. "
            "Check the model's training script."
        )

    score = float(features @ model["weights"] + model["bias"])

    if score >= 0:
        probability = 1.0 / (1.0 + np.exp(-score))
    else:
        exp_score = np.exp(score)
        probability = exp_score / (1.0 + exp_score)

    predicted_class = 1 if probability >= 0.5 else 0
    return score, probability, predicted_class


def analyze_properties(molecule):
    return {
        "Molecular weight": f"{Descriptors.MolWt(molecule):.2f} g/mol",
        "Exact molecular weight": (
            f"{rdMolDescriptors.CalcExactMolWt(molecule):.2f} Da"
        ),
        "LogP": f"{Descriptors.MolLogP(molecule):.2f}",
        "TPSA": f"{rdMolDescriptors.CalcTPSA(molecule):.2f} Å²",
        "Hydrogen bond donors": Lipinski.NumHDonors(molecule),
        "Hydrogen bond acceptors": Lipinski.NumHAcceptors(molecule),
        "Rotatable bonds": Lipinski.NumRotatableBonds(molecule),
        "Ring count": rdMolDescriptors.CalcNumRings(molecule),
    }


def find_similar(molecule, generator):
    if not DATA_PATH.exists():
        print("Dataset not found:", DATA_PATH)
        return []

    df = pd.read_csv(DATA_PATH)
    query_fp = generator.GetFingerprint(molecule)
    query_canonical = Chem.MolToSmiles(molecule, canonical=True)
    matches = []

    for _, row in df.iterrows():
        candidate_smiles = str(row["smiles"])
        candidate = Chem.MolFromSmiles(candidate_smiles)

        if candidate is None:
            continue

        canonical = Chem.MolToSmiles(candidate, canonical=True)

        if canonical == query_canonical:
            continue

        fingerprint = generator.GetFingerprint(candidate)
        similarity = DataStructs.TanimotoSimilarity(
            query_fp, fingerprint
        )

        matches.append({
            "name": str(row.get("name", "Unknown")),
            "similarity": similarity,
            "label": int(row["p_np"]),
            "smiles": candidate_smiles,
        })

    matches.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )
    return matches[:TOP_K]


def main():
    print("\n" + "=" * 50)
    print("             DRUGONAUT")
    print("     Molecular Analysis & BBB Research")
    print("=" * 50)

    if not DATA_PATH.exists():
        print("Dataset not found:", DATA_PATH)
        return

    model = load_model()
    if model is None:
        return

    smiles = input("\nEnter a molecule's SMILES string: ").strip()
    if not smiles:
        print("Please enter a SMILES string.")
        return

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        print("Invalid SMILES. Please check your input.")
        return

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=FINGERPRINT_BITS,
    )

    score, probability, predicted_class = predict_bbb(
        molecule, model, generator
    )
    properties = analyze_properties(molecule)
    similar = find_similar(molecule, generator)

    print("\n[1] MOLECULE")
    print("Canonical SMILES:", Chem.MolToSmiles(molecule, canonical=True))

    print("\n[2] BBBP MODEL PREDICTION")
    print(
        "Predicted dataset class:",
        f"{predicted_class} "
        f"({'BBBP-positive' if predicted_class == 1 else 'BBBP-negative'})",
    )
    print(f"Raw model score: {score:.4f}")
    print(f"Sigmoid output (not calibrated): {probability:.3f}")

    print("\n[3] MOLECULAR PROPERTIES")
    for name, value in properties.items():
        print(f"{name}: {value}")

    print("\n[4] TOP SIMILAR MOLECULES")
    if not similar:
        print("No similar molecules found.")
    else:
        for rank, item in enumerate(similar, start=1):
            label = (
                "BBBP-positive"
                if item["label"] == 1
                else "BBBP-negative"
            )
            print(f"\n{rank}. {item['name']}")
            print(f"   Similarity: {item['similarity']:.3f}")
            print(f"   Dataset label: {label}")
            print(f"   SMILES: {item['smiles']}")

    print("\nIMPORTANT")
    print("These are research predictions, not experimental proof.")
    print("Similarity does not establish BBB permeability, safety, or effectiveness.")


if __name__ == "__main__":
    main()
