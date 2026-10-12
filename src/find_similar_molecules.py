
from pathlib import Path
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "BBBP_prepared.csv"


def main():
    if not DATA_PATH.exists():
        print("Dataset not found:", DATA_PATH)
        return

    smiles = input("Enter a molecule's SMILES string: ").strip()
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        print("Invalid SMILES string.")
        return

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=1024
    )

    query_fp = generator.GetFingerprint(molecule)
    query_canonical = Chem.MolToSmiles(molecule, canonical=True)
    df = pd.read_csv(DATA_PATH)
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
        similarity = DataStructs.TanimotoSimilarity(query_fp, fingerprint)

        matches.append({
            "name": str(row.get("name", "Unknown")),
            "smiles": candidate_smiles,
            "similarity": similarity,
            "label": int(row["p_np"])
        })

    matches.sort(key=lambda item: item["similarity"], reverse=True)

    print("\nDRUGONAUT - SIMILAR MOLECULES")
    print("=" * 40)

    if not matches:
        print("No similar molecules found.")
        return

    for rank, item in enumerate(matches[:5], start=1):
        label = "BBBP-positive" if item["label"] == 1 else "BBBP-negative"
        print(f"\n{rank}. {item['name']}")
        print(f"Similarity: {item['similarity']:.3f}")
        print(f"Dataset label: {label}")
        print(f"SMILES: {item['smiles']}")

    print("\nNote: Similarity and dataset labels do not prove BBB permeability or safety.")


if __name__ == "__main__":
    main()