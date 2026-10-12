
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "BBBP.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "BBBP_prepared.csv"


def calculate_features(smiles):
    """Convert a SMILES string into numerical molecular features."""
    molecule = Chem.MolFromSmiles(str(smiles))

    if molecule is None:
        return None

    return {
        "MolWt": Descriptors.MolWt(molecule),
        "MolLogP": Descriptors.MolLogP(molecule),
        "TPSA": rdMolDescriptors.CalcTPSA(molecule),
        "HBD": Lipinski.NumHDonors(molecule),
        "HBA": Lipinski.NumHAcceptors(molecule),
        "RotatableBonds": Lipinski.NumRotatableBonds(molecule),
        "RingCount": rdMolDescriptors.CalcNumRings(molecule),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {"smiles", "p_np"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain: {required}")

    records = []

    for _, row in df.iterrows():
        features = calculate_features(row["smiles"])

        if features is None or pd.isna(row["p_np"]):
            continue

        records.append({
            "name": row.get("name", ""),
            "smiles": row["smiles"],
            "p_np": int(row["p_np"]),
            **features,
        })

    prepared = pd.DataFrame(records)

    if prepared.empty:
        raise ValueError("No valid molecules were found.")

    prepared.to_csv(OUTPUT_PATH, index=False)

    print("DRUGONAUT: DATA PREPARATION COMPLETE")
    print("-" * 42)
    print(f"Original records: {len(df)}")
    print(f"Prepared records: {len(prepared)}")
    print(f"Removed records: {len(df) - len(prepared)}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("\nClass counts:")
    print(prepared["p_np"].value_counts().sort_index())
    print("\nFeature columns:")
    print(list(features.keys()))


if __name__ == "__main__":
    main()
