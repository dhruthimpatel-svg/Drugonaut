
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Locate the dataset from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "BBBP.csv"


def main():
    print("=" * 50)
    print("       DRUGONAUT: DATASET INSPECTOR")
    print("=" * 50)

    # 1. Check whether the dataset exists
    if not DATA_PATH.exists():
        print(f"\nDataset not found: {DATA_PATH}")
        print("Please download BBBP.csv into the data folder.")
        return

    # 2. Load the dataset
    df = pd.read_csv(DATA_PATH)

    print(f"\nTotal rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nColumn names: {list(df.columns)}")

    # 3. Display a few example molecules
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

    # 4. Check missing values
    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())

    # 5. Check duplicate rows
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    # 6. Inspect the target labels
    if "p_np" in df.columns:
        print("\nBBB permeability label counts:")
        print(df["p_np"].value_counts(dropna=False).to_string())

    # 7. Check whether RDKit can read the SMILES strings
    if "smiles" not in df.columns:
        print("\nNo 'smiles' column found. Check the dataset columns.")
        return

    valid_count = 0
    invalid_count = 0
    molecular_weights = []

    for smiles in df["smiles"].dropna():
        molecule = Chem.MolFromSmiles(str(smiles))

        if molecule is None:
            invalid_count += 1
            continue

        valid_count += 1
        molecular_weights.append(
            Descriptors.MolWt(molecule)
        )

    print("\nChemical structure checks:")
    print(f"Valid SMILES: {valid_count}")
    print(f"Invalid SMILES: {invalid_count}")

    if molecular_weights:
        print(
            "Average molecular weight: "
            f"{sum(molecular_weights) / len(molecular_weights):.2f} g/mol"
        )

    print("\nDataset inspection complete!")


if __name__ == "__main__":
    main()
