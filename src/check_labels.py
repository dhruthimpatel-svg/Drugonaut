
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "BBBP.csv"

df = pd.read_csv(DATA_PATH)

print("DATASET SUMMARY")
print("-" * 35)

print("Rows:", len(df))
print("Columns:", list(df.columns))

print("\nBBB label counts:")
print(df["p_np"].value_counts(dropna=False).sort_index())

print("\nMissing values:")
print(df[["p_np", "smiles"]].isnull().sum())

print("\nDuplicate SMILES:", df["smiles"].duplicated().sum())
