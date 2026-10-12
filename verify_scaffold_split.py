from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "BBBP_prepared.csv"
MODEL_PATH = ROOT / "models" / "drugonaut_scaffold.npz"

TEST_FRACTION = 0.20
RANDOM_SEED = 42


def get_scaffold_key(molecule):
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold is not None and scaffold.GetNumAtoms() > 0:
        return "SCAFFOLD:" + Chem.MolToSmiles(scaffold)
    return "ACYCLIC:" + Chem.MolToSmiles(molecule)


def main():
    print("DRUGONAUT SCAFFOLD SPLIT REPRODUCIBILITY CHECK")
    print("=" * 50)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)
    saved = np.load(MODEL_PATH)
    scaffold_keys = []

    for _, row in df.iterrows():
        if pd.isna(row["smiles"]) or pd.isna(row["p_np"]):
            continue
        molecule = Chem.MolFromSmiles(str(row["smiles"]))
        if molecule is None:
            continue
        try:
            label = int(row["p_np"])
        except (ValueError, TypeError):
            continue
        if label not in (0, 1):
            continue
        scaffold_keys.append(get_scaffold_key(molecule))

    scaffold_keys = np.asarray(scaffold_keys, dtype=object)
    n = len(scaffold_keys)

    unique_groups = list(dict.fromkeys(scaffold_keys.tolist()))
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)

    target_test_count = round(n * TEST_FRACTION)
    test_groups = set()
    test_count = 0

    for group in shuffled_groups:
        if test_count >= target_test_count:
            break
        test_groups.add(group)
        test_count += int(np.sum(scaffold_keys == group))

    reproduced_test_idx = np.flatnonzero(
        np.isin(scaffold_keys, list(test_groups))
    )
    reproduced_train_idx = np.flatnonzero(
        ~np.isin(scaffold_keys, list(test_groups))
    )

    saved_test_idx = saved["test_indices"].astype(int)
    saved_train_idx = saved["train_indices"].astype(int)

    same_test = np.array_equal(reproduced_test_idx, saved_test_idx)
    same_train = np.array_equal(reproduced_train_idx, saved_train_idx)

    test_scaffolds = set(scaffold_keys[reproduced_test_idx].tolist())
    train_scaffolds = set(scaffold_keys[reproduced_train_idx].tolist())
    overlap = test_scaffolds.intersection(train_scaffolds)

    print("Valid molecules:", n)
    print("Unique scaffold groups:", len(unique_groups))
    print("Saved training rows:", len(saved_train_idx))
    print("Recreated training rows:", len(reproduced_train_idx))
    print("Saved test rows:", len(saved_test_idx))
    print("Recreated test rows:", len(reproduced_test_idx))
    print()
    print("Training indices reproduced:", same_train)
    print("Test indices reproduced:", same_test)
    print("Recreated scaffold overlap:", len(overlap))

    if same_train and same_test and len(overlap) == 0:
        print("\nPASS: The saved scaffold split was reproduced.")
    else:
        print("\nNOT A FULL MATCH: The recreated split differs.")
        print("Check the original training script and dataset order.")

    print("\nNo model or dataset files were modified.")


if __name__ == "__main__":
    main()
