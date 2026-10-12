
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


def analyze_molecule():
    print("\n" + "=" * 42)
    print("       DRUGONAUT MOLECULAR ANALYSIS")
    print("=" * 42)

    smiles = input("\nEnter a molecule's SMILES string: ").strip()

    if not smiles:
        print("Please enter a SMILES string.")
        return

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        print("Invalid SMILES string. Please check the structure.")
        return

    # Calculate molecular properties
    molecular_weight = Descriptors.MolWt(molecule)
    logp = Descriptors.MolLogP(molecule)
    tpsa = rdMolDescriptors.CalcTPSA(molecule)
    hbd = Lipinski.NumHDonors(molecule)
    hba = Lipinski.NumHAcceptors(molecule)
    rotatable_bonds = Lipinski.NumRotatableBonds(molecule)
    rings = rdMolDescriptors.CalcNumRings(molecule)

    print("\n--- MOLECULAR PROPERTIES ---")
    print(f"Molecular weight:       {molecular_weight:.2f} g/mol")
    print(f"LogP:                   {logp:.2f}")
    print(f"TPSA:                   {tpsa:.2f} Å²")
    print(f"Hydrogen-bond donors:   {hbd}")
    print(f"Hydrogen-bond acceptors: {hba}")
    print(f"Rotatable bonds:        {rotatable_bonds}")
    print(f"Ring count:             {rings}")

    print("\n--- WHAT THESE MEAN ---")

    print(
        "\nMolecular weight:\n"
        "The estimated mass of the molecule."
    )

    print(
        "\nLogP:\n"
        "Estimates how the neutral molecule partitions between "
        "octanol and water. It is not a direct measurement of "
        "brain penetration."
    )

    print(
        "\nTPSA:\n"
        "Estimates the polar surface area. Polarity can influence "
        "membrane permeability."
    )

    print(
        "\nHydrogen-bond donors and acceptors:\n"
        "Describe ways the molecule can participate in hydrogen "
        "bonding, which can affect interactions with water and membranes."
    )

    print(
        "\nRotatable bonds:\n"
        "Approximate the number of rotatable bonds and give a "
        "rough indication of flexibility."
    )

    print(
        "\nRing count:\n"
        "Reports the number of rings detected in the molecular structure."
    )

    print("\n--- IMPORTANT LIMITATION ---")
    print(
        "These are computed properties, not experimental measurements."
    )
    print(
        "No single property can confirm blood-brain barrier permeability."
    )
    print(
        "Consider the model prediction alongside experimental evidence."
    )


if __name__ == "__main__":
    analyze_molecule()
