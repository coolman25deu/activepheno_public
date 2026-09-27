"""Deterministic Morgan fingerprints for canonical SMILES."""

from collections.abc import Iterable

import numpy as np


def morgan_fingerprints(smiles: Iterable[str], *, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Return a float32 bit matrix. RDKit is loaded only when this is called."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    values = list(smiles)
    if not values or radius < 0 or n_bits < 1:
        raise ValueError("SMILES must be nonempty and fingerprint settings valid")
    molecules = [Chem.MolFromSmiles(value) for value in values]
    if any(molecule is None for molecule in molecules):
        raise ValueError("Invalid SMILES supplied for Morgan fingerprints")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.stack([generator.GetFingerprintAsNumPy(mol) for mol in molecules]).astype(np.float32)
