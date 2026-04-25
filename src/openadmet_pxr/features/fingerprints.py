"""Molecular featurizers: Morgan fingerprints and RDKit 2D descriptors."""

from __future__ import annotations

import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

_DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]
_DESCRIPTOR_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_DESCRIPTOR_NAMES)


def morgan_fingerprints(
    smiles: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    result = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        result[i] = np.array(fp)
    return result


def rdkit_2d_descriptors(
    smiles: list[str],
    normalize: bool = True,
) -> tuple[np.ndarray, list[str]]:
    n = len(smiles)
    n_desc = len(_DESCRIPTOR_NAMES)
    result = np.full((n, n_desc), np.nan, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            result[i] = _DESCRIPTOR_CALC.CalcDescriptors(mol)

    result = np.where(np.isinf(result), np.nan, result)
    col_means = np.nanmean(result, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(result)
    result[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    if normalize:
        col_std = np.std(result, axis=0)
        col_std = np.where(col_std == 0, 1.0, col_std)
        result = (result - col_means) / col_std

    return result.astype(np.float32), _DESCRIPTOR_NAMES


def combined_features(
    smiles: list[str],
    use_morgan: bool = True,
    use_rdkit_2d: bool = True,
    morgan_radius: int = 2,
    morgan_n_bits: int = 2048,
) -> np.ndarray:
    parts = []
    if use_morgan:
        parts.append(morgan_fingerprints(smiles, radius=morgan_radius, n_bits=morgan_n_bits))
    if use_rdkit_2d:
        desc, _ = rdkit_2d_descriptors(smiles, normalize=True)
        parts.append(desc)
    if not parts:
        raise ValueError("At least one featurizer must be enabled.")
    return np.concatenate(parts, axis=1)
