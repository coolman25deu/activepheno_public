"""Frozen MoLFormer embeddings from caller-supplied, resolved SMILES."""

from __future__ import annotations

import numpy as np


def canonicalize_smiles(smiles: str, *, keep_stereochemistry: bool = True) -> str:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=keep_stereochemistry)


def encode_smiles(
    smiles: list[str], *, model_id: str, revision: str, cache_dir: str,
    batch_size: int = 16, device: str = "cpu", allow_download: bool = False,
) -> np.ndarray:
    """Encode with a pinned Hugging Face checkpoint; networking is opt-in.

    This model family loads repository code with ``trust_remote_code=True``.
    Review the code and checkpoint license before use.
    """
    if not revision or batch_size < 1 or not smiles:
        raise ValueError("A pinned revision, positive batch size, and SMILES are required")
    import torch
    from transformers import AutoModel, AutoTokenizer

    local_only = not allow_download
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, cache_dir=cache_dir,
        trust_remote_code=True, local_files_only=local_only,
    )
    model = AutoModel.from_pretrained(
        model_id, revision=revision, cache_dir=cache_dir,
        trust_remote_code=True, local_files_only=local_only,
    )
    model.eval().to(device)
    canonical = [canonicalize_smiles(value) for value in smiles]
    vectors = []
    with torch.inference_mode():
        for offset in range(0, len(canonical), batch_size):
            batch = tokenizer(canonical[offset:offset + batch_size], padding=True, truncation=False, return_tensors="pt")
            output = model(**{key: value.to(device) for key, value in batch.items()})
            pooled = getattr(output, "pooler_output", None)
            if pooled is None:
                raise ValueError("Checkpoint does not expose pooler_output")
            if not torch.isfinite(pooled).all():
                raise ValueError("MoLFormer returned non-finite embeddings")
            vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
    result = np.concatenate(vectors, axis=0)
    if result.ndim != 2 or result.shape[0] != len(smiles):
        raise ValueError("Unexpected embedding matrix shape")
    return result
