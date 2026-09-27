#!/usr/bin/env python3
"""Extract frozen MoLFormer features for a resolved compound table."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from activepheno.chemistry.molformer import canonicalize_smiles, encode_smiles
from activepheno.utils.provenance import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compounds", type=Path, required=True, help="parquet with compound_id and canonical_smiles")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="ibm-research/MoLFormer-XL-both-10pct")
    parser.add_argument("--revision", required=True, help="pin a reviewed model revision")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-download", action="store_true", help="allow Hugging Face network access")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    frame = pd.read_parquet(args.compounds).sort_values("compound_id").reset_index(drop=True)
    if not {"compound_id", "canonical_smiles"}.issubset(frame) or frame.compound_id.duplicated().any():
        raise ValueError("Compound table requires unique IDs and canonical_smiles")
    if frame.canonical_smiles.isna().any():
        raise ValueError("All compounds need resolved SMILES")
    model_smiles = [canonicalize_smiles(value) for value in frame.canonical_smiles.astype(str)]
    matrix = encode_smiles(model_smiles, model_id=args.model_id, revision=args.revision,
                           cache_dir=str(args.cache_dir), batch_size=args.batch_size,
                           device=args.device, allow_download=args.allow_download)
    args.output.mkdir(parents=True)
    index = pd.DataFrame({"matrix_row": np.arange(len(frame), dtype=np.int32),
                          "compound_id": frame.compound_id.astype(str),
                          "model_smiles": model_smiles,
                          "model_smiles_sha256": [hashlib.sha256(s.encode()).hexdigest() for s in model_smiles]})
    index.to_parquet(args.output / "compound_index.parquet", index=False)
    np.save(args.output / "embeddings.npy", matrix, allow_pickle=False)
    manifest = {"model_id": args.model_id, "revision": args.revision,
                "remote_code_trust_enabled": True, "network_download_allowed": args.allow_download,
                "n_compounds": len(frame), "embedding_shape": list(matrix.shape),
                "source_sha256": sha256_file(args.compounds),
                "index_sha256": sha256_file(args.output / "compound_index.parquet"),
                "embeddings_sha256": sha256_file(args.output / "embeddings.npy")}
    (args.output / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote embeddings for {len(frame)} compounds to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
