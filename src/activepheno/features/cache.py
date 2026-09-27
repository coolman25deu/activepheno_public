"""Atomic cache entries with content and configuration integrity checks."""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np


def cache_key(descriptor: dict) -> str:
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_entry(path: Path, descriptor: dict) -> tuple[np.ndarray, dict] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as entry:
        vector = entry["embedding"]
        metadata = json.loads(str(entry["metadata"].item()))
    if metadata["descriptor"] != descriptor or vector.ndim != 1 or vector.dtype != np.float32 or not np.isfinite(vector).all():
        raise ValueError(f"invalid cache entry: {path}")
    if hashlib.sha256(vector.tobytes()).hexdigest() != metadata["embedding_sha256"]:
        raise ValueError(f"cache checksum mismatch: {path}")
    return vector, metadata


def save_entry(path: Path, vector: np.ndarray, descriptor: dict, provenance: dict) -> None:
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("embedding must be a finite vector")
    metadata = {"descriptor": descriptor, "provenance": provenance,
                "embedding_sha256": hashlib.sha256(vector.tobytes()).hexdigest()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez(handle, embedding=vector, metadata=json.dumps(metadata, sort_keys=True))
            handle.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
