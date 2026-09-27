"""Bounded-memory tile extraction, frame pooling and resumable feature indexing."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time
from typing import Protocol
import PIL

import numpy as np
import pandas as pd

from activepheno.data.lctp import sha256_file
from activepheno.data.schema import require_strings
from activepheno.data.validation import validate_images
from .cache import cache_key, load_entry, save_entry
from .tiling import TileConfig, coordinates, read_grayscale, tile_batches


class Encoder(Protocol):
    descriptor: dict
    output_dim: int

    def encode(self, tiles: list[np.ndarray]) -> np.ndarray: ...


def validate_observations(observations: pd.DataFrame, raw_root: Path) -> None:
    require_strings(observations, ["observation_id", "image_path", "series_id", "compound_id"])
    if observations.empty or observations.observation_id.duplicated().any():
        raise ValueError("observations must be nonempty with unique observation IDs")
    validate_images(observations, raw_root)


def inspect_images(observations: pd.DataFrame, raw_root: Path, config: TileConfig) -> pd.DataFrame:
    validate_observations(observations, raw_root)
    records = []
    for row in observations.itertuples():
        _, info = read_grayscale(raw_root / row.image_path, config)
        records.append({"observation_id": row.observation_id, **info,
                        "n_tiles": len(coordinates(info["height"], info["width"], config))})
    return pd.DataFrame(records)


def extract(observations: pd.DataFrame, raw_root: Path, output_root: Path,
            config: TileConfig, encoder: Encoder, *, batch_size: int = 8,
            run_metadata: dict | None = None) -> dict:
    validate_observations(observations, raw_root)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".extraction.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another extractor is writing this cache") from exc
        start = time.perf_counter()
        implementation = cache_key({p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
        rows = []
        hits = 0
        for row in observations.sort_values("observation_id").itertuples():
            image = raw_root / row.image_path
            image_hash = sha256_file(image)
            descriptor = {"image_sha256": image_hash, "encoder": encoder.descriptor,
                          "tiling": config.descriptor(), "frame_pooling": "equal_tile_mean-float64-accumulator-v1",
                          "batch_size": batch_size, "cache_version": 1,
                          "implementation_sha256": implementation,
                          "numpy_version": np.__version__, "pillow_version": PIL.__version__}
            key = cache_key(descriptor)
            entry_path = output_root / "entries" / f"{key}.npz"
            cached = load_entry(entry_path, descriptor)
            if cached is not None:
                vector, metadata = cached
                if vector.shape != (encoder.output_dim,):
                    raise ValueError("cached feature dimension does not match encoder")
                provenance = metadata["provenance"]
                hits += 1
            else:
                frame, info = read_grayscale(image, config)
                boxes = coordinates(info["height"], info["width"], config)
                total = np.zeros(encoder.output_dim, dtype=np.float64)
                count = 0
                for batch in tile_batches(frame, config, batch_size):
                    features = np.asarray(encoder.encode(batch))
                    if features.shape != (len(batch), encoder.output_dim) or not np.isfinite(features).all():
                        raise ValueError("encoder returned incorrect shape or nonfinite features")
                    total += features.astype(np.float64).sum(axis=0)
                    count += len(batch)
                vector = (total / count).astype(np.float32)
                if sha256_file(image) != image_hash:
                    raise ValueError(f"image changed during extraction: {image}")
                provenance = {"image": info, "coordinates": boxes, "n_tiles": count}
                save_entry(entry_path, vector, descriptor, provenance)
            rows.append({"observation_id": row.observation_id, "compound_id": row.compound_id,
                         "series_id": row.series_id, "image_path": row.image_path,
                         "cache_key": key, "embedding_path": entry_path.relative_to(output_root).as_posix(),
                         "image_sha256": image_hash, "embedding_dim": encoder.output_dim,
                         "n_tiles": provenance["n_tiles"]})
            if len(rows) % 100 == 0 or len(rows) == len(observations):
                elapsed = time.perf_counter() - start
                print(json.dumps({"progress": len(rows), "total": len(observations),
                                  "cache_hits": hits, "elapsed_seconds": round(elapsed, 2),
                                  "remaining_seconds_estimate": round(elapsed / len(rows) *
                                                                      (len(observations) - len(rows)), 2)}), flush=True)
        index = pd.DataFrame(rows)
        index_path = output_root / "features.parquet"
        temp_index = output_root / "features.parquet.tmp"
        index.to_parquet(temp_index, index=False)
        temp_index.replace(index_path)
        manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "n_observations": len(rows),
                    "cache_hits": hits, "computed_observations": len(rows)-hits,
                    "n_tiles": int(index.n_tiles.sum()), "encoder": encoder.descriptor,
                    "tiling": config.descriptor(), "batch_size": batch_size,
                    "frame_pooling": "equal_tile_mean", "embedding_storage_dtype": "float32",
                    "feature_index_sha256": sha256_file(index_path),
                    "wall_clock_seconds": time.perf_counter()-start,
                    "implementation_sha256": implementation,
                    "runtime": encoder.runtime() if hasattr(encoder, "runtime") else {},
                    "run_metadata": run_metadata or {}}
        temp_manifest = output_root / "extraction_manifest.json.tmp"
        temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        temp_manifest.replace(output_root / "extraction_manifest.json")
        return manifest
