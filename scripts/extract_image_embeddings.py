"""Inspect real frames or extract frozen segmentation-free DINO features."""
import argparse
from pathlib import Path
import json

import pandas as pd
import yaml

from activepheno.data.lctp import sha256_file
from activepheno.features.extraction import extract, inspect_images
from activepheno.features.tiling import TileConfig
from activepheno.seed import seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--revision", help="immutable 40-character model commit hash")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="explicit pilot subset, sorted by observation_id")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="decode and inspect tiles without loading models or writing outputs")
    args = parser.parse_args()
    try:
        config = yaml.safe_load(args.config.read_text())
        allowed = {"model_id", "revision", "token_pooling", "tiling"}
        if not isinstance(config, dict) or set(config) - allowed or not isinstance(config.get("model_id"), str):
            raise ValueError("invalid feature config; expected model_id, revision, token_pooling, tiling")
        tile_config = TileConfig(**config.get("tiling", {}))
        if tile_config.tile_size % 14 or tile_config.tile_size % 16:
            raise ValueError("matched DINOv2/v3 tile_size must be divisible by both 14 and 16")
        revision = args.revision or config.get("revision")
        print("Resolved config:", json.dumps({**vars(args), "features": config, "revision": revision}, default=str, sort_keys=True))
        seed_everything(args.seed)
        observations = pd.read_parquet(args.observations).sort_values("observation_id")
        full_count = len(observations)
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("limit must be positive")
            observations = observations.head(args.limit)
        if args.dry_run:
            info = inspect_images(observations, args.raw_root, tile_config)
            print(info.to_json(orient="records", indent=2))
            print(f"Inspected {len(info)}/{full_count} observations; {info.n_tiles.sum()} tiles; no model loaded.")
            return 0
        if args.output_root is None:
            raise ValueError("--output-root is required for extraction")
        from activepheno.features.image_encoder import FrozenDinoEncoder
        encoder = FrozenDinoEncoder(config["model_id"], revision, device=args.device,
            precision=args.precision, token_pooling=config.get("token_pooling", "patch_mean"),
            local_files_only=args.local_files_only)
        metadata = {"seed": args.seed, "observations_sha256": sha256_file(args.observations),
                    "config_sha256": sha256_file(args.config), "source_observations": full_count,
                    "selected_observations": len(observations), "limit": args.limit}
        result = extract(observations, args.raw_root, args.output_root, tile_config, encoder,
                         batch_size=args.batch_size, run_metadata=metadata)
        print(json.dumps({k: result[k] for k in ["n_observations", "n_tiles", "cache_hits", "computed_observations", "wall_clock_seconds"]}, indent=2))
        return 0
    except (ValueError, OSError, TypeError, KeyError, ImportError, yaml.YAMLError) as exc:
        parser.exit(1, f"Image extraction failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
