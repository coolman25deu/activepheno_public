#!/usr/bin/env python3
"""Train simple MoA baselines on one measured timepoint and frozen features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from activepheno.chemistry.fingerprints import morgan_fingerprints
from activepheno.data.lctp import sha256_file
from activepheno.evaluation.metrics import classification_metrics, compound_bootstrap_ci
from activepheno.evaluation.temporal import apply_temperature, cumulative_compound_mean, fit_temperature
from activepheno.features.cache import cache_key, load_entry
from activepheno.models.classification import fit_classifier


def load_image_observations(features_root: Path, observations: pd.DataFrame, split: pd.DataFrame, hour: float):
    manifest_path = features_root / "extraction_manifest.json"
    index_path = features_root / "features.parquet"
    manifest = json.loads(manifest_path.read_text())
    if sha256_file(index_path) != manifest["feature_index_sha256"]:
        raise ValueError("Feature index checksum mismatch")
    index = pd.read_parquet(index_path)
    if index.observation_id.duplicated().any():
        raise ValueError("Duplicate observation in feature index")
    selected = observations.merge(split[["series_id", "compound_id", "split"]],
                                  on=["series_id", "compound_id"], validate="many_to_one")
    selected = selected.loc[selected.time_hours.eq(hour)].sort_values("observation_id").copy()
    if selected.empty:
        raise ValueError(f"No measured observations at {hour}h")
    if selected.observation_id.duplicated().any():
        raise ValueError("Duplicate measured time point")
    joined = selected.merge(index, on="observation_id", validate="one_to_one", suffixes=("", "_feature"))
    for column in ["series_id", "compound_id", "image_path"]:
        if not joined[column].eq(joined[f"{column}_feature"]).all():
            raise ValueError(f"Feature identity mismatch: {column}")
    vectors = []
    for row in joined.itertuples():
        path = (features_root / row.embedding_path).resolve()
        if not path.is_relative_to(features_root.resolve()):
            raise ValueError("Unsafe embedding cache path")
        with np.load(path, allow_pickle=False) as entry:
            descriptor = json.loads(str(entry["metadata"].item()))["descriptor"]
        if cache_key(descriptor) != row.cache_key or descriptor["image_sha256"] != row.image_sha256:
            raise ValueError("Embedding cache identity mismatch")
        vector, _ = load_entry(path, descriptor)
        vectors.append(vector)
    matrix = np.stack(vectors)
    joined["matrix_row"] = np.arange(len(joined), dtype=np.int64)
    return joined, matrix, manifest


def evaluate_model(name: str, x: np.ndarray, frame: pd.DataFrame, classes: tuple[str, ...], seed: int, draws: int):
    train_mask = frame.split.eq("train").to_numpy()
    val_mask = frame.split.eq("validation").to_numpy()
    test_mask = frame.split.eq("test").to_numpy()
    model, selected_classes, trials = fit_classifier(
        x[train_mask], frame.loc[train_mask], x[val_mask], frame.loc[val_mask], seed=seed)
    if selected_classes != classes:
        raise ValueError("Model class order changed")
    mapping = {label: index for index, label in enumerate(classes)}
    y_val = frame.loc[val_mask, "moa_id"].map(mapping).to_numpy()
    y_test = frame.loc[test_mask, "moa_id"].map(mapping).to_numpy()
    val_logits = np.log(np.clip(model.predict_proba(x[val_mask]), 1e-12, 1))
    temperature = fit_temperature(val_logits, y_val)
    test_logits = np.log(np.clip(model.predict_proba(x[test_mask]), 1e-12, 1))
    probability = apply_temperature(test_logits, temperature)
    result = classification_metrics(y_test, probability)
    result.update(model=name, selected_C=float(model.named_steps["logisticregression"].C if hasattr(model, "named_steps") else model.C),
                  validation_temperature=temperature, classes=list(classes),
                  ci95=compound_bootstrap_ci(y_test, probability, seed=seed, draws=draws))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="optional YAML config; explicit CLI values take precedence")
    parser.add_argument("--compounds", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--features", type=Path, help="image embedding cache directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--time-hours", type=float, help="one directly measured timepoint")
    parser.add_argument("--molformer-dir", type=Path, help="optional embeddings.npy and compound_index.parquet")
    parser.add_argument("--morgan-radius", type=int)
    parser.add_argument("--morgan-bits", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--bootstrap-draws", type=int)
    args = parser.parse_args()
    if args.config:
        config = yaml.safe_load(args.config.read_text())
        allowed = {"compounds", "observations", "split_manifest", "features", "output", "time_hours",
                   "molformer_dir", "morgan_radius", "morgan_bits", "seed", "bootstrap_draws"}
        if not isinstance(config, dict) or set(config) - allowed:
            parser.error("config must be a mapping with only supported train_moa settings")
        for key, value in config.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    for key in ["compounds", "observations", "split_manifest", "features", "output"]:
        value = getattr(args, key)
        if value is None:
            parser.error(f"--{key.replace('_', '-')} or a config value is required")
        setattr(args, key, Path(value))
    args.morgan_radius = 2 if args.morgan_radius is None else args.morgan_radius
    args.morgan_bits = 2048 if args.morgan_bits is None else args.morgan_bits
    args.seed = 42 if args.seed is None else args.seed
    args.bootstrap_draws = 1000 if args.bootstrap_draws is None else args.bootstrap_draws
    args.time_hours = 24.0 if args.time_hours is None else args.time_hours
    if args.molformer_dir is not None:
        args.molformer_dir = Path(args.molformer_dir)
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    if args.bootstrap_draws < 1:
        parser.error("bootstrap-draws must be positive")

    compounds = pd.read_parquet(args.compounds).sort_values("compound_id").reset_index(drop=True)
    observations = pd.read_parquet(args.observations)
    split = pd.read_parquet(args.split_manifest)
    if compounds.compound_id.duplicated().any() or split.series_id.duplicated().any():
        raise ValueError("Compound IDs and split series IDs must be unique")
    if set(split.split) != {"train", "validation", "test"}:
        raise ValueError("Split manifest must contain train, validation, and test")
    if (split.groupby("compound_id").split.nunique() > 1).any():
        raise ValueError("A compound appears in multiple splits")

    image_meta, image_rows, feature_manifest = load_image_observations(args.features, observations, split, args.time_hours)
    image_meta, x_image = cumulative_compound_mean(image_meta, image_rows, args.time_hours)
    image_frame = image_meta.copy()
    classes = tuple(sorted(compounds.moa_id.unique()))
    if set(image_frame.moa_id) != set(classes):
        raise ValueError("Measured image timepoint does not cover the complete compound universe")
    if not compounds.structure_status.isin(["resolved", "manual_override"]).all():
        raise ValueError("Resolve all compound structures before chemistry baselines")
    by_id = compounds.set_index("compound_id").reindex(image_frame.compound_id)
    x_morgan = morgan_fingerprints(by_id.canonical_smiles.astype(str), radius=args.morgan_radius, n_bits=args.morgan_bits)
    jobs = [("image", x_image), ("morgan", x_morgan), ("image+morgan", np.concatenate([x_image, x_morgan], axis=1))]

    if args.molformer_dir:
        index = pd.read_parquet(args.molformer_dir / "compound_index.parquet")
        vectors = np.load(args.molformer_dir / "embeddings.npy", allow_pickle=False)
        if index.compound_id.duplicated().any() or len(index) != len(vectors):
            raise ValueError("Invalid MoLFormer index or matrix")
        lookup = dict(zip(index.compound_id, index.matrix_row))
        if not set(image_frame.compound_id).issubset(lookup):
            raise ValueError("MoLFormer embeddings are missing evaluated compounds")
        x_molformer = vectors[[lookup[c] for c in image_frame.compound_id]]
        jobs.extend([("molformer", x_molformer), ("image+molformer", np.concatenate([x_image, x_molformer], axis=1))])

    results = [evaluate_model(name, matrix, image_frame, classes, args.seed, args.bootstrap_draws)
               for name, matrix in jobs]
    args.output.mkdir(parents=True)
    pd.DataFrame(results).to_csv(args.output / "metrics.csv", index=False)
    manifest = {
        "status": "complete", "task": "measured-timepoint MoA classification",
        "time_hours": args.time_hours, "seed": args.seed,
        "classes": list(classes), "models": [name for name, _ in jobs],
        "validation_selection": "macro-F1, then NLL, then smaller C; test used once after selection",
        "calibration": "scalar temperature fitted on validation compounds",
        "bootstrap": "stratified compound bootstrap conditional on fitted model",
        "sources_sha256": {str(path): sha256_file(path) for path in [args.compounds, args.observations, args.split_manifest,
            args.features / "features.parquet", args.features / "extraction_manifest.json"]},
        "image_encoder": feature_manifest.get("encoder"),
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"models": len(results), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
