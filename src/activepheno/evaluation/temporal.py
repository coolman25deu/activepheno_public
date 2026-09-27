"""Measured-timepoint pooling and validation-set probability calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax


def cumulative_compound_mean(meta: pd.DataFrame, matrix: np.ndarray, horizon_hours: float,
                             *, cell_line: str = "pooled") -> tuple[pd.DataFrame, np.ndarray]:
    """Mean-pool measured embeddings up to a horizon, then across replicates."""
    if not np.isfinite(horizon_hours) or horizon_hours < 0:
        raise ValueError("horizon_hours must be finite and nonnegative")
    required = ["series_id", "compound_id", "moa_id", "time_hours", "matrix_row"]
    if not set(required).issubset(meta.columns):
        raise ValueError(f"metadata missing: {sorted(set(required) - set(meta.columns))}")
    selected = meta if cell_line == "pooled" else meta[meta.cell_line.eq(cell_line)]
    selected = selected[selected.time_hours <= horizon_hours].copy()
    if selected.empty or selected[required].isna().any().any():
        raise ValueError("No valid measured embeddings at or before horizon")
    if not pd.api.types.is_integer_dtype(selected.matrix_row) or (selected.matrix_row < 0).any() or (selected.matrix_row >= len(matrix)).any():
        raise ValueError("matrix_row is outside the embedding matrix")
    for group_col, columns in [("series_id", ["compound_id", "moa_id"]), ("compound_id", ["moa_id"])]:
        if (selected.groupby(group_col)[columns].nunique() > 1).any().any():
            raise ValueError("Trajectory identity or MoA label changes")
    if selected.duplicated(["series_id", "time_hours"]).any():
        raise ValueError("Duplicate measured time point in a series")
    vectors = np.asarray(matrix)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embedding matrix must be finite and two-dimensional")
    selected["_embedding"] = [vectors[int(row)] for row in selected.matrix_row]
    series_records = []
    for series_id, group in selected.groupby("series_id", sort=True):
        record = {"series_id": series_id, "compound_id": group.compound_id.iloc[0],
                  "moa_id": group.moa_id.iloc[0], "embedding": np.mean(np.stack(group["_embedding"]), axis=0)}
        if "split" in group:
            if group.split.nunique() != 1:
                raise ValueError("Series appears in multiple splits")
            record["split"] = group.split.iloc[0]
        series_records.append(record)
    series = pd.DataFrame(series_records)
    compound_records = []
    for compound_id, group in series.groupby("compound_id", sort=True):
        if group.moa_id.nunique() != 1 or ("split" in group and group.split.nunique() != 1):
            raise ValueError("Compound label or split changes across replicates")
        record = {"compound_id": compound_id, "moa_id": group.moa_id.iloc[0],
                  "embedding": np.mean(np.stack(group.embedding), axis=0)}
        if "split" in group:
            record["split"] = group.split.iloc[0]
        if "cell_line" in selected:
            record["cell_line"] = cell_line
        compound_records.append(record)
    frame = pd.DataFrame(compound_records)
    return frame.drop(columns="embedding"), np.stack(frame.embedding).astype(np.float32)


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    logits, y = np.asarray(logits, dtype=float), np.asarray(y_true, dtype=int)
    if logits.ndim != 2 or len(y) != len(logits) or not len(y) or not np.isfinite(logits).all():
        raise ValueError("Invalid validation logits and labels")
    def nll(log_temperature):
        probability = softmax(logits / np.exp(log_temperature), axis=1)
        return float(-np.log(probability[np.arange(len(y)), y] + 1e-15).mean())
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    return softmax(np.asarray(logits, dtype=float) / temperature, axis=1).astype(np.float32)
