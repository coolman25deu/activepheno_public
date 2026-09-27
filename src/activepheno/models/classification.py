"""Validation-selected, compound-weighted logistic MoA classification."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from activepheno.evaluation.metrics import classification_metrics


def fit_classifier(
    x_train: np.ndarray, train: pd.DataFrame, x_validation: np.ndarray,
    validation: pd.DataFrame, *, c_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    scale: bool = True, seed: int = 42,
):
    """Select regularization on validation compounds and return model/classes/trials."""
    required = {"compound_id", "moa_id"}
    if not required.issubset(train) or not required.issubset(validation):
        raise ValueError("train and validation need compound_id and moa_id columns")
    if train.empty or validation.empty or set(train.compound_id) & set(validation.compound_id):
        raise ValueError("Nonempty train and validation sets must be compound-disjoint")
    x_train, x_validation = np.asarray(x_train), np.asarray(x_validation)
    if x_train.ndim != 2 or x_validation.ndim != 2 or x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("Feature matrices must be two-dimensional and aligned")
    if len(x_train) != len(train) or len(x_validation) != len(validation):
        raise ValueError("Feature rows must align with metadata")
    if not np.isfinite(x_train).all() or not np.isfinite(x_validation).all():
        raise ValueError("Features must be finite")
    classes = tuple(sorted(train.moa_id.unique()))
    if len(classes) < 2 or set(validation.moa_id) != set(classes):
        raise ValueError("Train and validation must contain the same two or more MoA classes")
    class_to_index = {name: index for index, name in enumerate(classes)}
    y_train = train.moa_id.map(class_to_index).to_numpy()
    y_validation = validation.moa_id.map(class_to_index).to_numpy()
    counts = train.compound_id.value_counts()
    weights = train.compound_id.map(1 / counts).to_numpy(dtype=float, copy=True)
    weights /= weights.mean()
    best_model, best_score, trials = None, None, []
    for c_value in sorted(c_values):
        if c_value <= 0:
            raise ValueError("C values must be positive")
        classifier = LogisticRegression(C=c_value, max_iter=3000, random_state=seed, solver="lbfgs")
        model = make_pipeline(StandardScaler(), classifier) if scale else classifier
        fit_args = {"logisticregression__sample_weight": weights} if scale else {"sample_weight": weights}
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x_train, y_train, **fit_args)
        probabilities = model.predict_proba(x_validation)
        result = classification_metrics(y_validation, probabilities)
        trials.append({"C": float(c_value), **result})
        score = (result["macro_f1"], -result["nll"], -c_value)
        if best_score is None or score > best_score:
            best_model, best_score = model, score
    return best_model, classes, trials
