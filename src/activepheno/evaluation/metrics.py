"""Generic multiclass metrics and compound bootstrap intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def classification_metrics(y_true, probabilities) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 2 or len(y) != len(p) or len(y) == 0:
        raise ValueError("Expected nonempty labels and aligned probability matrix")
    n_classes = p.shape[1]
    if n_classes < 2 or (y < 0).any() or (y >= n_classes).any():
        raise ValueError("Class indices must match probability columns")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Probabilities must be finite, nonnegative, and sum to one")
    cm = confusion_matrix(y, p.argmax(axis=1), labels=np.arange(n_classes))
    tp = cm.diagonal().astype(float)
    support = cm.sum(axis=1)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    denominator = cm.sum(axis=1) + cm.sum(axis=0)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(np.mean(bins == b) * abs(correct[bins == b].mean() - confidence[bins == b].mean())
              for b in range(10) if (bins == b).any())
    return {"accuracy": float(correct.mean()), "balanced_accuracy": float(recall.mean()),
            "macro_f1": float(f1.mean()), "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0)).mean()),
            "ece": float(ece), "n_compounds": int(len(y)), "confusion_matrix": cm.tolist(),
            "class_recall": recall.tolist(), "class_f1": f1.tolist(), "class_support": support.tolist()}


def compound_bootstrap_ci(y_true, probabilities, *, seed: int = 42, draws: int = 1000) -> dict:
    """Stratified compound bootstrap conditional on already fitted predictions."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if draws < 1 or len(y) != len(p):
        raise ValueError("Invalid bootstrap inputs")
    groups = [np.flatnonzero(y == c) for c in range(p.shape[1])]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Every class needs at least one compound")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        index = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        samples.append(classification_metrics(y[index], p[index])["balanced_accuracy"])
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"balanced_accuracy": [float(low), float(high)], "draws": int(draws),
            "unit": "compound, stratified by class; conditional on fitted model"}
