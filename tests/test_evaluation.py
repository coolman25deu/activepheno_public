import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
from activepheno.evaluation.metrics import compound_bootstrap_ci
from activepheno.evaluation.temporal import apply_temperature, cumulative_compound_mean, fit_temperature


def test_measured_embedding_pooling_and_calibration():
    meta = pd.DataFrame({
        "series_id": ["s1", "s1", "s2", "s2"], "compound_id": ["a", "a", "b", "b"],
        "moa_id": ["x", "x", "y", "y"], "time_hours": [0, 4, 0, 4],
        "matrix_row": [0, 1, 2, 3], "split": ["train", "train", "test", "test"],
    })
    frame, pooled = cumulative_compound_mean(meta, np.eye(4, dtype=np.float32), 4)
    assert frame.compound_id.tolist() == ["a", "b"]
    assert pooled.shape == (2, 4)
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1, 0, 1])
    temperature = fit_temperature(logits, labels)
    assert temperature > 0
    assert apply_temperature(logits, temperature).shape == logits.shape
    ci = compound_bootstrap_ci(labels, apply_temperature(logits, temperature), draws=20)
    assert 0 <= ci["balanced_accuracy"][0] <= ci["balanced_accuracy"][1] <= 1
