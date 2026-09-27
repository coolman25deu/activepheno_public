import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from activepheno.chemistry.fingerprints import morgan_fingerprints
from activepheno.evaluation.metrics import classification_metrics
from activepheno.models.classification import fit_classifier


def test_morgan_and_validation_selected_classifier():
    x = morgan_fingerprints(["CCO", "CCN", "c1ccccc1", "c1ccncc1", "CCCO", "CCCN"], n_bits=64)
    train = pd.DataFrame({"compound_id": ["a", "b", "c", "d"], "moa_id": ["x", "x", "y", "y"]})
    validation = pd.DataFrame({"compound_id": ["e", "f"], "moa_id": ["x", "y"]})
    model, classes, trials = fit_classifier(x[:4], train, x[4:], validation, c_values=(0.1, 1.0))
    assert classes == ("x", "y") and len(trials) == 2
    assert model.predict_proba(x[4:]).shape == (2, 2)
    assert classification_metrics([0, 1], model.predict_proba(x[4:]))["n_compounds"] == 2


def test_classifier_rejects_compound_overlap():
    frame = pd.DataFrame({"compound_id": ["a", "b"], "moa_id": ["x", "y"]})
    with pytest.raises(ValueError, match="compound-disjoint"):
        fit_classifier(np.eye(2), frame, np.eye(2), frame)
