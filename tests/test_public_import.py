"""Public package imports and representative public APIs."""


def test_public_package_imports():
    import activepheno
    from activepheno.chemistry.fingerprints import morgan_fingerprints
    from activepheno.evaluation.temporal import cumulative_compound_mean, fit_temperature
    from activepheno.models.classification import fit_classifier
    from activepheno.splits.generation import make_split

    assert activepheno.__version__ == "0.1.0"
    assert all(callable(obj) for obj in [morgan_fingerprints, cumulative_compound_mean, fit_temperature, fit_classifier, make_split])
