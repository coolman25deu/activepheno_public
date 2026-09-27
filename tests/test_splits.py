import pandas as pd
import pytest
pytest.importorskip("rdkit")
pytest.importorskip("json5")
from rdkit import Chem

from activepheno.data.observations import build_tables
from activepheno.data.schema import ValidationError
from activepheno.splits.generation import make_split, scaffold_key


@pytest.fixture
def split_data():
    rows = []
    structures = {}
    for label, prefix in [("M1", "C"), ("M2", "CC")]:
        for i, ring in enumerate(["c1ccccc1", "c1ccncc1", "C1CCCCC1"]):
            name = f"{label}-{i}"
            structures[name] = prefix + ring
            for replicate in [1, 2]:
                for time in [0, 2]:
                    rows.append({"compound_name_raw": name, "moa_id": label, "moa_name": label,
                        "cell_line": "synthetic", "plate_id": f"P{replicate}", "well_id": name,
                        "replicate_id": str(replicate), "site_id": "1", "channel": "Ph",
                        "time_hours": time, "image_path": f"{name}/{replicate}/{time}.tif"})
    obs, compounds = build_tables(pd.DataFrame(rows))
    compounds["canonical_smiles"] = compounds.compound_name_raw.map(structures)
    compounds["inchikey"] = compounds.canonical_smiles.map(lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)))
    compounds["structure_status"] = "resolved"
    return compounds, obs


@pytest.mark.parametrize("strategy", ["compound_disjoint", "scaffold_disjoint"])
def test_disjoint_deterministic_and_replicates_kept(split_data, strategy):
    compounds, obs = split_data
    a, report = make_split(compounds, obs, strategy=strategy, attempts=100)
    b, _ = make_split(compounds.sample(frac=1, random_state=2), obs.sample(frac=1, random_state=1), strategy=strategy, attempts=100)
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby("compound_id").split.nunique().max() == 1
    assert a.series_id.nunique() == obs.series_id.nunique()
    assert set(a.split) == {"train", "validation", "test"}
    if strategy == "scaffold_disjoint":
        assert a.groupby("scaffold").split.nunique().max() == 1
    assert report["n_compounds"] == 6


def test_missing_trajectories_prevent_split(split_data):
    c, obs = split_data
    with pytest.raises(ValidationError, match="no indexed trajectories"):
        make_split(c, obs.iloc[:1])


def test_unresolved_chemistry_prevents_split(split_data):
    c, obs = split_data
    c.loc[0, "structure_status"] = "ambiguous"
    with pytest.raises(ValidationError, match="resolve all"):
        make_split(c, obs)


def test_duplicate_observations_prevent_split(split_data):
    c, obs = split_data
    with pytest.raises(ValidationError, match="duplicate observation_id"):
        make_split(c, pd.concat([obs, obs.iloc[:1]]))


def test_series_identity_change_prevents_split(split_data):
    c, obs = split_data
    obs.loc[0, "replicate_id"] = "changed"
    with pytest.raises(ValidationError, match="trajectory identity"):
        make_split(c, obs)


def test_infeasible_scaffold_split_fails(split_data):
    c, obs = split_data
    # Six distinct acyclic molecules all share the empty Murcko scaffold.
    c["canonical_smiles"] = ["C" * i for i in range(2, 8)]
    c["inchikey"] = c.canonical_smiles.map(lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)))
    with pytest.raises(ValidationError, match="three scaffolds"):
        make_split(c, obs, strategy="scaffold_disjoint")


def test_scaffold_parent_drops_counterions_and_ignores_stereo():
    assert scaffold_key("Cc1ccccc1.Cl") == scaffold_key("CCc1ccccc1")
    assert scaffold_key("CCCC") == "ACYCLIC"
