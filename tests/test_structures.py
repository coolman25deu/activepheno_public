import pandas as pd
import pytest
pytest.importorskip("rdkit")
pytest.importorskip("json5")
from rdkit import Chem

from activepheno.data.compounds import compounds_from_wells
from activepheno.data.observations import normalize_name, stable_id
from activepheno.data.schema import ValidationError
from activepheno.chemistry.structures import canonical_structure, read_catalog, read_overrides, resolve_compounds


def fixture_tables():
    name = "Example"
    compounds = pd.DataFrame([{"compound_id": stable_id("cmp", normalize_name(name)),
        "compound_name_raw": name, "compound_name_normalized": normalize_name(name),
        "moa_id": "M1", "moa_name": "M1"}])
    catalog = pd.DataFrame([{"compound_name_original": name, "compound_name": "Other alias",
        "metadata_moa": "M1", "smiles": "CCO", "inchi_key": Chem.MolToInchiKey(Chem.MolFromSmiles("CCO")),
        "reference_url": "https://example.org/source"}])
    return compounds, catalog


def test_exact_resolution_and_no_fuzzy_match():
    c, catalog = fixture_tables()
    out, report = resolve_compounds(c, catalog, read_overrides(None))
    assert report["ready_for_chemistry"]
    assert out.structure_status.tolist() == ["resolved"]
    c["compound_name_raw"] = "Example with extra salt"
    out, report = resolve_compounds(c, catalog, read_overrides(None))
    assert out.structure_status.tolist() == ["missing"]
    assert not report["ready_for_chemistry"]


@pytest.mark.parametrize("field,value", [("smiles", "invalid"), ("inchi_key", "wrong"), ("metadata_moa", "other")])
def test_inconsistent_source_is_ambiguous(field, value):
    c, catalog = fixture_tables()
    catalog.loc[0, field] = value
    out, report = resolve_compounds(c, catalog, read_overrides(None))
    assert out.structure_status.tolist() == ["ambiguous"]
    assert not report["ready_for_chemistry"]


def test_multiple_matches_not_silently_selected():
    c, catalog = fixture_tables()
    out, _ = resolve_compounds(c, pd.concat([catalog, catalog]), read_overrides(None))
    assert out.structure_status.tolist() == ["ambiguous"]


def test_override_is_validated_and_preserves_source_issue(tmp_path):
    c, catalog = fixture_tables()
    key = catalog.loc[0, "inchi_key"]
    catalog.loc[0, "smiles"] = "bad"
    override = pd.DataFrame([{"compound_name_raw": "Example", "canonical_smiles": "CCO", "inchikey": key,
        "structure_source": "https://example.org/correction", "reason": "verified source correction", "evidence_sha256": "a" * 64}])
    path = tmp_path / "overrides.csv"
    override.to_csv(path, index=False)
    out, report = resolve_compounds(c, catalog, read_overrides(path))
    assert report["ready_for_chemistry"]
    assert out.structure_status.tolist() == ["manual_override"]
    assert out.resolution_issue.tolist() == ["invalid SMILES"]
    override["inchikey"] = "wrong"
    with pytest.raises(ValidationError, match="disagreement"):
        resolve_compounds(c, catalog, override)


def test_duplicate_inchikey_blocks_chemistry():
    c, catalog = fixture_tables()
    c2 = c.copy()
    c2["compound_id"] = "another-id"
    c2["compound_name_raw"] = "Other alias"
    _, report = resolve_compounds(pd.concat([c, c2]), catalog, read_overrides(None))
    assert report["duplicate_inchikeys"]
    assert not report["ready_for_chemistry"]


def test_native_exclusions_and_id_compatibility():
    df = pd.DataFrame({"Metadata_cmpd_cmpdname": ["Example", "Example", "dmso", None],
                       "Metadata_cmpd_moa_group": ["M1", "M1", "dmso", None]})
    with pytest.raises(ValidationError, match="unlabelled"):
        compounds_from_wells(df)
    out, exclusions = compounds_from_wells(df, exclude_controls=True, exclude_unlabelled=True)
    assert exclusions == {"unlabelled_wells": 1, "control_wells": 1}
    assert out.compound_id.tolist() == [stable_id("cmp", "example")]


def test_catalog_parsed_as_data_only(tmp_path):
    path = tmp_path / "table.html"
    path.write_text("const data = [{compound_name_original:'x', compound_name:'x', metadata_moa:'M', smiles:'CCO', inchi_key:'key', reference_url:'source'}]; throw Error('never execute');")
    assert read_catalog(path).compound_name.tolist() == ["x"]


def test_canonicalization_keeps_salt():
    smiles = "CCN.Cl"
    canonical, _ = canonical_structure(smiles, Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)))
    assert "." in canonical
