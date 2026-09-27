import pandas as pd
import pytest

from activepheno.data.observations import build_tables
from activepheno.data.schema import ValidationError, validate_input


@pytest.mark.parametrize("value", ["bad", "", "NaN", "inf", "-1", True])
def test_invalid_time(rows, value):
    rows["time_hours"] = rows.time_hours.astype(object)
    rows.loc[0, "time_hours"] = value
    with pytest.raises(ValidationError):
        validate_input(rows)


def test_missing_column(rows):
    with pytest.raises(ValidationError, match="missing required"):
        validate_input(rows.drop(columns="moa_id"))


@pytest.mark.parametrize("value", [None, "", 123])
def test_invalid_compound(rows, value):
    rows["compound_name_raw"] = rows.compound_name_raw.astype(object)
    rows.loc[0, "compound_name_raw"] = value
    with pytest.raises(ValidationError, match="compound_name_raw"):
        validate_input(rows)


def test_label_conflict(rows):
    rows.loc[0, ["moa_id", "moa_name"]] = "M2"
    with pytest.raises(ValidationError, match="conflicting MoA"):
        build_tables(rows)


def test_normalization_collision(rows):
    rows.loc[0, "compound_name_raw"] = "Compound A"
    with pytest.raises(ValidationError, match="normalization collision"):
        build_tables(rows)


def test_structures_remain_unresolved(rows):
    _, compounds = build_tables(rows)
    assert compounds.structure_status.tolist() == ["missing"]
    assert compounds.canonical_smiles.isna().all()


def test_empty():
    with pytest.raises(ValidationError, match="no observations"):
        validate_input(pd.DataFrame())
