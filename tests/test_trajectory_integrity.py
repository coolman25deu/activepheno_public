import pandas as pd
import pytest

from activepheno.data.observations import build_tables
from activepheno.data.schema import ValidationError
from activepheno.data.validation import validate_images, validate_trajectories


def test_sorted_and_stable(rows):
    a, _ = build_tables(rows)
    b, _ = build_tables(rows.sample(frac=1, random_state=3))
    pd.testing.assert_frame_equal(a, b)
    assert a.time_hours.tolist() == [0, 2, 4]
    assert a.series_id.nunique() == 1


def test_duplicate_times(rows):
    rows.loc[0, "time_hours"] = 0
    with pytest.raises(ValidationError, match="duplicate time"):
        build_tables(rows)


def test_distinct_sites_are_distinct_series(rows):
    other = rows.copy()
    other["site_id"] = "2"
    other["image_path"] = "other/" + other.image_path
    obs, _ = build_tables(pd.concat([rows, other], ignore_index=True))
    assert obs.series_id.nunique() == 2


def test_gaps_reported_not_interpolated(rows):
    obs, _ = build_tables(rows[rows.time_hours != 2])
    report = validate_trajectories(obs, 2)
    assert len(obs) == 2
    assert report["n_internal_gaps"] == 1


def test_duplicate_observation_id(rows):
    obs, _ = build_tables(rows)
    obs.loc[1, "observation_id"] = obs.loc[0, "observation_id"]
    with pytest.raises(ValidationError, match="duplicate observation_id"):
        validate_trajectories(obs, 2)


def test_changed_compound_in_well(rows):
    rows.loc[0, "compound_name_raw"] = "another compound"
    with pytest.raises(ValidationError, match="physical trajectory"):
        build_tables(rows)


def test_missing_and_escaping_images(rows, tmp_path):
    with pytest.raises(ValidationError, match="missing image"):
        validate_images(rows, tmp_path)
    rows.loc[0, "image_path"] = "../escape.tif"
    with pytest.raises(ValidationError, match="inside raw_root"):
        validate_images(rows, tmp_path)
