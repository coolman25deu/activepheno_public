import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from activepheno.config import PrepareConfig, load_config
from activepheno.data.lctp import prepare
from activepheno.data.schema import ValidationError


def test_csv_end_to_end(rows, tmp_path):
    for image in rows.image_path:
        (tmp_path / image).write_bytes(b"synthetic image bytes")
    metadata = tmp_path / "metadata.csv"
    rows.to_csv(metadata, index=False)
    out = tmp_path / "processed"
    cfg = PrepareConfig(str(tmp_path), str(metadata), str(out), input_format="csv")
    prepare(cfg, dry_run=True)
    assert not out.exists()
    first = prepare(cfg)
    assert first["n_images"] == 3
    assert first["n_series"] == 1
    assert first["seed"] == 42
    assert len(first["image_files"]) == 3
    assert json.loads((out / "dataset_manifest.json").read_text())["output_sha256"]
    rows.iloc[::-1].to_csv(metadata, index=False)
    second_out = tmp_path / "again"
    prepare(replace(cfg, output_root=str(second_out)))
    pd.testing.assert_frame_equal(pd.read_parquet(out / "observations.parquet"),
                                  pd.read_parquet(second_out / "observations.parquet"))
    with pytest.raises(ValidationError, match="already exists"):
        prepare(cfg)


def native_fixture(tmp_path):
    image = tmp_path / "A549-WT/ScanData/2410/21/1300/433/A2-1-Ph.tif"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"synthetic image bytes")
    metadata = tmp_path / "metadata.parquet"
    pd.DataFrame([{"Metadata_Cells": "A549-WT", "Metadata_VesselID": "433",
                   "Metadata_Well": "A2", "Metadata_Plate": "P106068",
                   "Metadata_cmpd_cmpdname": "Example", "Metadata_cmpd_moa_group": "MAPK"}]).to_parquet(metadata)
    starts = tmp_path / "starts.json"
    starts.write_text(json.dumps({"A549-WT/433": "2410/21/1100"}))
    return PrepareConfig(str(tmp_path), str(metadata), str(tmp_path / "out"), plate_starts=str(starts))


def test_native_partial_download_keeps_time_origin(tmp_path):
    cfg = native_fixture(tmp_path)
    prepare(cfg)
    obs = pd.read_parquet(Path(cfg.output_root) / "observations.parquet")
    assert obs.time_hours.tolist() == [2.0]
    assert obs.plate_id.tolist() == ["P106068"]
    assert obs.vessel_id_raw.tolist() == ["433"]


def test_native_requires_start(tmp_path):
    cfg = native_fixture(tmp_path)
    with pytest.raises(ValidationError, match="plate-starts"):
        prepare(replace(cfg, plate_starts=None))


def test_unlabelled_native_well_fails(tmp_path):
    cfg = native_fixture(tmp_path)
    df = pd.read_parquet(cfg.metadata)
    df["Metadata_cmpd_moa_group"] = None
    df.to_parquet(cfg.metadata)
    with pytest.raises(ValidationError, match="unlabelled well"):
        prepare(cfg)


def test_duplicate_native_metadata_fails(tmp_path):
    cfg = native_fixture(tmp_path)
    df = pd.read_parquet(cfg.metadata)
    pd.concat([df, df]).to_parquet(cfg.metadata)
    with pytest.raises(ValidationError, match="duplicate cell/vessel/well"):
        prepare(cfg)


def test_cli_help_and_error():
    script = Path(__file__).parents[1] / "scripts/prepare_lctp.py"
    help_result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0
    error = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert error.returncode == 1
    assert "missing configuration fields" in error.stderr


def test_unknown_config_key(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("raw_rooot: wrong\n")
    with pytest.raises(ValueError, match="unknown configuration"):
        load_config(path, {})
