"""Native published LCTP ScanData and explicit observation CSV ingestion."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import re
import subprocess
import tempfile
from typing import Any

import pandas as pd

from activepheno.config import PrepareConfig
from .observations import build_tables
from .schema import ValidationError, require_strings
from .validation import validate_images, validate_trajectories

SCAN_PATTERN = re.compile(
    r"^(?P<cell>[^/]+)/ScanData/(?P<month>\d{4})/(?P<day>\d{2})/"
    r"(?P<clock>\d{4})/(?P<vessel>[^/]+)/(?P<well>[A-P]\d{1,2})-"
    r"(?P<site>\d+)-(?P<channel>[^/]+)\.tif$"
)
NATIVE_COLUMNS = ["Metadata_Cells", "Metadata_VesselID", "Metadata_Well",
                  "Metadata_Plate", "Metadata_cmpd_cmpdname", "Metadata_cmpd_moa_group"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_native(config: PrepareConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    root = Path(config.raw_root)
    if not config.plate_starts:
        raise ValidationError("native LCTP requires --plate-starts JSON (cell/vessel -> YYMM/DD/HHMM); do not infer t0 from a partial download")
    starts = json.loads(Path(config.plate_starts).read_text())
    if not isinstance(starts, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in starts.items()):
        raise ValidationError("plate_starts must map cell/vessel to YYMM/DD/HHMM strings")
    start_dates = {k: datetime.strptime(v, "%y%m/%d/%H%M") for k, v in starts.items()}
    metadata = pd.read_parquet(config.metadata)
    missing = set(NATIVE_COLUMNS) - set(metadata.columns)
    if missing:
        raise ValidationError(f"native metadata missing columns: {sorted(missing)}")
    keys = ["Metadata_Cells", "Metadata_VesselID", "Metadata_Well"]
    require_strings(metadata, keys + ["Metadata_Plate"])
    if metadata.duplicated(keys).any():
        raise ValidationError("duplicate cell/vessel/well in native metadata")
    if (metadata.groupby(keys[:2]).Metadata_Plate.nunique() > 1).any():
        raise ValidationError("cell/vessel maps to multiple plates")
    lookup = metadata.set_index(keys)
    records: list[dict[str, Any]] = []
    excluded = {"unlabelled_images": 0, "control_images": 0}
    for image in sorted(root.glob("*/ScanData/**/*.tif")):
        relative = image.relative_to(root).as_posix()
        match = SCAN_PATTERN.fullmatch(relative)
        if not match:
            raise ValidationError(f"unrecognized LCTP ScanData image path: {relative}")
        info = match.groupdict()
        key = (info["cell"], info["vessel"], info["well"])
        if key not in lookup.index:
            raise ValidationError(f"image has no well metadata: {relative}")
        row = lookup.loc[key]
        label, name = row.Metadata_cmpd_moa_group, row.Metadata_cmpd_cmpdname
        if any(not isinstance(v, str) or not v.strip() for v in (label, name)):
            if config.exclude_unlabelled:
                excluded["unlabelled_images"] += 1
                continue
            raise ValidationError(f"unlabelled well: {key}; curate metadata or explicitly --exclude-unlabelled")
        if config.exclude_controls and label.casefold() == "dmso":
            excluded["control_images"] += 1
            continue
        plate_key = f"{info['cell']}/{info['vessel']}"
        if plate_key not in start_dates:
            raise ValidationError(f"missing plate start for {plate_key}")
        stamp = f"{info['month']}/{info['day']}/{info['clock']}"
        elapsed = (datetime.strptime(stamp, "%y%m/%d/%H%M") - start_dates[plate_key]).total_seconds() / 3600
        records.append({**row.to_dict(), "compound_name_raw": name, "moa_id": label,
                        "moa_name": label, "cell_line": info["cell"],
                        "plate_id": row.Metadata_Plate, "well_id": info["well"],
                        "replicate_id": f"{row.Metadata_Plate}/{info['well']}",
                        "site_id": info["site"], "channel": info["channel"],
                        "time_hours": elapsed, "image_path": relative,
                        "acquisition_timestamp_raw": stamp,
                        "vessel_id_raw": info["vessel"]})
    return pd.DataFrame(records), excluded


def prepare(config: PrepareConfig, *, dry_run: bool = False) -> dict[str, Any]:
    root = Path(config.raw_root)
    if not root.is_dir():
        raise ValidationError(f"raw_root is not a directory: {root}")
    if config.input_format == "lctp":
        frame, excluded = read_native(config)
    else:
        frame = pd.read_csv(config.metadata, dtype=str, keep_default_na=False)
        excluded = {}
    observations, compounds = build_tables(frame)
    validate_images(observations, root)
    report = validate_trajectories(observations, config.expected_interval_hours)
    sources = [Path(config.metadata)]
    if config.plate_starts:
        sources.append(Path(config.plate_starts))
    source_records = [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in sources]
    # Image hashes identify the exact bytes indexed, including a partial acquisition.
    image_records = [{"path": p, "sha256": sha256_file(root / p)} for p in sorted(observations.image_path)]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=Path(__file__).parent, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    code_files = sorted(Path(__file__).parents[1].rglob("*.py"))
    code_hash = hashlib.sha256("".join(p.relative_to(Path(__file__).parents[1]).as_posix() + sha256_file(p) for p in code_files).encode()).hexdigest()
    from datetime import timezone
    manifest: dict[str, Any] = {
        "dataset_name": "LCTP", "dataset_version": config.dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_images": len(observations), "n_series": observations.series_id.nunique(),
        "n_compounds": len(compounds), "n_moas": compounds.moa_id.nunique(),
        "cell_lines": sorted(observations.cell_line.unique().tolist()),
        "time_points": sorted(observations.time_hours.unique().tolist()),
        "plates": sorted(observations.plate_id.unique().tolist()),
        "source_files": source_records, "image_files": image_records,
        "code_commit": commit, "code_dirty": dirty, "source_code_sha256": code_hash,
        "config": config.to_dict(), "seed": config.seed,
        "exclusions": excluded, "validation": report,
        "coverage": "locally available images only; not a claim of full dataset completeness",
        "moa_names": "native labels retained verbatim; no biological descriptions generated",
        "id_version": "sha256-v1",
    }
    if not dry_run:
        output = Path(config.output_root)
        if output.exists():
            raise ValidationError(f"output already exists: {output}; choose a new output directory")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".activepheno-", dir=output.parent) as temporary:
            staging = Path(temporary) / "result"
            staging.mkdir()
            observations.to_parquet(staging / "observations.parquet", index=False)
            compounds.to_parquet(staging / "compounds_unresolved.parquet", index=False)
            manifest["output_sha256"] = {p.name: sha256_file(p) for p in staging.iterdir()}
            (staging / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            staging.rename(output)
    return manifest
