"""Compound universe from published well metadata, without inventing trajectories."""
import pandas as pd

from .observations import normalize_name, stable_id
from .schema import ValidationError, require_strings


def compounds_from_wells(metadata: pd.DataFrame, *, exclude_controls: bool = False,
                         exclude_unlabelled: bool = False) -> tuple[pd.DataFrame, dict[str, int]]:
    columns = ["Metadata_cmpd_cmpdname", "Metadata_cmpd_moa_group"]
    if not set(columns) <= set(metadata.columns):
        raise ValidationError(f"well metadata requires {columns}")
    valid = metadata[columns].map(lambda x: isinstance(x, str) and bool(x.strip())).all(axis=1)
    if not valid.all() and not exclude_unlabelled:
        raise ValidationError("unlabelled wells exist; explicitly exclude or curate them")
    frame = metadata.loc[valid, columns].copy()
    controls = frame.Metadata_cmpd_moa_group.str.casefold().eq("dmso")
    if controls.any() and not exclude_controls:
        raise ValidationError("DMSO controls need an explicit exclusion policy before chemistry resolution")
    report = {"unlabelled_wells": int((~valid).sum()), "control_wells": int(controls.sum())}
    frame = frame.loc[~controls].drop_duplicates().rename(columns={
        columns[0]: "compound_name_raw", columns[1]: "moa_id"})
    require_strings(frame, ["compound_name_raw", "moa_id"])
    if frame.empty:
        raise ValidationError("no labeled treatment compounds")
    frame["compound_name_normalized"] = frame.compound_name_raw.map(normalize_name)
    if frame.compound_name_normalized.duplicated().any():
        raise ValidationError("duplicate normalized names or conflicting compound labels")
    frame["compound_id"] = frame.compound_name_normalized.map(lambda x: stable_id("cmp", x))
    frame["moa_name"] = frame.moa_id
    return frame.sort_values("compound_id").reset_index(drop=True), report
