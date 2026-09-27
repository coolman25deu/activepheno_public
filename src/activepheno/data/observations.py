"""Content-derived IDs, independent of row order and local root directory."""
import hashlib
import json
import unicodedata

import pandas as pd

from .schema import ValidationError, validate_input


def stable_id(kind: str, *parts: object) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return f"{kind}_{hashlib.sha256(encoded).hexdigest()[:24]}"


def normalize_name(name: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", name).split()).casefold()


def build_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_input(frame)
    obs = frame.copy()
    obs["time_hours"] = pd.to_numeric(obs.time_hours, errors="raise").astype(float)
    obs["compound_name_normalized"] = obs.compound_name_raw.map(normalize_name)
    # A normalization collision needs review, never an automatic identity merge.
    if (obs.groupby("compound_name_normalized").compound_name_raw.nunique() > 1).any():
        raise ValidationError("compound name normalization collision; resolve explicitly in source metadata")
    obs["compound_id"] = obs.compound_name_normalized.map(lambda x: stable_id("cmp", x))
    if (obs.groupby("compound_id").moa_id.nunique() > 1).any():
        raise ValidationError("compound has conflicting MoA labels")
    physical = ["cell_line", "plate_id", "well_id", "site_id", "channel"]
    if (obs.groupby(physical)[["compound_id", "replicate_id"]].nunique() > 1).any().any():
        raise ValidationError("physical trajectory changes compound or replicate")
    keys = physical + ["replicate_id", "compound_id"]
    obs["series_id"] = [stable_id("series", *row) for row in obs[keys].itertuples(index=False, name=None)]
    if obs.duplicated(["series_id", "time_hours"]).any():
        raise ValidationError("duplicate time point within a trajectory")
    obs["observation_id"] = [
        stable_id("obs", s, t) for s, t in zip(obs.series_id, obs.time_hours)
    ]
    compounds = obs[["compound_id", "compound_name_raw", "compound_name_normalized", "moa_id", "moa_name"]].drop_duplicates()
    for column in ["canonical_smiles", "inchikey", "structure_source"]:
        compounds[column] = pd.Series(pd.NA, index=compounds.index, dtype="string")
    compounds["structure_status"] = "missing"
    return (obs.sort_values(["series_id", "time_hours"]).reset_index(drop=True),
            compounds.sort_values("compound_id").reset_index(drop=True))
