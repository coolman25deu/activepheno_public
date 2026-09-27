"""Validation shared by native LCTP and explicit CSV adapters."""
import pandas as pd
import numpy as np


class ValidationError(ValueError):
    """Invalid input with an actionable message."""


INPUT_COLUMNS = [
    "compound_name_raw", "moa_id", "moa_name", "cell_line", "plate_id",
    "well_id", "replicate_id", "site_id", "channel", "time_hours", "image_path",
]


def require_strings(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValidationError(f"missing required columns: {sorted(missing)}")
    for column in columns:
        valid = frame[column].map(lambda x: isinstance(x, str) and bool(x.strip()))
        if not valid.all():
            raise ValidationError(f"{column}: missing or non-string values at rows {frame.index[~valid].tolist()[:8]}")


def validate_input(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValidationError("no observations found")
    require_strings(frame, [c for c in INPUT_COLUMNS if c != "time_hours"])
    if "time_hours" not in frame:
        raise ValidationError("missing required column: time_hours")
    try:
        values = pd.to_numeric(frame.time_hours, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValidationError("time_hours must contain numeric elapsed hours") from exc
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValidationError("time_hours must be finite and nonnegative")
    if frame.time_hours.map(lambda x: isinstance(x, bool)).any():
        raise ValidationError("time_hours cannot be boolean")
    if frame.image_path.duplicated().any():
        raise ValidationError("duplicate image_path values")
    if (frame.groupby("moa_id").moa_name.nunique() > 1).any():
        raise ValidationError("moa_id maps to conflicting moa_name values")
