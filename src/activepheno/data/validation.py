"""Trajectory and image integrity checks."""
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import ValidationError


def validate_images(frame: pd.DataFrame, raw_root: Path) -> None:
    root = raw_root.resolve()
    for image in frame.image_path:
        path = Path(image)
        if path.is_absolute() or not (root / path).resolve().is_relative_to(root):
            raise ValidationError(f"image_path must stay inside raw_root: {image}")
        if not (root / path).is_file():
            raise ValidationError(f"missing image: {image}")


def validate_trajectories(frame: pd.DataFrame, interval: float) -> dict[str, Any]:
    for column in ["observation_id"]:
        if frame[column].duplicated().any():
            raise ValidationError(f"duplicate {column}")
    gaps = []
    for series, group in frame.groupby("series_id", sort=True):
        times = group.time_hours.tolist()
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValidationError(f"trajectory {series} is not strictly time ordered")
        for a, b in zip(times, times[1:]):
            if b - a > interval + 1e-6:
                gaps.append({"series_id": series, "after_hours": a, "before_hours": b})
    return {"n_internal_gaps": len(gaps), "internal_gaps": gaps,
            "expected_interval_hours": interval,
            "boundary_completeness": "unknown; absent first/final frames require an acquisition schedule"}
