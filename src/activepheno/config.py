"""Strict preparation configuration."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PrepareConfig:
    raw_root: str
    metadata: str
    output_root: str = "data/processed"
    input_format: str = "lctp"
    plate_starts: str | None = None
    seed: int = 42
    dataset_version: str = "unspecified"
    exclude_unlabelled: bool = False
    exclude_controls: bool = False
    expected_interval_hours: float = 2.0

    def __post_init__(self) -> None:
        if self.input_format not in {"lctp", "csv"}:
            raise ValueError("input_format must be lctp or csv")
        for key in ("raw_root", "metadata", "output_root", "dataset_version"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be a nonempty string")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for key in ("exclude_unlabelled", "exclude_controls"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a boolean")
        import math
        if (type(self.expected_interval_hours) not in (int, float)
                or not math.isfinite(self.expected_interval_hours)
                or self.expected_interval_hours <= 0):
            raise ValueError("expected_interval_hours must be positive and finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path | None, overrides: dict[str, Any]) -> PrepareConfig:
    data = yaml.safe_load(path.read_text()) if path else {}
    if not isinstance(data, dict):
        raise ValueError("configuration must be a mapping")
    unknown = data.keys() - {f.name for f in fields(PrepareConfig)}
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    data.update({k: v for k, v in overrides.items() if v is not None})
    missing = {"raw_root", "metadata"} - data.keys()
    if missing:
        raise ValueError(f"missing configuration fields: {sorted(missing)}")
    return PrepareConfig(**data)
