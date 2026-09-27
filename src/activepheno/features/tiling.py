"""Fixed grids with explicit intensity conversion and no foreground selection."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TileConfig:
    tile_size: int = 224
    stride: int = 224
    intensity_min: float = 0.0
    intensity_max: float = 255.0
    image_page: int = 0

    def __post_init__(self) -> None:
        if type(self.image_page) is not int or self.image_page < 0:
            raise ValueError("image_page must be a nonnegative integer")
        if type(self.tile_size) is not int or type(self.stride) is not int or not 0 < self.stride <= self.tile_size:
            raise ValueError("tile_size and stride must be positive integers; stride cannot leave gaps")
        if not all(math.isfinite(v) for v in (self.intensity_min, self.intensity_max)) or self.intensity_max <= self.intensity_min:
            raise ValueError("intensity bounds must be finite and increasing")

    def descriptor(self) -> dict:
        return {**asdict(self), "padding": "edge_right_bottom", "version": "fixed-grid-v1",
                "channels": "grayscale_repeated_rgb", "tile_order": "row_major"}


def read_grayscale(path: Path, config: TileConfig) -> tuple[np.ndarray, dict]:
    with Image.open(path) as image:
        page_count = getattr(image, "n_frames", 1)
        if config.image_page >= page_count:
            raise ValueError(f"configured image_page is absent: {path}")
        image.seek(config.image_page)
        raw = np.asarray(image)
        mode = image.mode
    if raw.ndim != 2 or raw.dtype.kind not in "uif":
        raise ValueError(f"expected one grayscale numeric plane: {path}")
    if not np.isfinite(raw).all() or raw.size == 0:
        raise ValueError(f"nonfinite or empty image: {path}")
    lo, hi = float(raw.min()), float(raw.max())
    if lo < config.intensity_min or hi > config.intensity_max:
        raise ValueError(f"image range [{lo}, {hi}] exceeds configured intensity bounds; specify sensor bounds explicitly")
    frame = (raw.astype(np.float32) - config.intensity_min) / (config.intensity_max - config.intensity_min)
    return frame, {"height": int(raw.shape[0]), "width": int(raw.shape[1]),
                   "dtype": str(raw.dtype), "mode": mode, "minimum": lo, "maximum": hi,
                   "image_page": config.image_page, "page_count": page_count}


def coordinates(height: int, width: int, config: TileConfig) -> list[dict]:
    if min(height, width) <= 0:
        raise ValueError("image dimensions must be positive")
    ny = max(1, math.ceil((height - config.tile_size) / config.stride) + 1)
    nx = max(1, math.ceil((width - config.tile_size) / config.stride) + 1)
    return [{"x": x * config.stride, "y": y * config.stride,
             "valid_width": min(config.tile_size, width - x * config.stride),
             "valid_height": min(config.tile_size, height - y * config.stride)}
            for y in range(ny) for x in range(nx)]


def tile_batches(frame: np.ndarray, config: TileConfig, batch_size: int):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    batch = []
    for box in coordinates(*frame.shape, config):
        x, y, w, h = box["x"], box["y"], box["valid_width"], box["valid_height"]
        tile = frame[y:y+h, x:x+w]
        tile = np.pad(tile, ((0, config.tile_size-h), (0, config.tile_size-w)), mode="edge")
        batch.append(np.repeat(tile[:, :, None], 3, axis=2))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
