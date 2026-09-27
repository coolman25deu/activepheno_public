import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PIL")
from PIL import Image

from activepheno.features.cache import cache_key, load_entry, save_entry
from activepheno.features.extraction import extract, inspect_images
from activepheno.features.tiling import TileConfig, coordinates, read_grayscale, tile_batches


def test_grid_covers_borders_and_padding_repeats_edges():
    config = TileConfig(tile_size=4, stride=4)
    frame = np.arange(35, dtype=np.float32).reshape(5, 7)
    boxes = coordinates(5, 7, config)
    cover = np.zeros(frame.shape, dtype=int)
    for box in boxes:
        x, y, w, h = (box[k] for k in ["x", "y", "valid_width", "valid_height"])
        cover[y:y+h, x:x+w] += 1
    assert np.all(cover == 1)
    tiles = [tile for batch in tile_batches(frame, config, 3) for tile in batch]
    assert len(tiles) == 4
    assert tiles[-1].shape == (4, 4, 3)
    assert np.all(tiles[-1][:, -1, :] == frame[-1, -1])
    assert np.array_equal(tiles[0][:, :, 0], tiles[0][:, :, 2])


def test_small_image_and_overlap_grid():
    assert len(coordinates(2, 3, TileConfig(4, 4))) == 1
    assert len(coordinates(5, 7, TileConfig(4, 2))) == 6
    with pytest.raises(ValueError, match="gaps"):
        TileConfig(4, 5)


def test_pyramidal_tiff_uses_explicit_page(tmp_path):
    path = tmp_path / "pyramid.tif"
    large = Image.fromarray(np.ones((8, 8), dtype=np.uint8))
    small = Image.fromarray(np.zeros((4, 4), dtype=np.uint8))
    large.save(path, save_all=True, append_images=[small])
    frame, info = read_grayscale(path, TileConfig())
    assert frame.shape == (8, 8)
    assert info["page_count"] == 2
    frame, _ = read_grayscale(path, TileConfig(image_page=1))
    assert frame.shape == (4, 4)


def test_intensity_conversion_is_fixed_not_per_frame(tmp_path):
    path = tmp_path / "image.tif"
    Image.fromarray(np.array([[0, 127], [255, 64]], dtype=np.uint8)).save(path)
    frame, info = read_grayscale(path, TileConfig())
    np.testing.assert_allclose(frame, [[0, 127/255], [1, 64/255]])
    assert info["dtype"] == "uint8"
    Image.fromarray(np.array([[0, 1024]], dtype=np.uint16)).save(path)
    with pytest.raises(ValueError, match="sensor bounds"):
        read_grayscale(path, TileConfig())
    scaled, _ = read_grayscale(path, TileConfig(intensity_max=65535))
    assert scaled[0, 1] == np.float32(1024/65535)


class DiagnosticEncoder:
    """Test double only; never exported as a pretrained representation."""
    descriptor = {"model_id": "test-only", "revision": "fixture"}
    output_dim = 2

    def __init__(self):
        self.calls = 0

    def encode(self, tiles):
        self.calls += 1
        return np.array([[tile.mean(), tile.max()] for tile in tiles], dtype=np.float32)


def inputs(tmp_path):
    image = np.arange(35, dtype=np.uint8).reshape(5, 7)
    Image.fromarray(image).save(tmp_path / "image.tif")
    obs = pd.DataFrame([{"observation_id": "obs1", "compound_id": "cmp1", "series_id": "s1", "image_path": "image.tif"}])
    return obs


def test_streaming_mean_handles_short_final_batch_and_resume(tmp_path):
    obs = inputs(tmp_path)
    config = TileConfig(4, 4)
    encoder = DiagnosticEncoder()
    output = tmp_path / "cache"
    result = extract(obs, tmp_path, output, config, encoder, batch_size=3)
    assert result["n_tiles"] == 4
    index = pd.read_parquet(output / "features.parquet")
    with np.load(output / index.iloc[0].embedding_path) as entry:
        got = entry["embedding"]
        info = json.loads(entry["metadata"].item())
    frame, _ = read_grayscale(tmp_path / "image.tif", config)
    tiles = [t for b in tile_batches(frame, config, 3) for t in b]
    expected = encoder.encode(tiles).mean(axis=0)
    np.testing.assert_allclose(got, expected)
    assert len(info["provenance"]["coordinates"]) == 4
    calls = encoder.calls
    resumed = extract(obs, tmp_path, output, config, encoder, batch_size=3)
    assert resumed["cache_hits"] == 1
    assert encoder.calls == calls
    # Byte changes force a new cache entry despite the same observation ID/path.
    Image.fromarray(np.zeros((5, 7), dtype=np.uint8)).save(tmp_path / "image.tif")
    changed = extract(obs, tmp_path, output, config, encoder, batch_size=3)
    assert changed["cache_hits"] == 0


def test_cache_invalidation_and_corruption(tmp_path):
    descriptor = {"image": "hash", "revision": "a", "pooling": "mean"}
    assert cache_key(descriptor) != cache_key({**descriptor, "revision": "b"})
    assert cache_key(descriptor) != cache_key({**descriptor, "pooling": "cls"})
    path = tmp_path / "entry.npz"
    save_entry(path, np.array([1, 2]), descriptor, {})
    with np.load(path) as entry:
        metadata = str(entry["metadata"].item())
    np.savez(path, embedding=np.array([9, 9], dtype=np.float32), metadata=metadata)
    with pytest.raises(ValueError, match="checksum"):
        load_entry(path, descriptor)


def test_invalid_encoder_does_not_write_index(tmp_path):
    obs = inputs(tmp_path)
    encoder = DiagnosticEncoder()
    encoder.encode = lambda tiles: np.full((len(tiles), 2), np.nan)
    with pytest.raises(ValueError, match="nonfinite"):
        extract(obs, tmp_path, tmp_path / "cache", TileConfig(4, 4), encoder)
    assert not (tmp_path / "cache/features.parquet").exists()


def test_inspect_has_no_output_side_effects(tmp_path):
    obs = inputs(tmp_path)
    result = inspect_images(obs, tmp_path, TileConfig(4, 4))
    assert result.n_tiles.tolist() == [4]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["image.tif"]
