import io
import json
import zipfile

import numpy as np
import pytest
pytest.importorskip("PIL")
from PIL import Image

from activepheno.data.lctp import sha256_file
from activepheno.data.repack import repack_archive


def make_archive(tmp_path, *, reverse=False):
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    large = Image.fromarray(pixels)
    small = Image.fromarray(pixels[::2, ::2])
    pages = [small, large] if reverse else [large, small]
    buffer = io.BytesIO()
    pages[0].save(buffer, format="TIFF", save_all=True, append_images=pages[1:],
                  compression="tiff_deflate", dpi=(100, 100))
    archive = tmp_path / "A549-WT.zip"
    member = "A549-WT/ScanData/2410/21/1100/433/A2-1-Ph.tif"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(member, buffer.getvalue())
        z.writestr("A549-WT/Masks/ignore.tif", buffer.getvalue())
    return archive, member, pixels


def test_repack_lossless_metadata_resume_and_repair(tmp_path):
    archive, member, pixels = make_archive(tmp_path)
    output = tmp_path / "repacked"
    source_hash = sha256_file(archive)
    result = repack_archive(archive, output, source_hash, workers=2)
    assert result["n_images"] == 1
    assert result["reused_images"] == 0
    assert sha256_file(archive) == source_hash
    assert not (output / "A549-WT/Masks").exists()
    with Image.open(output / member) as image:
        assert image.n_frames == 1
        assert np.array_equal(np.array(image), pixels)
        assert float(image.tag_v2[282]) == 100.0
    record = json.loads((output / result["manifest"]).read_text())
    assert len(record["source_pages"]) == 2
    assert record["pixels_identical"]
    assert record["output_sha256"] == sha256_file(output / member)
    resumed = repack_archive(archive, output, source_hash)
    assert resumed["reused_images"] == 1
    (output / member).write_bytes(b"damaged output")
    repaired = repack_archive(archive, output, source_hash)
    assert repaired["reused_images"] == 0
    assert sha256_file(output / member) == record["output_sha256"]


def test_repack_rejects_changed_archive(tmp_path):
    archive, _, _ = make_archive(tmp_path)
    output = tmp_path / "out"
    repack_archive(archive, output, "original")
    with pytest.raises(ValueError, match="archive/policy differs"):
        repack_archive(archive, output, "changed")


def test_repack_rejects_non_pyramid_page_order(tmp_path):
    archive, _, _ = make_archive(tmp_path, reverse=True)
    with pytest.raises(ValueError, match="descending pyramid"):
        repack_archive(archive, tmp_path / "out", sha256_file(archive))
