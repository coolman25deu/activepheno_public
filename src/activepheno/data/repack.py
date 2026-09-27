"""Lossless, resumable removal of TIFF pyramid pages from verified archives."""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tempfile
import zipfile

import numpy as np
from PIL import Image, TiffImagePlugin

from .lctp import sha256_file


def json_tag(value):
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode()}
    if isinstance(value, (list, tuple)):
        return [json_tag(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__, "repr": str(value)}


def convert_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo, output: Path) -> dict:
    relative = PurePosixPath(member.filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe archive path: {member.filename}")
    destination = output / member.filename
    if not destination.resolve().is_relative_to(output.resolve()):
        raise ValueError(f"output escapes destination: {member.filename}")
    source_bytes = archive.read(member)  # ZipFile checks the member CRC here.
    with Image.open(io.BytesIO(source_bytes)) as image:
        pages = []
        for page in range(image.n_frames):
            image.seek(page)
            pages.append({"page": page, "width": image.width, "height": image.height,
                          "mode": image.mode,
                          "tags": {str(k): json_tag(v) for k, v in image.tag_v2.items()}})
        image.seek(0)
        if any(p["width"] >= image.width and p["height"] >= image.height for p in pages[1:]):
            raise ValueError(f"not a descending pyramid with page 0 full resolution: {member.filename}")
        full = image.copy()
        pixels = np.array(image)
        if pixels.ndim != 2:
            raise ValueError(f"expected grayscale phase-contrast image: {member.filename}")
        # Preserve descriptive/calibration tags; structural TIFF offsets are rebuilt.
        tags = TiffImagePlugin.ImageFileDirectory_v2()
        for key in [270, 274, 282, 283, 296, 306, 315, 33432]:
            if key in image.tag_v2:
                tags[key] = image.tag_v2[key]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tif.tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        full.save(temporary, format="TIFF", compression="tiff_deflate", tiffinfo=tags)
        with Image.open(temporary) as check:
            decoded = np.array(check)
            if (check.n_frames != 1 or check.mode != full.mode or decoded.dtype != pixels.dtype
                    or not np.array_equal(decoded, pixels)):
                raise ValueError(f"lossless pixel verification failed: {member.filename}")
        record = {"path": member.filename, "source_bytes": len(source_bytes),
                  "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                  "source_crc32": member.CRC, "output_bytes": temporary.stat().st_size,
                  "output_sha256": sha256_file(temporary), "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                  "dtype": str(pixels.dtype), "width": full.width, "height": full.height,
                  "selected_page": 0, "compression": "tiff_deflate", "pixels_identical": True,
                  "source_pages": pages}
        temporary.replace(destination)
        return record
    finally:
        temporary.unlink(missing_ok=True)


def repack_archive(archive_path: Path, output: Path, archive_sha256: str,
                   *, workers: int = 4, progress=None) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    output.mkdir(parents=True, exist_ok=True)
    cell = archive_path.stem
    database = output / f"{cell}_repack.sqlite"
    policy = {"archive_sha256": archive_sha256, "version": "page0-deflate-v1"}
    with sqlite3.connect(database) as db, zipfile.ZipFile(archive_path) as archive:
        db.execute("CREATE TABLE IF NOT EXISTS policy (value TEXT NOT NULL)")
        stored = db.execute("SELECT value FROM policy").fetchone()
        if stored and json.loads(stored[0]) != policy:
            raise ValueError("archive/policy differs from the existing repack manifest")
        if not stored:
            db.execute("INSERT INTO policy VALUES (?)", (json.dumps(policy),))
        db.execute("CREATE TABLE IF NOT EXISTS images (path TEXT PRIMARY KEY, record TEXT NOT NULL)")
        db.commit()
        members = sorted((m for m in archive.infolist()
                          if m.filename.startswith(cell + "/ScanData/") and m.filename.endswith("-Ph.tif")),
                         key=lambda m: m.filename)
        if not members or len({m.filename for m in members}) != len(members):
            raise ValueError("archive must contain unique phase-contrast images")
        expected_paths = {m.filename for m in members}
        existing_paths = {r[0] for r in db.execute("SELECT path FROM images")}
        if not existing_paths <= expected_paths:
            raise ValueError("manifest contains images absent from this archive")
        missing = []
        reused = 0
        for member in members:
            previous = db.execute("SELECT record FROM images WHERE path=?", (member.filename,)).fetchone()
            if previous:
                record = json.loads(previous[0])
                destination = output / member.filename
                if destination.is_file() and sha256_file(destination) == record["output_sha256"]:
                    reused += 1
                    continue
            missing.append(member)
        # Conservative: enough for source-size output plus 10 GiB operating reserve.
        required = sum(m.file_size for m in missing) + 10 * 1024**3
        if shutil.disk_usage(output).free < required:
            raise ValueError("insufficient free space for repack and 10 GiB reserve")
        if progress:
            progress(reused, len(members))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            completed = reused
            for start in range(0, len(missing), 32):
                chunk = missing[start:start + 32]
                for record in executor.map(lambda member: convert_member(archive, member, output), chunk):
                    db.execute("INSERT OR REPLACE INTO images VALUES (?, ?)", (record["path"], json.dumps(record)))
                    completed += 1
                db.commit()
                if progress:
                    progress(completed, len(members))
        records = (json.loads(r[0]) for r in db.execute("SELECT record FROM images ORDER BY path"))
        original_bytes = output_bytes = count = 0
        # Export portable JSONL so consumers do not need SQLite.
        manifest = output / f"{cell}_repack_manifest.jsonl"
        temporary = manifest.with_suffix(".jsonl.tmp")
        with temporary.open("w") as handle:
            for record in records:
                original_bytes += record["source_bytes"]
                output_bytes += record["output_bytes"]
                count += 1
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        temporary.replace(manifest)
        if count != len(members):
            raise ValueError("repack manifest count does not match archive inventory")
    return {"archive": archive_path.name, "archive_sha256": archive_sha256,
            "n_images": count, "source_image_bytes": original_bytes, "repacked_image_bytes": output_bytes,
            "saved_bytes": original_bytes-output_bytes, "saved_percent": 100*(1-output_bytes/original_bytes),
            "manifest_sha256": sha256_file(manifest), "manifest": manifest.name,
            "completed_at": datetime.now(timezone.utc).isoformat(), "reused_images": reused,
            "policy": policy, "original_archive_retained": True}
