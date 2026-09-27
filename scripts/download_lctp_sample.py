"""Download official metadata and three images using ZIP byte ranges (stdlib only).

This is an explicit acquisition step. Preparation itself never accesses the network.
"""
import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import struct
from urllib.request import Request, urlopen
import zipfile
import zlib

ARCHIVE_URL = "https://ndownloader.figshare.com/files/60040382"
ARCHIVE_SIZE = 42977225126
GITHUB = "https://raw.githubusercontent.com/edvinforsgren/LCTP/main/"


def fetch(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {"User-Agent": "ActivePheno-data-preparation/0.1"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
        # Distinct query strings avoid proxies serving a cached different range.
        url += f"?activepheno_range={start}-{end}"
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        if start is not None:
            expected = end - start + 1
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or content_range != f"bytes {start}-{end}/{ARCHIVE_SIZE}":
                raise ValueError(f"server did not honor exact byte range: {content_range}")
        else:
            expected = 5 * 1024 * 1024  # Metadata only, never accidentally download an archive.
        data = response.read(expected + 1)
        if len(data) > expected or (start is not None and len(data) != expected):
            raise ValueError("unexpected response length")
        return data


def directory_entries(directory: bytes) -> list[zipfile.ZipInfo]:
    """Rebase ZIP64 directory pointers for a standalone directory byte buffer.

    Member header offsets remain their absolute offsets in the remote archive.
    """
    local = bytearray(directory)
    end = local.rfind(b"PK\x06\x06")
    locator = local.rfind(b"PK\x06\x07")
    if end < 0 or locator < 0:
        raise ValueError("missing ZIP64 directory records")
    struct.pack_into("<Q", local, end + 48, 0)
    struct.pack_into("<Q", local, locator + 8, end)
    with zipfile.ZipFile(io.BytesIO(local)) as archive:
        return archive.infolist()


def download(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    provenance: dict = {"archive_url": ARCHIVE_URL, "archive_size": ARCHIVE_SIZE, "sources": [], "files": []}
    for name, url in {
        "figshare_article.json": "https://api.figshare.com/v2/articles/29315642",
        "README.md": "https://ndownloader.figshare.com/files/60981295",
        "metadata.parquet.gzip": GITHUB + "dino_feature_extraction/data/metadata.parquet.gzip",
        "Supplementary_Table_6.html": GITHUB + "assets/Supplementary_Table_6.html",
    }.items():
        data = fetch(url)
        (root / name).write_bytes(data)
        provenance["sources"].append({"path": name, "url": url, "sha256": hashlib.sha256(data).hexdigest()})
    tail = fetch(ARCHIVE_URL, ARCHIVE_SIZE - 65536, ARCHIVE_SIZE - 1)
    position = tail.rfind(b"PK\x06\x06")
    if position < 0:
        raise ValueError("expected ZIP64 archive end record")
    record = struct.unpack_from("<4sQ2H2I4Q", tail, position)
    directory_offset = record[-1]
    if not 0 < ARCHIVE_SIZE - directory_offset < 64 * 1024 * 1024:
        raise ValueError("unexpected archive directory size")
    directory = fetch(ARCHIVE_URL, directory_offset, ARCHIVE_SIZE - 1)
    (root / "a549_zip_directory.bin").write_bytes(directory)
    entries = directory_entries(directory)
    starts: dict[str, str] = {}
    for entry in entries:
        parts = entry.filename.split("/")
        if len(parts) == 7 and parts[1] == "ScanData" and parts[-1].endswith(".tif"):
            key, stamp = f"{parts[0]}/{parts[5]}", "/".join(parts[2:5])
            datetime.strptime(stamp, "%y%m/%d/%H%M")
            starts[key] = min(starts.get(key, stamp), stamp)
    (root / "plate_starts.json").write_text(json.dumps(starts, indent=2) + "\n")
    selected = sorted((e for e in entries if "/ScanData/" in e.filename and e.filename.endswith("/433/A2-1-Ph.tif")), key=lambda e: e.filename)[:3]
    if len(selected) != 3:
        raise ValueError("expected at least three observations for vessel 433, well A2")
    for entry in selected:
        offset = entry.header_offset
        header = fetch(ARCHIVE_URL, offset, offset + 29)
        fields = struct.unpack("<4s5H3I2H", header)
        if fields[0] != b"PK\x03\x04" or entry.compress_size > 16 * 1024 * 1024:
            raise ValueError("invalid or unexpectedly large image entry")
        payload_start = offset + 30 + fields[-2] + fields[-1]
        packed = fetch(ARCHIVE_URL, payload_start, payload_start + entry.compress_size - 1)
        if entry.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(packed, -15)
        elif entry.compress_type == zipfile.ZIP_STORED:
            data = packed
        else:
            raise ValueError("unsupported ZIP compression")
        if len(data) != entry.file_size or zlib.crc32(data) != entry.CRC:
            raise ValueError(f"image failed ZIP CRC validation: {entry.filename}")
        path = root / entry.filename
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("unsafe archive member path")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        provenance["files"].append({"path": entry.filename, "archive_offset": offset,
                                    "crc32": entry.CRC, "sha256": hashlib.sha256(data).hexdigest()})
        print(entry.filename)
    (root / "sample_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/lctp"))
    args = parser.parse_args()
    try:
        download(args.output_root)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Download failed: {exc}\n")
