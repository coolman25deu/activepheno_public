import importlib.util
import io
from pathlib import Path
import struct
import zipfile


def test_zip64_directory_preserves_remote_member_offsets():
    path = Path(__file__).parents[1] / "scripts/download_lctp_sample.py"
    spec = importlib.util.spec_from_file_location("download_sample", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("example.tif", b"image")
    data = buffer.getvalue()
    directory_start = data.index(b"PK\x01\x02")
    end = data.index(b"PK\x05\x06")
    directory = data[directory_start:end]
    zip64 = struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0,
                        1, 1, len(directory), directory_start)
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, end, 1)
    classic = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 65535, 65535,
                          len(directory), 0xFFFFFFFF, 0)
    entries = module.directory_entries(directory + zip64 + locator + classic)
    assert len(entries) == 1
    assert entries[0].filename == "example.tif"
    assert entries[0].header_offset == 0
