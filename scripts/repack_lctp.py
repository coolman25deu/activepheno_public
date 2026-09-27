"""Wait for verified downloads, then repack phase-contrast TIFFs losslessly."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time

from activepheno.data.repack import repack_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/lctp"))
    parser.add_argument("--output-root", type=Path, default=Path("data/repacked/lctp"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--archives", nargs="+", choices=["A549-WT.zip", "U2OS-WT.zip"],
                        default=["A549-WT.zip", "U2OS-WT.zip"],
                        help="process selected archives; useful when one is already complete")
    parser.add_argument("--wait", action="store_true", help="wait for downloader checksum verification")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".repack.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory = json.loads((args.raw_root / "figshare_article.json").read_text())
        files = {f["name"]: f for f in inventory["files"]}
        for name in dict.fromkeys(args.archives):
            status_path = args.output_root / (name + ".repack.status.json")

            def status(state: str, **extra) -> None:
                record = {"archive": name, "state": state,
                          "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
                temp = status_path.with_suffix(".json.tmp")
                temp.write_text(json.dumps(record, indent=2) + "\n")
                temp.replace(status_path)

            try:
                status("waiting_for_verified_download")
                while True:
                    download_status = args.raw_root / (name + ".status.json")
                    check = json.loads(download_status.read_text()) if download_status.exists() else {}
                    if check.get("state") == "complete":
                        break
                    if check.get("state") == "failed":
                        raise ValueError(f"download failed: {check.get('error')}")
                    if not args.wait:
                        raise ValueError(f"archive not yet checksum-verified: {name}")
                    time.sleep(15)
                archive = args.raw_root / name
                if (archive.stat().st_size != files[name]["size"]
                        or check.get("md5") != files[name]["computed_md5"]
                        or not check.get("sha256")):
                    raise ValueError("archive does not match downloader verification record")
                before = archive.stat()
                print(f"Repacking {name}", flush=True)
                report = repack_archive(archive, args.output_root, check["sha256"], workers=args.workers,
                    progress=lambda done, total: status("repacking", completed_images=done, total_images=total))
                after = archive.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("archive changed during repacking")
                status("complete", **{k: v for k, v in report.items() if k != "archive"})
                print(json.dumps(report), flush=True)
            except Exception as exc:
                status("failed", error=str(exc))
                raise


if __name__ == "__main__":
    main()
