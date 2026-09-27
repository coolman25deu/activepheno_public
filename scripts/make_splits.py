"""Construct a split only after full trajectory integrity checks."""
import argparse
import json
from pathlib import Path
import tempfile

import pandas as pd

from activepheno.data.lctp import sha256_file
from activepheno.splits.generation import make_split
from activepheno.data.validation import validate_images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compounds", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--strategy", choices=["compound_disjoint", "scaffold_disjoint"], default="compound_disjoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempts", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print("Resolved config:", json.dumps(vars(args), default=str, sort_keys=True))
    if args.output_root.exists() and not args.dry_run:
        parser.exit(1, "Output already exists; choose a new directory\n")
    try:
        compounds = pd.read_parquet(args.compounds)
        observations = pd.read_parquet(args.observations)
        validate_images(observations, args.raw_root)
        manifest, report = make_split(compounds, observations, strategy=args.strategy, seed=args.seed, attempts=args.attempts)
        report["sources"] = {str(p): sha256_file(p) for p in [args.compounds, args.observations]}
        report["status"] = "success"
    except (ValueError, OSError, KeyError, TypeError) as exc:
        report = {"status": "failed", "reason": str(exc), "strategy": args.strategy, "seed": args.seed}
        if not args.dry_run:
            args.output_root.mkdir(parents=True)
            (args.output_root / "split_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1
    if not args.dry_run:
        args.output_root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output_root.parent) as tmp:
            output = Path(tmp) / "result"
            output.mkdir()
            manifest.to_parquet(output / f"{args.strategy}.parquet", index=False)
            (output / "split_report.json").write_text(json.dumps(report, indent=2) + "\n")
            output.rename(args.output_root)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
