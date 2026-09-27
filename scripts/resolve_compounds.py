"""Freeze the compound universe and structures from local source metadata."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import pandas as pd

from activepheno.data.compounds import compounds_from_wells
from activepheno.data.lctp import sha256_file
from activepheno.chemistry.structures import read_catalog, read_overrides, resolve_compounds
from activepheno.seed import seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/chemistry"))
    parser.add_argument("--exclude-controls", action="store_true")
    parser.add_argument("--exclude-unlabelled", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print("Resolved config:", json.dumps(vars(args), default=str, sort_keys=True))
    try:
        seed_everything(args.seed)
        compounds, exclusions = compounds_from_wells(pd.read_parquet(args.metadata),
            exclude_controls=args.exclude_controls, exclude_unlabelled=args.exclude_unlabelled)
        catalog = read_catalog(args.catalog)
        result, report = resolve_compounds(compounds, catalog, read_overrides(args.overrides))
        report.update(exclusions=exclusions, seed=args.seed, created_at=datetime.now(timezone.utc).isoformat(),
                      sources=[{"path": str(p), "sha256": sha256_file(p)} for p in [args.metadata, args.catalog, args.overrides] if p])
        print(json.dumps(report, indent=2))
        if not args.dry_run:
            if args.output_root.exists():
                raise ValueError(f"output already exists: {args.output_root}; choose a new directory")
            args.output_root.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=args.output_root.parent) as tmp:
                output = Path(tmp) / "result"
                output.mkdir()
                result.to_parquet(output / "compounds.parquet", index=False)
                catalog.to_csv(output / "source_catalog.csv", index=False)
                report["compounds_sha256"] = sha256_file(output / "compounds.parquet")
                (output / "compound_resolution_report.json").write_text(json.dumps(report, indent=2) + "\n")
                output.rename(args.output_root)
        return 0 if report["ready_for_chemistry"] else 1
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Compound resolution failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
