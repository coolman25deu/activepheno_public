"""Offline preparation CLI."""
import argparse
import json
import logging
from pathlib import Path
import yaml

from activepheno.config import load_config
from activepheno.data.lctp import prepare
from activepheno.logging import configure_logging
from activepheno.seed import seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description="Index native LCTP ScanData images or explicit observation CSV")
    parser.add_argument("--config", type=Path)
    for name in ["raw-root", "metadata", "output-root", "plate-starts", "dataset-version"]:
        parser.add_argument(f"--{name}")
    parser.add_argument("--input-format", choices=["lctp", "csv"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--expected-interval-hours", type=float)
    parser.add_argument("--exclude-unlabelled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--exclude-controls", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing outputs")
    args = vars(parser.parse_args())
    config_path, dry_run = args.pop("config"), args.pop("dry_run")
    configure_logging()
    try:
        config = load_config(config_path, args)
        logging.info("Resolved config: %s", json.dumps(config.to_dict(), sort_keys=True))
        seed_everything(config.seed)
        manifest = prepare(config, dry_run=dry_run)
    except (ValueError, OSError, TypeError, KeyError, yaml.YAMLError) as exc:
        logging.error("Preparation failed: %s", exc)
        return 1
    print(json.dumps({k: manifest[k] for k in ["n_images", "n_series", "n_compounds", "n_moas", "exclusions"]}, indent=2))
    return 0
