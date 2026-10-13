#!/usr/bin/env python3
"""Download, prepare, and inspect independent Alzheimer cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ad_cohort_utils import (
    build_registry_status,
    canonical_metadata_summary,
    download_public_cohort,
    load_cohort_registry,
    prepare_registered_cohort,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="configs/ad_external_cohorts.yaml",
        help="Cohort registry YAML path.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Write and print cohort readiness status.")

    download = subparsers.add_parser("download", help="Download one public cohort's processed sources.")
    download.add_argument("--cohort", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare one downloaded cohort as canonical H5AD.")
    prepare.add_argument("--cohort", required=True)
    prepare.add_argument("--overwrite", action="store_true")

    inspect = subparsers.add_parser("inspect", help="Inspect canonical metadata in a prepared object.")
    inspect.add_argument("--cohort", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_cohort_registry(args.registry)
    project_root = Path(registry["_project_root"])
    status_path = project_root / "results" / "independent_ad_cohorts" / "cohort_registry_status.csv"

    if args.command == "status":
        status = build_registry_status(registry)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status.to_csv(status_path, index=False)
        print(status.to_string(index=False))
        print(f"\nSaved: {status_path}")
        return 0

    if args.command == "download":
        records = download_public_cohort(registry, args.cohort)
        for record in records:
            print(f"{record.status}: {record.path} ({record.size_bytes} bytes, sha256={record.sha256})")
        return 0

    output = prepare_registered_cohort(registry, args.cohort, overwrite=getattr(args, "overwrite", False))
    if args.command == "prepare":
        print(f"Prepared: {output}")
        return 0

    summary = canonical_metadata_summary(output)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
