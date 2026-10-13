#!/usr/bin/env python3
"""Run expanded external evidence queries for the AD candidate panel."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from src.ad_cohort_utils import load_cohort_registry
from src.expanded_database_evidence_utils import run_expanded_database_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/ad_external_cohorts.yaml")
    parser.add_argument("--output", default="results/independent_ad_cohorts/external_evidence")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_cohort_registry(args.registry)
    root = Path(registry["_project_root"])
    output = Path(args.output).expanduser()
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}")
        shutil.rmtree(output)
    tables = run_expanded_database_evidence(registry["candidate_genes"], output)
    for resource, table in tables.items():
        print(f"{resource}: {table.shape[0]} rows")
    print(f"Expanded evidence completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
