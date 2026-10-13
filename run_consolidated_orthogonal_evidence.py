#!/usr/bin/env python3
"""Run and consolidate orthogonal evidence for the full independent-AD panel."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from src.ad_cohort_utils import load_cohort_registry
from src.consolidated_orthogonal_evidence_utils import run_consolidated_orthogonal_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/ad_external_cohorts.yaml")
    parser.add_argument("--output", default="results/independent_ad_cohorts/orthogonal_evidence")
    parser.add_argument("--expanded-evidence", default="results/independent_ad_cohorts/external_evidence")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_cohort_registry(args.registry)
    root = Path(registry["_project_root"])
    output = _resolve(root, args.output)
    expanded = _resolve(root, args.expanded_evidence)
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}")
        shutil.rmtree(output)
    paths = run_consolidated_orthogonal_evidence(
        registry["candidate_genes"], output, expanded, args.timeout_seconds
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
