#!/usr/bin/env python3
"""Create a cross-dataset candidate replication summary from finished run folders."""

from __future__ import annotations

import argparse

from src.validation_utils import build_cross_dataset_candidate_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge candidate evidence outputs from multiple finished runs."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        dest="run_dirs",
        required=True,
        help="Path to a finished results/<run_name> directory. Provide this flag multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/cross_dataset_validation",
        help="Directory where the merged validation CSVs should be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    merged, summary_path = build_cross_dataset_candidate_summary(
        run_dirs=args.run_dirs,
        output_dir=args.output_dir,
    )
    print("Cross-dataset validation summary written successfully.")
    print(f"Rows: {len(merged)}")
    print(f"Output: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
