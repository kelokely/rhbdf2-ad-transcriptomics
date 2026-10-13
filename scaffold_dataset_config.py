#!/usr/bin/env python3
"""Inspect a processed H5AD file and scaffold a dataset config YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

from src.dataset_onboarding_utils import scaffold_dataset_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaffold a config YAML for an additional processed H5AD dataset."
    )
    parser.add_argument("--dataset-name", required=True, help="Dataset name to use in the scaffolded config.")
    parser.add_argument("--input-path", required=True, help="Path to the processed .h5ad file.")
    parser.add_argument(
        "--output-config",
        required=True,
        help="Destination YAML path, for example configs/gse_example.yaml",
    )
    parser.add_argument(
        "--gene-of-interest",
        default="RHBDF2",
        help="Gene of interest to center the scaffold on. Default: RHBDF2",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    if input_path.suffix.lower() != ".h5ad":
        raise SystemExit("Input must be a processed .h5ad file.")
    adata = ad.read_h5ad(input_path)
    result = scaffold_dataset_config(
        adata=adata,
        dataset_name=args.dataset_name,
        input_path=input_path,
        output_config_path=args.output_config,
        gene_of_interest=args.gene_of_interest,
    )
    print("Dataset scaffold written successfully.")
    print(f"Config: {result.output_config_path}")
    print(f"Metadata suggestions: {result.metadata_suggestions_path}")
    print(f"Chosen matrix source: {result.matrix_source}")
    print(f"Suggested diagnosis column: {result.diagnosis_column}")
    print(f"Suggested donor column: {result.donor_column}")
    print(f"Suggested cluster column: {result.cluster_column}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
