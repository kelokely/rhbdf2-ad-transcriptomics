#!/usr/bin/env python3
"""Prepare the GSE146639 Smart-seq2 microglia validation dataset as an H5AD file."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset_prep_utils import prepare_gse146639_smartseq2_h5ad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an analysis-ready H5AD file from GSE146639 Smart-seq2 supplementary counts."
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory containing the extracted GSE146639 supplementary files.",
    )
    parser.add_argument(
        "--gene-annotation",
        default="data/raw/GSM4403285_10X_Donor2019-010_genes.tsv.gz",
        help="Path to the included 10x genes.tsv.gz file used for Ensembl-to-symbol mapping.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/GSE146639_barcoded_smartseq2_microglia.h5ad",
        help="Destination H5AD path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_gse146639_smartseq2_h5ad(
        raw_dir=Path(args.raw_dir),
        output_path=Path(args.output),
        gene_annotation_path=Path(args.gene_annotation),
    )

    print("Prepared dataset successfully.")
    print(f"Dataset: {summary.dataset_name}")
    print(f"Output: {summary.output_path}")
    print(f"Cells: {summary.n_cells}")
    print(f"Genes: {summary.n_genes}")
    print(f"Donors: {summary.n_donors}")
    print(f"Diagnosis counts: {summary.diagnosis_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
