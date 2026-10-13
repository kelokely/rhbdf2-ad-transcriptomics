"""Prepare ancillary source-data inputs without redistributing research data."""

from pathlib import Path
import argparse
import csv
import gzip
import sys

import anndata as ad
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tnf_cross_species_utils import prepare_gse148405
from run_tnf_cross_species import fetch_if_missing
from scripts.audit_ad_donor_metadata import write_seaad


def source_count_qc(matrix_path, obs):
    """Count detected source gene rows, not members of the targeted panel."""
    if not obs.index.is_unique:
        raise ValueError("Prepared cell identifiers are duplicated")
    with gzip.open(matrix_path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if not header or header[0] != "GeneID" or len(set(header[1:])) != len(header)-1:
            raise ValueError("Unexpected or duplicated source count header")
        lookup = {name: i+1 for i, name in enumerate(header[1:])}
        if any(str(cell) not in lookup for cell in obs.index):
            raise ValueError("Prepared cells do not match the source matrix")
        positions = [lookup[str(cell)] for cell in obs.index]
        total = np.zeros(len(obs), dtype=np.int64)
        detected = np.zeros(len(obs), dtype=np.int64)
        for row in reader:
            if len(row) != len(header):
                raise ValueError("Malformed source count row")
            if row[0].startswith("ERCC-"):
                continue
            values = np.array([int(row[i]) for i in positions], dtype=np.int64)
            if (values < 0).any():
                raise ValueError("Negative source counts")
            total += values
            detected += values > 0
    if not np.allclose(total, obs.source_library_counts.to_numpy(float)):
        raise ValueError("Source count totals disagree with prepared full-library denominators")
    result = obs[["mouse_id", "genotype", "region", "source_library_counts"]].copy()
    result.index.name = "cell_id"
    result["source_genes_detected"] = detected
    result["assigned_gene_counts"] = total
    result["fewer_than_500_detected_genes"] = detected < 500
    result["fewer_than_5000_assigned_gene_counts"] = total < 5000
    result["more_than_700000_assigned_gene_counts"] = total > 700000
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tnf_cross_species.yaml")
    parser.add_argument("--mouse", action="store_true", help="Prepare mouse panel and full-source QC")
    parser.add_argument("--download", action="store_true", help="Download missing registered mouse sources")
    parser.add_argument("--seaad-metadata", action="store_true", help="Join the supplied public SEA-AD donor workbook")
    args = parser.parse_args()
    if not args.mouse and not args.seaad_metadata:
        parser.error("Select --mouse and/or --seaad-metadata")
    if args.download and not args.mouse:
        parser.error("--download requires --mouse")
    if args.mouse:
        cfg = yaml.safe_load((ROOT / args.config).read_text())
        mouse, orthology = cfg["mouse"], cfg["orthology"]
        inputs = [(mouse["matrix_url"], ROOT/mouse["matrix_path"]),
                  (mouse["metadata_url"], ROOT/mouse["metadata_path"]),
                  (orthology["source_url"], ROOT/orthology["path"])]
        for url, path in inputs:
            if args.download:
                fetch_if_missing(url, path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing mouse source: {path}; use --download")
        prepared = ROOT/mouse["prepared_path"]
        if not prepared.is_file():
            prepare_gse148405(inputs[0][1], inputs[1][1], inputs[2][1], cfg["modules"],
                             prepared, cfg["direct_gene_checks"])
        obj = ad.read_h5ad(prepared, backed="r")
        try:
            qc = source_count_qc(inputs[0][1], obj.obs)
        finally:
            obj.file.close()
        output = ROOT/"report/jad_submission_audit_2026-10-12/mouse_source_cell_qc_flags.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        qc.to_csv(output)
        print(f"Wrote full-source count QC: {output} ({len(qc)} cells)")
    if args.seaad_metadata:
        source = ROOT/"data/external_ad_cohorts/SEAAD_MTG/source/sea-ad_cohort-donor-metadata_2025-07-25.xlsx"
        prepared = ROOT/"data/external_ad_cohorts/SEAAD_MTG/processed/SEAAD_MTG_targeted_canonical.h5ad"
        if not source.is_file() or not prepared.is_file():
            raise FileNotFoundError("SEA-AD donor workbook and primary prepared H5AD are required; see docs/DATA_PREPARATION.md")
        output = ROOT/"results/committee_extensions/donor_metadata_audit"
        output.mkdir(parents=True, exist_ok=True)
        write_seaad(source, prepared, output)
        print(f"Wrote public metadata join: {output}")


if __name__ == "__main__":
    main()
