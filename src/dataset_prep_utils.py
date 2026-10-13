"""Utilities for preparing validation datasets into analysis-ready H5AD files."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import re
import ssl
import subprocess
import tarfile
from typing import Iterable
import urllib.request

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io as scipy_io
from scipy import sparse


GSE146639_FILENAME_RE = re.compile(
    r"^(?P<gsm>GSM\d+)_(?P<donor_id>[^_]+)_(?P<diagnosis>CTR|AD)_(?P<sample_id>[^_]+)_(?P<library_id>S\d+)_completeCounts\.txt\.gz$"
)
GSE161045_FILENAME_RE = re.compile(
    r"^(?P<gsm>GSM\d+)_(?P<sample_code>(?P<diagnosis_prefix>CTRL|AD|PD)\d+)_(?P<file_kind>barcodes|features|matrix)\.(?P<extension>tsv|mtx)\.gz$"
)
GSE161045_DIAGNOSIS_LABELS = {
    "CTRL": "Control",
    "AD": "Alzheimer's disease",
    "PD": "Parkinson's disease",
}
GSE161045_BROAD_CELLTYPE_MARKERS = {
    "microglia": ["CSF1R", "CX3CR1", "AIF1", "TYROBP", "FCER1G", "C1QA", "C1QB", "C1QC", "P2RY12", "TMEM119"],
    "astro": ["GFAP", "AQP4", "ALDH1L1", "FGFR3", "SLC1A3"],
    "oligo": ["MBP", "PLP1", "MOG", "MAG", "MOBP"],
    "opc": ["VCAN", "PDGFRA", "CSPG4", "OLIG1", "OLIG2"],
    "vascular": ["FLT1", "PECAM1", "PDGFRB", "RGS5", "DCN", "CLDN5"],
    "tcell_nk": ["CD3D", "CD3E", "TRAC", "IL7R", "LTB", "CCL5", "NKG7", "KLRD1"],
}
GSE161045_MICROGLIA_HOMEOSTATIC_MARKERS = ["P2RY12", "CX3CR1", "CSF1R", "TMEM119", "C1QA", "C1QB", "C1QC"]
GSE161045_MICROGLIA_ACTIVATED_MARKERS = ["APOE", "B2M", "HLA-DRA", "FTL", "GPNMB", "CST7", "CLEC7A", "ITGAX", "LPL", "TYROBP", "SYK"]
GSE157783_DIAGNOSIS_LABELS = {
    "C": "Control",
    "CTRL": "Control",
    "PD": "Parkinson's disease",
    "IPD": "Parkinson's disease",
}
GSE157783_ENSEMBL_LOOKUP_URL = "https://rest.ensembl.org/lookup/id"
GSE157783_ENSEMBL_BIOMART_URL = "https://www.ensembl.org/biomart/martservice"
GSE157783_PURITY_MICROGLIA_MARKERS = ["P2RY12", "CX3CR1", "CSF1R", "AIF1", "TYROBP", "FCER1G", "SPI1", "LST1"]
GSE157783_PURITY_TCELL_MARKERS = ["CD3D", "CD3E", "CD3G", "TRBC1", "TRBC2", "CD2", "IL7R", "LTB", "CD8A", "CD8B", "NKG7"]
GSE157783_PURITY_NEURON_MARKERS = ["SNAP25", "RBFOX3", "SYT1", "SLC17A7", "GAD1", "GAD2"]
GSE157783_PURITY_OLIGO_MARKERS = ["MBP", "MOG", "PLP1", "MOBP", "MAG"]
GSE157783_PURITY_ASTROCYTE_MARKERS = ["GFAP", "AQP4", "ALDH1L1", "SLC1A3"]


@dataclass
class PreparedDatasetSummary:
    """Small summary for one prepared dataset."""

    dataset_name: str
    n_cells: int
    n_genes: int
    n_donors: int
    diagnosis_counts: dict[str, int]
    output_path: Path


def parse_gse146639_completecounts_filename(path: str | Path) -> dict[str, str]:
    """Extract donor and diagnosis metadata from a GEO completeCounts filename."""
    name = Path(path).name
    match = GSE146639_FILENAME_RE.match(name)
    if not match:
        raise ValueError(f"Unrecognized GSE146639 completeCounts filename: {name}")
    metadata = match.groupdict()
    metadata["dataset_name"] = "GSE146639"
    metadata["technology"] = "bc_SmartSeq2"
    return metadata


def parse_gse161045_triplet_filename(path: str | Path) -> dict[str, str]:
    """Extract sample metadata from a GSE161045 triplet filename."""
    name = Path(path).name
    match = GSE161045_FILENAME_RE.match(name)
    if not match:
        raise ValueError(f"Unrecognized GSE161045 filename: {name}")
    metadata = match.groupdict()
    prefix = metadata["diagnosis_prefix"]
    metadata["dataset_name"] = "GSE161045"
    metadata["diagnosis"] = GSE161045_DIAGNOSIS_LABELS[prefix]
    metadata["diagnosis_short"] = prefix
    metadata["donor_id"] = metadata["sample_code"]
    metadata["technology"] = "10x_snRNAseq"
    return metadata


def load_gse146639_gene_symbol_map(gene_tsv_path: str | Path) -> dict[str, str]:
    """Load the included 10x genes.tsv annotation file as an Ensembl-to-symbol map."""
    gene_tsv_path = Path(gene_tsv_path)
    genes = pd.read_csv(
        gene_tsv_path,
        sep="\t",
        header=None,
        compression="gzip",
        names=["ensembl_id", "gene_symbol", "feature_type"],
    )
    genes = genes.drop_duplicates(subset="ensembl_id")
    mapping = {
        str(row.ensembl_id): str(row.gene_symbol) if pd.notna(row.gene_symbol) and str(row.gene_symbol) else str(row.ensembl_id)
        for row in genes.itertuples(index=False)
    }
    return mapping


def prepare_gse146639_smartseq2_h5ad(
    raw_dir: str | Path,
    output_path: str | Path,
    gene_annotation_path: str | Path,
) -> PreparedDatasetSummary:
    """Combine the GEO barcoded Smart-seq2 count files into one analysis-ready H5AD."""
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    gene_symbol_map = load_gse146639_gene_symbol_map(gene_annotation_path)

    files = sorted(raw_dir.glob("GSM*_completeCounts.txt.gz"))
    if not files:
        raise FileNotFoundError(
            f"No GSE146639 completeCounts files were found in {raw_dir}."
        )

    gene_order = collect_union_gene_ids(files, preferred_order=gene_symbol_map.keys())
    gene_to_index = {gene_id: idx for idx, gene_id in enumerate(gene_order)}

    matrices: list[sparse.csr_matrix] = []
    obs_rows: list[dict[str, object]] = []

    for file_path in files:
        file_metadata = parse_gse146639_completecounts_filename(file_path)
        counts = pd.read_csv(file_path, sep="\t", compression="gzip")
        gene_ids = counts.iloc[:, 0].astype(str).tolist()
        if not gene_ids:
            continue

        values = counts.iloc[:, 1:].to_numpy(dtype=np.int32, copy=False)
        local_matrix = sparse.csr_matrix(values.T)
        projection = build_gene_projection_matrix(gene_ids, gene_to_index, n_output_genes=len(gene_order))
        matrices.append(local_matrix @ projection)

        for barcode in counts.columns[1:]:
            obs_rows.append(
                {
                    "obs_name": f"{file_metadata['gsm']}|{barcode}",
                    "barcode": str(barcode),
                    **file_metadata,
                }
            )

    if not matrices:
        raise ValueError("No matrices were assembled from the GSE146639 completeCounts files.")

    matrix = sparse.vstack(matrices, format="csr")
    matrix, var, mapping_table = collapse_ensembl_matrix_to_gene_symbols(matrix, gene_order, gene_symbol_map)

    obs = pd.DataFrame(obs_rows).set_index("obs_name")
    if matrix.shape[0] != obs.shape[0]:
        raise ValueError(
            f"Cell metadata mismatch while preparing GSE146639: matrix has {matrix.shape[0]} rows but obs has {obs.shape[0]} rows."
        )

    obs["total_counts"] = np.asarray(matrix.sum(axis=1)).ravel().astype(int)
    obs["n_genes_by_counts"] = matrix.getnnz(axis=1).astype(int)

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.uns["dataset_preparation"] = {
        "dataset_name": "GSE146639",
        "source": "GEO supplementary completeCounts files",
        "technology": "bc_SmartSeq2",
        "n_source_files": len(files),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)

    mapping_table_path = output_path.with_name(output_path.stem + "_gene_mapping.csv")
    mapping_table.to_csv(mapping_table_path, index=False)

    diagnosis_counts = {
        str(key): int(value)
        for key, value in obs["diagnosis"].value_counts().sort_index().items()
    }
    return PreparedDatasetSummary(
        dataset_name="GSE146639",
        n_cells=int(adata.n_obs),
        n_genes=int(adata.n_vars),
        n_donors=int(obs["donor_id"].nunique()),
        diagnosis_counts=diagnosis_counts,
        output_path=output_path,
    )


def prepare_gse161045_striatal_glia_h5ad(
    raw_dir: str | Path,
    output_path: str | Path,
    allowed_diagnosis_prefixes: Iterable[str] | None = None,
) -> PreparedDatasetSummary:
    """Combine the GSE161045 10x triplets into one analysis-ready H5AD file."""
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    allowed_prefixes = _normalize_allowed_gse161045_diagnoses(allowed_diagnosis_prefixes)

    grouped_files = _collect_gse161045_triplets(raw_dir, allowed_prefixes)
    if not grouped_files:
        allowed_text = ", ".join(sorted(allowed_prefixes))
        raise FileNotFoundError(
            f"No GSE161045 triplets were found in {raw_dir} for diagnoses: {allowed_text}."
        )

    matrices: list[sparse.csr_matrix] = []
    obs_rows: list[dict[str, object]] = []
    reference_gene_ids: list[str] | None = None
    reference_gene_symbols: list[str] | None = None
    reference_feature_types: list[str] | None = None

    for sample_code, sample_files in sorted(grouped_files.items()):
        metadata = parse_gse161045_triplet_filename(sample_files["matrix"])
        features = pd.read_csv(
            sample_files["features"],
            sep="\t",
            header=None,
            compression="gzip",
            names=["ensembl_id", "gene_symbol", "feature_type"],
        )
        gene_ids = features["ensembl_id"].astype(str).tolist()
        gene_symbols = features["gene_symbol"].astype(str).tolist()
        feature_types = features["feature_type"].astype(str).tolist()
        if reference_gene_ids is None:
            reference_gene_ids = gene_ids
            reference_gene_symbols = gene_symbols
            reference_feature_types = feature_types
        elif gene_ids != reference_gene_ids:
            raise ValueError(
                f"GSE161045 features mismatch detected for sample {sample_code}; "
                "the preparation utility currently expects a shared 10x feature order."
            )

        matrix = scipy_io.mmread(sample_files["matrix"]).tocsr().transpose()
        barcodes = pd.read_csv(
            sample_files["barcodes"],
            sep="\t",
            header=None,
            compression="gzip",
            names=["barcode"],
        )["barcode"].astype(str).tolist()
        if matrix.shape[0] != len(barcodes):
            raise ValueError(
                f"Barcode mismatch for GSE161045 sample {sample_code}: "
                f"matrix rows={matrix.shape[0]} but barcodes={len(barcodes)}."
            )

        matrices.append(matrix)
        for barcode in barcodes:
            obs_rows.append(
                {
                    "obs_name": f"{metadata['gsm']}|{barcode}",
                    "barcode": barcode,
                    "sample_code": metadata["sample_code"],
                    "donor_id": metadata["donor_id"],
                    "diagnosis": metadata["diagnosis"],
                    "diagnosis_short": metadata["diagnosis_short"],
                    "gsm": metadata["gsm"],
                    "dataset_name": metadata["dataset_name"],
                    "technology": metadata["technology"],
                }
            )

    if not matrices or reference_gene_ids is None or reference_gene_symbols is None:
        raise ValueError("No matrices were assembled from the GSE161045 triplets.")

    matrix = sparse.vstack(matrices, format="csr")
    gene_symbol_map = dict(zip(reference_gene_ids, reference_gene_symbols, strict=True))
    matrix, var, mapping_table = collapse_ensembl_matrix_to_gene_symbols(
        matrix=matrix,
        ensembl_ids=reference_gene_ids,
        gene_symbol_map=gene_symbol_map,
    )
    if reference_feature_types is not None:
        feature_type_map = dict(zip(reference_gene_ids, reference_feature_types, strict=True))
        mapping_table["feature_type"] = mapping_table["ensembl_id"].map(feature_type_map)

    obs = pd.DataFrame(obs_rows).set_index("obs_name")
    if matrix.shape[0] != obs.shape[0]:
        raise ValueError(
            f"Cell metadata mismatch while preparing GSE161045: matrix has {matrix.shape[0]} rows but obs has {obs.shape[0]} rows."
        )

    obs["total_counts"] = np.asarray(matrix.sum(axis=1)).ravel().astype(int)
    obs["n_genes_by_counts"] = matrix.getnnz(axis=1).astype(int)

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.uns["dataset_preparation"] = {
        "dataset_name": "GSE161045",
        "source": "GEO supplementary 10x triplets",
        "technology": "10x_snRNAseq",
        "allowed_diagnosis_prefixes": sorted(allowed_prefixes),
        "n_source_samples": len(grouped_files),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)

    mapping_table_path = output_path.with_name(output_path.stem + "_gene_mapping.csv")
    mapping_table.to_csv(mapping_table_path, index=False)

    diagnosis_counts = {
        str(key): int(value)
        for key, value in obs["diagnosis"].value_counts().sort_index().items()
    }
    return PreparedDatasetSummary(
        dataset_name="GSE161045",
        n_cells=int(adata.n_obs),
        n_genes=int(adata.n_vars),
        n_donors=int(obs["donor_id"].nunique()),
        diagnosis_counts=diagnosis_counts,
        output_path=output_path,
    )


def derive_gse161045_microglia_subset_h5ad(
    input_path: str | Path,
    output_path: str | Path,
    min_microglia_score: float = 0.05,
) -> PreparedDatasetSummary:
    """Derive an explicit microglia subset from the prepared GSE161045 glia H5AD."""
    import scanpy as sc

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"GSE161045 input H5AD not found: {input_path}")
    if min_microglia_score < 0:
        raise ValueError("'min_microglia_score' must be greater than or equal to 0.")

    adata = ad.read_h5ad(input_path)
    analysis_adata = adata.copy()
    sc.pp.normalize_total(analysis_adata, target_sum=10000)
    sc.pp.log1p(analysis_adata)

    broad_score_columns: list[str] = []
    for class_name, markers in GSE161045_BROAD_CELLTYPE_MARKERS.items():
        score_column = f"{class_name}_score"
        broad_score_columns.append(score_column)
        adata.obs[score_column] = _compute_mean_log1p_marker_score(analysis_adata, markers)
    adata.obs["dominant_broad_cell_class"] = (
        adata.obs[broad_score_columns].idxmax(axis=1).str.replace("_score", "", regex=False)
    )
    non_microglia_columns = [column for column in broad_score_columns if column != "microglia_score"]
    adata.obs["microglia_enrichment_score"] = (
        adata.obs["microglia_score"] - adata.obs[non_microglia_columns].max(axis=1)
    )

    microglia_mask = (
        adata.obs["dominant_broad_cell_class"].eq("microglia")
        & (adata.obs["microglia_score"] >= float(min_microglia_score))
    )
    if int(microglia_mask.sum()) == 0:
        raise ValueError(
            "No cells passed the GSE161045 microglia derivation filter. "
            "Try lowering 'min_microglia_score' or inspect the prepared H5AD."
        )

    adata.obs["microglia_homeostatic_score"] = _compute_mean_log1p_marker_score(
        analysis_adata,
        GSE161045_MICROGLIA_HOMEOSTATIC_MARKERS,
    )
    adata.obs["microglia_activated_score"] = _compute_mean_log1p_marker_score(
        analysis_adata,
        GSE161045_MICROGLIA_ACTIVATED_MARKERS,
    )
    adata.obs["microglia_activation_balance"] = (
        adata.obs["microglia_activated_score"] - adata.obs["microglia_homeostatic_score"]
    )

    microglia = adata[microglia_mask].copy()
    activation_threshold = float(microglia.obs["microglia_activation_balance"].median())
    microglia.obs["microglia_activation_proxy_state"] = np.where(
        microglia.obs["microglia_activation_balance"] >= activation_threshold,
        "higher_activation",
        "lower_activation",
    )
    microglia.uns["dataset_preparation"] = {
        "dataset_name": "GSE161045_microglia",
        "source": str(input_path),
        "subset_method": (
            "Broad cell class scored from normalized/log1p marker expression; "
            "cells retained when dominant_broad_cell_class == 'microglia' "
            f"and microglia_score >= {float(min_microglia_score):.3f}."
        ),
        "broad_celltype_markers": GSE161045_BROAD_CELLTYPE_MARKERS,
        "microglia_homeostatic_markers": GSE161045_MICROGLIA_HOMEOSTATIC_MARKERS,
        "microglia_activated_markers": GSE161045_MICROGLIA_ACTIVATED_MARKERS,
        "microglia_activation_balance_threshold": activation_threshold,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    microglia.write_h5ad(output_path)

    broad_counts = (
        adata.obs.groupby(["dominant_broad_cell_class", "diagnosis"], observed=True)
        .size()
        .reset_index(name="n_cells")
    )
    broad_counts.to_csv(output_path.with_name(output_path.stem + "_broad_cell_class_counts.csv"), index=False)

    donor_counts = (
        microglia.obs.groupby(["donor_id", "diagnosis"], observed=True)
        .size()
        .reset_index(name="n_cells")
    )
    donor_counts.to_csv(output_path.with_name(output_path.stem + "_donor_counts.csv"), index=False)

    diagnosis_counts = {
        str(key): int(value)
        for key, value in microglia.obs["diagnosis"].value_counts().sort_index().items()
    }
    return PreparedDatasetSummary(
        dataset_name="GSE161045_microglia",
        n_cells=int(microglia.n_obs),
        n_genes=int(microglia.n_vars),
        n_donors=int(microglia.obs["donor_id"].nunique()),
        diagnosis_counts=diagnosis_counts,
        output_path=output_path,
    )


def load_gse157783_family_sample_metadata(soft_path: str | Path) -> pd.DataFrame:
    """Parse donor-level metadata from the GSE157783 GEO family SOFT file."""
    soft_path = Path(soft_path)
    if not soft_path.exists():
        raise FileNotFoundError(f"GSE157783 family SOFT file not found: {soft_path}")

    rows: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    with gzip.open(soft_path, "rt", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                if current is not None:
                    rows.append(current)
                current = {"sample_geo_accession": line.split("=", 1)[1].strip()}
                continue
            if current is None:
                continue
            if line.startswith("!Sample_title = "):
                current["sample_title"] = line.split("=", 1)[1].strip()
            if line.startswith("!Sample_geo_accession = "):
                current["sample_geo_accession"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_source_name_ch1 = "):
                current["source_name"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_characteristics_ch1 = "):
                value = line.split("=", 1)[1].strip()
                if ":" not in value:
                    continue
                key, raw_value = value.split(":", 1)
                normalized_key = key.strip().lower()
                parsed_value = raw_value.strip()
                if normalized_key == "disease state":
                    current["disease_state_raw"] = parsed_value
                elif normalized_key == "age at death":
                    current["age_at_death"] = _safe_int(parsed_value)
                elif normalized_key == "sex":
                    current["sex"] = parsed_value
                elif normalized_key == "pmi (h)":
                    current["pmi_h"] = _safe_float(parsed_value)
                elif normalized_key == "tissue":
                    current["tissue"] = parsed_value
                elif normalized_key == "molecule subtype":
                    current["molecule_subtype"] = parsed_value
        if current is not None:
            rows.append(current)

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise ValueError(f"No sample metadata records were parsed from {soft_path}.")

    if "sample_title" not in metadata.columns:
        raise ValueError(f"GSE157783 SOFT parsing did not recover sample titles from {soft_path}.")
    metadata["donor_id"] = metadata["sample_title"].astype(str).map(_normalize_gse157783_donor_id)
    metadata["diagnosis"] = metadata["disease_state_raw"].map(_normalize_gse157783_disease_state)
    missing_diagnosis = metadata["diagnosis"].isna()
    if missing_diagnosis.any():
        metadata.loc[missing_diagnosis, "diagnosis"] = metadata.loc[missing_diagnosis, "donor_id"].map(
            _infer_gse157783_diagnosis_from_donor
        )
    metadata["diagnosis_short"] = metadata["donor_id"].astype(str).str.extract(r"^(PD|C)", expand=False)
    return metadata


def prepare_gse157783_midbrain_h5ad(
    umi_tar_path: str | Path,
    cell_tar_path: str | Path,
    genes_tar_path: str | Path,
    output_path: str | Path,
    sample_metadata_path: str | Path | None = None,
    gene_symbol_cache_path: str | Path | None = None,
) -> PreparedDatasetSummary:
    """Build an analysis-ready H5AD from the processed GSE157783 midbrain files."""
    umi_tar_path = Path(umi_tar_path)
    cell_tar_path = Path(cell_tar_path)
    genes_tar_path = Path(genes_tar_path)
    output_path = Path(output_path)
    sample_metadata_path = Path(sample_metadata_path) if sample_metadata_path else None
    gene_symbol_cache_path = Path(gene_symbol_cache_path) if gene_symbol_cache_path else None

    cells = _read_single_member_tsv_from_tar(cell_tar_path)
    genes = _read_single_member_tsv_from_tar(genes_tar_path)
    matrix, matrix_barcodes = _read_gse157783_umi_matrix_from_tar(
        umi_tar_path,
        expected_n_genes=int(genes.shape[0]),
    )

    if not {"barcode", "cell_ontology", "patient"}.issubset(cells.columns):
        raise ValueError(
            "GSE157783 cell metadata must contain 'barcode', 'cell_ontology', and 'patient' columns."
        )
    if "gene" not in genes.columns:
        raise ValueError("GSE157783 genes metadata must contain a 'gene' column with Ensembl IDs.")

    n_cells = int(cells.shape[0])
    n_genes = int(genes.shape[0])
    expected_barcodes = cells["barcode"].astype(str).tolist()
    if matrix.shape != (n_cells, n_genes):
        raise ValueError(
            "GSE157783 matrix dimensions do not match the processed metadata: "
            f"matrix shape={matrix.shape}, cells={n_cells}, genes={n_genes}."
        )
    if matrix_barcodes != expected_barcodes:
        raise ValueError(
            "GSE157783 matrix barcode order does not match the processed cell metadata order."
        )

    ensembl_ids = genes["gene"].astype(str).tolist()
    gene_symbol_map = resolve_human_ensembl_gene_symbols(
        ensembl_ids,
        cache_path=gene_symbol_cache_path,
    )
    collapsed_matrix, var, mapping_table = collapse_ensembl_matrix_to_gene_symbols(
        matrix=matrix,
        ensembl_ids=ensembl_ids,
        gene_symbol_map=gene_symbol_map,
    )
    var = var.copy()
    var.index = pd.Index(var["gene_symbol"].astype(str), name="gene_symbol")
    var["gene_symbol"] = var.index.astype(str)

    obs = cells.copy()
    obs["barcode"] = obs["barcode"].astype(str)
    obs["patient"] = obs["patient"].astype(str)
    obs["donor_id"] = obs["patient"].map(_normalize_gse157783_donor_id)
    obs["diagnosis"] = obs["donor_id"].map(_infer_gse157783_diagnosis_from_donor)
    obs["diagnosis_short"] = obs["donor_id"].str.extract(r"^(PD|C)", expand=False)
    obs["dataset_name"] = "GSE157783"
    obs["technology"] = "10x_snRNAseq"

    if sample_metadata_path is not None and sample_metadata_path.exists():
        sample_metadata = load_gse157783_family_sample_metadata(sample_metadata_path)
        metadata_columns = [
            "donor_id",
            "sample_title",
            "sample_geo_accession",
            "disease_state_raw",
            "age_at_death",
            "sex",
            "pmi_h",
            "tissue",
            "molecule_subtype",
            "source_name",
        ]
        obs = obs.merge(
            sample_metadata[metadata_columns],
            on="donor_id",
            how="left",
            validate="many_to_one",
        )

    obs = _coerce_object_columns_to_strings(obs)

    obs["obs_name"] = obs["donor_id"].astype(str) + "|" + obs["barcode"].astype(str)
    obs = obs.set_index("obs_name")
    obs["total_counts"] = np.asarray(collapsed_matrix.sum(axis=1)).ravel().astype(int)
    obs["n_genes_by_counts"] = collapsed_matrix.getnnz(axis=1).astype(int)

    adata = ad.AnnData(X=collapsed_matrix, obs=obs, var=var)
    adata.var = var.copy()
    adata.var_names = pd.Index(var.index.astype(str), name="gene_symbol")
    adata.uns["dataset_preparation"] = {
        "dataset_name": "GSE157783",
        "source": "GEO processed midbrain UMI, cell, and genes tarballs",
        "technology": "10x_snRNAseq",
        "sample_metadata_path": str(sample_metadata_path) if sample_metadata_path else None,
        "gene_symbol_cache_path": str(gene_symbol_cache_path) if gene_symbol_cache_path else None,
        "n_source_cells": n_cells,
        "n_source_genes": n_genes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)

    reloaded = ad.read_h5ad(output_path, backed="r")
    reloaded_var_names = reloaded.var_names.astype(str)
    has_numeric_placeholder_index = bool(
        len(reloaded_var_names)
        and reloaded_var_names[0].isdigit()
        and "RHBDF2" not in set(reloaded_var_names[: min(1000, len(reloaded_var_names))])
    )
    if has_numeric_placeholder_index:
        raise RuntimeError(
            "Prepared GSE157783 H5AD was written with numeric placeholder gene names instead of symbols."
        )

    mapping_output = output_path.with_name(output_path.stem + "_gene_mapping.csv")
    mapping_output.parent.mkdir(parents=True, exist_ok=True)
    mapping_table.to_csv(mapping_output, index=False)

    diagnosis_counts = {
        str(key): int(value)
        for key, value in obs["diagnosis"].value_counts().sort_index().items()
    }
    return PreparedDatasetSummary(
        dataset_name="GSE157783",
        n_cells=int(adata.n_obs),
        n_genes=int(adata.n_vars),
        n_donors=int(obs["donor_id"].nunique()),
        diagnosis_counts=diagnosis_counts,
        output_path=output_path,
    )


def derive_gse157783_microglia_subset_h5ad(
    input_path: str | Path,
    output_path: str | Path,
) -> PreparedDatasetSummary:
    """Subset the prepared GSE157783 dataset to annotated microglia and add activation proxy labels."""
    import scanpy as sc

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"GSE157783 input H5AD not found: {input_path}")

    adata = ad.read_h5ad(input_path)
    if "cell_ontology" not in adata.obs.columns:
        raise ValueError("GSE157783 input H5AD must include a 'cell_ontology' column.")

    microglia_mask = adata.obs["cell_ontology"].astype(str).eq("Microglia")
    if int(microglia_mask.sum()) == 0:
        raise ValueError("No annotated microglia cells were found in the prepared GSE157783 H5AD.")

    microglia = adata[microglia_mask].copy()
    analysis_adata = microglia.copy()
    sc.pp.normalize_total(analysis_adata, target_sum=10000)
    sc.pp.log1p(analysis_adata)

    microglia.obs["microglia_homeostatic_score"] = _compute_mean_log1p_marker_score(
        analysis_adata,
        GSE161045_MICROGLIA_HOMEOSTATIC_MARKERS,
    )
    microglia.obs["microglia_activated_score"] = _compute_mean_log1p_marker_score(
        analysis_adata,
        GSE161045_MICROGLIA_ACTIVATED_MARKERS,
    )
    microglia.obs["microglia_activation_balance"] = (
        microglia.obs["microglia_activated_score"] - microglia.obs["microglia_homeostatic_score"]
    )
    activation_threshold = float(microglia.obs["microglia_activation_balance"].median())
    microglia.obs["microglia_activation_proxy_state"] = np.where(
        microglia.obs["microglia_activation_balance"] >= activation_threshold,
        "higher_activation",
        "lower_activation",
    )
    microglia.uns["dataset_preparation"] = {
        "dataset_name": "GSE157783_microglia",
        "source": str(input_path),
        "subset_method": "Annotated cell_ontology == 'Microglia'",
        "microglia_homeostatic_markers": GSE161045_MICROGLIA_HOMEOSTATIC_MARKERS,
        "microglia_activated_markers": GSE161045_MICROGLIA_ACTIVATED_MARKERS,
        "microglia_activation_balance_threshold": activation_threshold,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    microglia.write_h5ad(output_path)

    celltype_counts = (
        adata.obs["cell_ontology"]
        .value_counts(dropna=False)
        .rename_axis("cell_ontology")
        .reset_index(name="n_cells")
    )
    celltype_counts.to_csv(output_path.with_name(output_path.stem + "_cell_ontology_counts.csv"), index=False)

    donor_counts = (
        microglia.obs.groupby(["donor_id", "diagnosis"], observed=True)
        .size()
        .reset_index(name="n_cells")
    )
    donor_counts.to_csv(output_path.with_name(output_path.stem + "_donor_counts.csv"), index=False)

    diagnosis_counts = {
        str(key): int(value)
        for key, value in microglia.obs["diagnosis"].value_counts().sort_index().items()
    }
    return PreparedDatasetSummary(
        dataset_name="GSE157783_microglia",
        n_cells=int(microglia.n_obs),
        n_genes=int(microglia.n_vars),
        n_donors=int(microglia.obs["donor_id"].nunique()),
        diagnosis_counts=diagnosis_counts,
        output_path=output_path,
    )


def derive_gse157783_microglia_purity_qc_h5ad(
    input_path: str | Path,
    output_path: str | Path,
    min_microglia_score: float = 0.0,
    min_microglia_markers: int = 2,
    max_allowed_tcell_markers: int = 1,
    contaminant_score_margin: float = 0.5,
) -> PreparedDatasetSummary:
    """Apply a conservative purity-QC filter to the derived GSE157783 microglia object."""
    import scanpy as sc

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"GSE157783 microglia input H5AD not found: {input_path}")

    adata = ad.read_h5ad(input_path)
    analysis_adata = adata.copy()
    sc.pp.normalize_total(analysis_adata, target_sum=10000)
    sc.pp.log1p(analysis_adata)

    marker_sets = {
        "microglia": GSE157783_PURITY_MICROGLIA_MARKERS,
        "tcell": GSE157783_PURITY_TCELL_MARKERS,
        "neuron": GSE157783_PURITY_NEURON_MARKERS,
        "oligo": GSE157783_PURITY_OLIGO_MARKERS,
        "astrocyte": GSE157783_PURITY_ASTROCYTE_MARKERS,
    }

    for label, markers in marker_sets.items():
        adata.obs[f"purity_{label}_score"] = _compute_mean_log1p_marker_score(analysis_adata, markers)
        adata.obs[f"purity_{label}_marker_count"] = _count_detected_markers(adata, markers)

    metadata_microglia_support = (
        adata.obs["cell_ontology"].astype(str).eq("Microglia").to_numpy()
        if "cell_ontology" in adata.obs.columns
        else np.zeros(adata.n_obs, dtype=bool)
    )
    microglia_score = adata.obs["purity_microglia_score"].to_numpy(dtype=float)
    tcell_score = adata.obs["purity_tcell_score"].to_numpy(dtype=float)
    neuron_score = adata.obs["purity_neuron_score"].to_numpy(dtype=float)
    oligo_score = adata.obs["purity_oligo_score"].to_numpy(dtype=float)
    astrocyte_score = adata.obs["purity_astrocyte_score"].to_numpy(dtype=float)

    microglia_marker_count = adata.obs["purity_microglia_marker_count"].to_numpy(dtype=int)
    tcell_marker_count = adata.obs["purity_tcell_marker_count"].to_numpy(dtype=int)
    neuron_marker_count = adata.obs["purity_neuron_marker_count"].to_numpy(dtype=int)
    oligo_marker_count = adata.obs["purity_oligo_marker_count"].to_numpy(dtype=int)
    astrocyte_marker_count = adata.obs["purity_astrocyte_marker_count"].to_numpy(dtype=int)

    microglia_support = (
        (microglia_score >= float(min_microglia_score))
        | (microglia_marker_count >= int(min_microglia_markers))
        | metadata_microglia_support
    )
    weak_microglia_identity = microglia_marker_count < int(min_microglia_markers)
    multiple_tcell_markers = tcell_marker_count > int(max_allowed_tcell_markers)
    tcell_dominant = (tcell_score > (microglia_score + float(contaminant_score_margin))) | multiple_tcell_markers
    neuron_dominant = (
        (neuron_score > (microglia_score + float(contaminant_score_margin)))
        & (neuron_marker_count >= 2)
        & weak_microglia_identity
    )
    oligo_dominant = (
        (oligo_score > (microglia_score + float(contaminant_score_margin)))
        & (oligo_marker_count >= 2)
        & weak_microglia_identity
    )
    astrocyte_dominant = (
        (astrocyte_score > (microglia_score + float(contaminant_score_margin)))
        & (astrocyte_marker_count >= 2)
        & weak_microglia_identity
    )

    keep_mask = (
        microglia_support
        & ~tcell_dominant
        & ~neuron_dominant
        & ~oligo_dominant
        & ~astrocyte_dominant
        & ~multiple_tcell_markers
    )
    exclusion_reasons = _build_purity_exclusion_reasons(
        microglia_support=microglia_support,
        tcell_dominant=tcell_dominant,
        neuron_dominant=neuron_dominant,
        oligo_dominant=oligo_dominant,
        astrocyte_dominant=astrocyte_dominant,
        multiple_tcell_markers=multiple_tcell_markers,
    )

    adata.obs["purity_qc_keep"] = keep_mask
    adata.obs["purity_qc_exclusion_reason"] = exclusion_reasons

    filtered = adata[keep_mask].copy()
    filtered_analysis = filtered.copy()
    sc.pp.normalize_total(filtered_analysis, target_sum=10000)
    sc.pp.log1p(filtered_analysis)

    filtered.obs["microglia_homeostatic_score"] = _compute_mean_log1p_marker_score(
        filtered_analysis,
        GSE161045_MICROGLIA_HOMEOSTATIC_MARKERS,
    )
    filtered.obs["microglia_activated_score"] = _compute_mean_log1p_marker_score(
        filtered_analysis,
        GSE161045_MICROGLIA_ACTIVATED_MARKERS,
    )
    filtered.obs["microglia_activation_balance"] = (
        filtered.obs["microglia_activated_score"] - filtered.obs["microglia_homeostatic_score"]
    )
    activation_threshold = float(filtered.obs["microglia_activation_balance"].median())
    filtered.obs["microglia_activation_proxy_state"] = np.where(
        filtered.obs["microglia_activation_balance"] >= activation_threshold,
        "higher_activation",
        "lower_activation",
    )
    filtered.uns["dataset_preparation"] = {
        "dataset_name": "GSE157783_microglia_purity_qc",
        "source": str(input_path),
        "subset_method": "Conservative purity-QC rerun from annotated GSE157783 microglia object",
        "purity_qc": {
            "min_microglia_score": float(min_microglia_score),
            "min_microglia_markers": int(min_microglia_markers),
            "max_allowed_tcell_markers": int(max_allowed_tcell_markers),
            "contaminant_score_margin": float(contaminant_score_margin),
            "microglia_markers": GSE157783_PURITY_MICROGLIA_MARKERS,
            "tcell_markers": GSE157783_PURITY_TCELL_MARKERS,
            "neuron_markers": GSE157783_PURITY_NEURON_MARKERS,
            "oligo_markers": GSE157783_PURITY_OLIGO_MARKERS,
            "astrocyte_markers": GSE157783_PURITY_ASTROCYTE_MARKERS,
            "input_cells": int(adata.n_obs),
            "retained_cells": int(filtered.n_obs),
            "removed_cells": int(adata.n_obs - filtered.n_obs),
        },
        "microglia_homeostatic_markers": GSE161045_MICROGLIA_HOMEOSTATIC_MARKERS,
        "microglia_activated_markers": GSE161045_MICROGLIA_ACTIVATED_MARKERS,
        "microglia_activation_balance_threshold": activation_threshold,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_h5ad(output_path)

    annotation_columns = [
        column
        for column in [
            "donor_id",
            "diagnosis",
            "cell_ontology",
            "purity_microglia_score",
            "purity_microglia_marker_count",
            "purity_tcell_score",
            "purity_tcell_marker_count",
            "purity_neuron_score",
            "purity_neuron_marker_count",
            "purity_oligo_score",
            "purity_oligo_marker_count",
            "purity_astrocyte_score",
            "purity_astrocyte_marker_count",
            "purity_qc_keep",
            "purity_qc_exclusion_reason",
        ]
        if column in adata.obs.columns
    ]
    adata.obs[annotation_columns].to_csv(
        output_path.with_name(output_path.stem + "_purity_qc_cell_annotations.csv"),
        index=True,
        index_label="cell_id",
    )

    summary = pd.DataFrame(
        [
            {
                "input_cells": int(adata.n_obs),
                "retained_cells": int(filtered.n_obs),
                "removed_cells": int(adata.n_obs - filtered.n_obs),
                "fraction_retained": float(filtered.n_obs / adata.n_obs) if int(adata.n_obs) else np.nan,
                "min_microglia_score": float(min_microglia_score),
                "min_microglia_markers": int(min_microglia_markers),
                "max_allowed_tcell_markers": int(max_allowed_tcell_markers),
                "contaminant_score_margin": float(contaminant_score_margin),
                "removed_insufficient_microglia_support": int((~microglia_support).sum()),
                "removed_tcell_module_exceeds_microglia": int(tcell_dominant.sum()),
                "removed_neuron_module_exceeds_microglia": int(neuron_dominant.sum()),
                "removed_oligo_module_exceeds_microglia": int(oligo_dominant.sum()),
                "removed_astrocyte_module_exceeds_microglia": int(astrocyte_dominant.sum()),
                "removed_multiple_tcell_markers": int(multiple_tcell_markers.sum()),
            }
        ]
    )
    summary.to_csv(output_path.with_name(output_path.stem + "_purity_qc_summary.csv"), index=False)

    flag_counts = (
        pd.Series(exclusion_reasons, name="reason")
        .replace("", "retained")
        .value_counts(dropna=False)
        .rename_axis("purity_qc_status")
        .reset_index(name="n_cells")
    )
    flag_counts.to_csv(output_path.with_name(output_path.stem + "_purity_qc_flag_counts.csv"), index=False)

    donor_counts = (
        filtered.obs.groupby(["donor_id", "diagnosis"], observed=True)
        .size()
        .reset_index(name="n_cells")
    )
    donor_counts.to_csv(output_path.with_name(output_path.stem + "_donor_counts.csv"), index=False)

    diagnosis_counts = {
        str(key): int(value)
        for key, value in filtered.obs["diagnosis"].value_counts().sort_index().items()
    }
    return PreparedDatasetSummary(
        dataset_name="GSE157783_microglia_purity_qc",
        n_cells=int(filtered.n_obs),
        n_genes=int(filtered.n_vars),
        n_donors=int(filtered.obs["donor_id"].nunique()),
        diagnosis_counts=diagnosis_counts,
        output_path=output_path,
    )


def collect_union_gene_ids(
    files: Iterable[Path],
    preferred_order: Iterable[str] | None = None,
) -> list[str]:
    """Collect a deterministic union of gene IDs across completeCounts files."""
    union: set[str] = set()
    for file_path in files:
        gene_ids = pd.read_csv(
            file_path,
            sep="\t",
            compression="gzip",
            usecols=[0],
        ).iloc[:, 0].astype(str)
        union.update(gene_ids.tolist())

    ordered: list[str] = []
    if preferred_order is not None:
        for gene_id in preferred_order:
            if gene_id in union:
                ordered.append(str(gene_id))
                union.remove(str(gene_id))
    ordered.extend(sorted(union))
    return ordered


def build_gene_projection_matrix(
    gene_ids: list[str],
    gene_to_index: dict[str, int],
    n_output_genes: int,
) -> sparse.csr_matrix:
    """Build a sparse matrix that projects file-specific genes into a global gene union."""
    row_index = np.arange(len(gene_ids), dtype=np.int32)
    col_index = np.array([gene_to_index[gene_id] for gene_id in gene_ids], dtype=np.int32)
    data = np.ones(len(gene_ids), dtype=np.int8)
    return sparse.csr_matrix(
        (data, (row_index, col_index)),
        shape=(len(gene_ids), n_output_genes),
    )


def collapse_ensembl_matrix_to_gene_symbols(
    matrix: sparse.csr_matrix,
    ensembl_ids: list[str],
    gene_symbol_map: dict[str, str],
) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Collapse Ensembl-indexed columns to gene-symbol columns by summing duplicates."""
    gene_symbols = [gene_symbol_map.get(ensembl_id, ensembl_id) for ensembl_id in ensembl_ids]
    symbol_codes, unique_symbols = pd.factorize(pd.Index(gene_symbols), sort=False)
    aggregator = sparse.csr_matrix(
        (
            np.ones(len(ensembl_ids), dtype=np.int8),
            (np.arange(len(ensembl_ids), dtype=np.int32), symbol_codes.astype(np.int32)),
        ),
        shape=(len(ensembl_ids), len(unique_symbols)),
    )
    collapsed = matrix @ aggregator

    symbol_counts = pd.Series(gene_symbols).value_counts()
    var = pd.DataFrame(
        {
            "gene_symbol": unique_symbols.astype(str),
            "n_ensembl_collapsed": [int(symbol_counts[symbol]) for symbol in unique_symbols],
        },
        index=pd.Index(unique_symbols.astype(str), name="gene_symbol"),
    )
    mapping_table = pd.DataFrame(
        {
            "ensembl_id": ensembl_ids,
            "gene_symbol": gene_symbols,
        }
    )
    return collapsed.tocsr(), var, mapping_table


def _collect_gse161045_triplets(
    raw_dir: Path,
    allowed_prefixes: set[str],
) -> dict[str, dict[str, Path]]:
    triplets: dict[str, dict[str, Path]] = {}
    for file_path in sorted(raw_dir.glob("GSM*_*.gz")):
        try:
            metadata = parse_gse161045_triplet_filename(file_path)
        except ValueError:
            continue
        if metadata["diagnosis_short"] not in allowed_prefixes:
            continue
        triplets.setdefault(metadata["sample_code"], {})[metadata["file_kind"]] = file_path

    complete_triplets: dict[str, dict[str, Path]] = {}
    for sample_code, sample_files in triplets.items():
        missing = {"barcodes", "features", "matrix"} - set(sample_files)
        if missing:
            raise FileNotFoundError(
                f"Incomplete GSE161045 triplet for sample {sample_code}; missing: {', '.join(sorted(missing))}."
            )
        complete_triplets[sample_code] = sample_files
    return complete_triplets


def _normalize_allowed_gse161045_diagnoses(
    allowed_diagnosis_prefixes: Iterable[str] | None,
) -> set[str]:
    if allowed_diagnosis_prefixes is None:
        return {"CTRL", "PD"}
    normalized = {str(value).strip().upper() for value in allowed_diagnosis_prefixes if str(value).strip()}
    unsupported = normalized - set(GSE161045_DIAGNOSIS_LABELS)
    if unsupported:
        raise ValueError(
            f"Unsupported GSE161045 diagnosis prefixes: {', '.join(sorted(unsupported))}. "
            f"Choose from: {', '.join(sorted(GSE161045_DIAGNOSIS_LABELS))}."
        )
    return normalized or {"CTRL", "PD"}


def _compute_mean_log1p_marker_score(
    adata: ad.AnnData,
    markers: Iterable[str],
) -> np.ndarray:
    present_markers = [str(marker) for marker in markers if str(marker) in adata.var_names]
    if not present_markers:
        return np.zeros(adata.n_obs, dtype=float)
    matrix = adata[:, present_markers].X
    if sparse.issparse(matrix):
        return np.asarray(matrix.mean(axis=1)).ravel().astype(float)
    return np.asarray(matrix, dtype=float).mean(axis=1)


def _count_detected_markers(
    adata: ad.AnnData,
    markers: Iterable[str],
) -> np.ndarray:
    present_markers = [str(marker) for marker in markers if str(marker) in adata.var_names]
    if not present_markers:
        return np.zeros(adata.n_obs, dtype=int)
    matrix = adata[:, present_markers].X
    if sparse.issparse(matrix):
        detected = matrix > 0
        return np.asarray(detected.sum(axis=1)).ravel().astype(int)
    return (np.asarray(matrix, dtype=float) > 0).sum(axis=1).astype(int)


def _build_purity_exclusion_reasons(
    *,
    microglia_support: np.ndarray,
    tcell_dominant: np.ndarray,
    neuron_dominant: np.ndarray,
    oligo_dominant: np.ndarray,
    astrocyte_dominant: np.ndarray,
    multiple_tcell_markers: np.ndarray,
) -> list[str]:
    reasons: list[str] = []
    for index in range(len(microglia_support)):
        cell_reasons: list[str] = []
        if not bool(microglia_support[index]):
            cell_reasons.append("insufficient_microglia_support")
        if bool(tcell_dominant[index]):
            cell_reasons.append("tcell_module_exceeds_microglia")
        if bool(neuron_dominant[index]):
            cell_reasons.append("neuron_module_exceeds_microglia")
        if bool(oligo_dominant[index]):
            cell_reasons.append("oligo_module_exceeds_microglia")
        if bool(astrocyte_dominant[index]):
            cell_reasons.append("astrocyte_module_exceeds_microglia")
        if bool(multiple_tcell_markers[index]):
            cell_reasons.append("multiple_tcell_markers_detected")
        reasons.append(";".join(cell_reasons))
    return reasons


def _coerce_object_columns_to_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed object columns into write-safe strings for AnnData/HDF5 export."""
    converted = frame.copy()
    object_columns = converted.select_dtypes(include=["object"]).columns.tolist()
    for column in object_columns:
        converted[column] = converted[column].map(lambda value: "" if pd.isna(value) else str(value))
    return converted


def resolve_human_ensembl_gene_symbols(
    ensembl_ids: Iterable[str],
    cache_path: str | Path | None = None,
    batch_size: int = 500,
) -> dict[str, str]:
    """Resolve human Ensembl gene IDs to symbols using the Ensembl REST lookup endpoint."""
    normalized_ids = [str(ensembl_id) for ensembl_id in ensembl_ids]
    mapping: dict[str, str] = {}
    cache_path = Path(cache_path) if cache_path else None

    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path)
        if {"ensembl_id", "gene_symbol"}.issubset(cached.columns):
            mapping.update(
                {
                    str(row.ensembl_id): str(row.gene_symbol)
                    for row in cached.itertuples(index=False)
                    if pd.notna(row.ensembl_id) and pd.notna(row.gene_symbol)
                }
            )

    missing_ids = [ensembl_id for ensembl_id in dict.fromkeys(normalized_ids) if ensembl_id not in mapping]
    if missing_ids:
        biomart_mapping = _download_ensembl_biomart_human_gene_symbol_map()
        for ensembl_id in missing_ids:
            if biomart_mapping.get(ensembl_id):
                mapping[ensembl_id] = biomart_mapping[ensembl_id]
        missing_ids = [ensembl_id for ensembl_id in dict.fromkeys(normalized_ids) if ensembl_id not in mapping]

    for start in range(0, len(missing_ids), batch_size):
        batch = missing_ids[start : start + batch_size]
        if not batch:
            continue
        payload = _fetch_ensembl_lookup_batch(batch)
        for ensembl_id in batch:
            entry = payload.get(ensembl_id, {}) or {}
            mapping[ensembl_id] = str(entry.get("display_name") or ensembl_id)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "ensembl_id": list(dict.fromkeys(normalized_ids)),
                "gene_symbol": [mapping.get(ensembl_id, ensembl_id) for ensembl_id in dict.fromkeys(normalized_ids)],
            }
        ).to_csv(cache_path, index=False)

    return {ensembl_id: mapping.get(ensembl_id, ensembl_id) for ensembl_id in normalized_ids}


def _read_single_member_tsv_from_tar(tar_path: Path) -> pd.DataFrame:
    with tarfile.open(tar_path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one file inside {tar_path}, found {len(members)}.")
        extracted = archive.extractfile(members[0])
        if extracted is None:
            raise ValueError(f"Could not extract {members[0].name} from {tar_path}.")
        return pd.read_csv(extracted, sep="\t")


def _read_gse157783_umi_matrix_from_tar(
    tar_path: Path,
    expected_n_genes: int | None = None,
) -> tuple[sparse.csr_matrix, list[str]]:
    """Stream the dense GSE157783 UMI TSV from its tarball into a sparse cell-by-gene matrix."""
    with tarfile.open(tar_path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one matrix file inside {tar_path}, found {len(members)}.")
        extracted = archive.extractfile(members[0])
        if extracted is None:
            raise ValueError(f"Could not extract {members[0].name} from {tar_path}.")

        header = extracted.readline().decode("utf-8").rstrip("\n")
        barcodes = [value for value in header.split("\t") if value]
        n_cells = len(barcodes)
        row_chunks: list[np.ndarray] = []
        col_chunks: list[np.ndarray] = []
        data_chunks: list[np.ndarray] = []
        n_gene_rows = 0

        for raw_line in extracted:
            line = raw_line.decode("utf-8").rstrip("\n")
            values = np.fromstring(line, sep="\t", dtype=np.int32)
            if values.size != n_cells:
                raise ValueError(
                    f"GSE157783 UMI row {n_gene_rows} has {values.size} columns but expected {n_cells}."
                )
            nonzero = np.nonzero(values)[0]
            if nonzero.size:
                row_chunks.append(nonzero.astype(np.int32, copy=False))
                col_chunks.append(np.full(nonzero.size, n_gene_rows, dtype=np.int32))
                data_chunks.append(values[nonzero].astype(np.int32, copy=False))
            n_gene_rows += 1

    if expected_n_genes is not None and n_gene_rows != expected_n_genes:
        raise ValueError(
            f"GSE157783 UMI matrix has {n_gene_rows} gene rows but expected {expected_n_genes}."
        )

    if data_chunks:
        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        data = np.concatenate(data_chunks)
    else:
        rows = np.array([], dtype=np.int32)
        cols = np.array([], dtype=np.int32)
        data = np.array([], dtype=np.int32)

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_cells, n_gene_rows), dtype=np.int32)
    return matrix, barcodes


def _normalize_gse157783_donor_id(value: str | object) -> str:
    donor_id = str(value).strip()
    if donor_id.startswith("IPD"):
        return "PD" + donor_id[3:]
    return donor_id


def _normalize_gse157783_disease_state(value: str | object) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    if "Parkinson" in text:
        return "Parkinson's disease"
    if "Healthy control" in text or text.lower() == "control":
        return "Control"
    return None


def _infer_gse157783_diagnosis_from_donor(donor_id: str | object) -> str:
    normalized = _normalize_gse157783_donor_id(donor_id)
    for prefix, label in GSE157783_DIAGNOSIS_LABELS.items():
        if normalized.startswith(prefix):
            return label
    raise ValueError(f"Could not infer GSE157783 diagnosis from donor id: {donor_id}")


def _safe_int(value: str | object) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _safe_float(value: str | object) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fetch_ensembl_lookup_batch(batch: list[str]) -> dict[str, object]:
    request = urllib.request.Request(
        GSE157783_ENSEMBL_LOOKUP_URL,
        data=json.dumps({"ids": batch}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    context = None
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = None

    try:
        with urllib.request.urlopen(request, timeout=60, context=context) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        completed = subprocess.run(
            [
                "curl",
                "-s",
                "-X",
                "POST",
                GSE157783_ENSEMBL_LOOKUP_URL,
                "-H",
                "Content-Type: application/json",
                "-H",
                "Accept: application/json",
                "-d",
                json.dumps({"ids": batch}),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)


def _download_ensembl_biomart_human_gene_symbol_map() -> dict[str, str]:
    query = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<!DOCTYPE Query>"
        "<Query virtualSchemaName=\"default\" formatter=\"TSV\" header=\"1\" uniqueRows=\"1\" datasetConfigVersion=\"0.6\">"
        "<Dataset name=\"hsapiens_gene_ensembl\" interface=\"default\">"
        "<Attribute name=\"ensembl_gene_id\"/>"
        "<Attribute name=\"hgnc_symbol\"/>"
        "</Dataset>"
        "</Query>"
    )
    completed = subprocess.run(
        [
            "curl",
            "-sG",
            GSE157783_ENSEMBL_BIOMART_URL,
            "--data-urlencode",
            f"query={query}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {}

    mapping: dict[str, str] = {}
    for line in lines[1:]:
        parts = line.split("\t")
        if not parts:
            continue
        ensembl_id = parts[0].strip()
        gene_symbol = parts[1].strip() if len(parts) > 1 else ""
        if ensembl_id:
            mapping[ensembl_id] = gene_symbol or ensembl_id
    return mapping
