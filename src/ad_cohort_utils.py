"""Independent Alzheimer cohort registry, download, and preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import tarfile
from typing import Any
import xml.etree.ElementTree as ET

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import requests
from scipy import sparse
import yaml

from .gene_utils import extract_expression_vector
from .cell_type_annotation_utils import (
    MAJOR_BRAIN_CELL_MARKERS,
    annotate_major_brain_cell_types,
    apply_basic_nucleus_qc,
    build_targeted_cell_type_object_from_geo_tar,
    read_geo_10x_triplet_tar,
    read_nested_geo_10x_tar,
)
from .stereo_spatial_utils import extract_cellbin_gef_files, prepare_spatial_cohort
from .seaad_utils import extract_h5ad_csr_columns, map_seaad_cell_types, read_h5ad_dataframe_column


CANONICAL_OBS_COLUMNS = (
    "donor_id",
    "analysis_unit",
    "pathology",
    "cell_type",
    "subpopulation",
    "brain_region",
    "cohort",
)


GSE138852_CELL_TYPE_MAP = {
    "astro": "Astrocytes",
    "endo": "Endothelial cells",
    "ex": "Excitatory neurons",
    "in": "Inhibitory neurons",
    "mg": "Microglia",
    "micro": "Microglia",
    "neuron": "Neurons",
    "oligo": "Oligodendrocytes",
    "opc": "OPCs",
    "per": "Pericytes",
    "unid": "Unassigned",
    "doublet": "Doublets",
}


@dataclass(frozen=True)
class DownloadRecord:
    """One downloaded or locally verified source file."""

    cohort: str
    url: str
    path: Path
    size_bytes: int
    sha256: str
    status: str


def load_cohort_registry(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the independent-cohort registry."""
    registry_path = Path(path).expanduser().resolve()
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    cohorts = registry.get("cohorts")
    if not isinstance(cohorts, dict) or not cohorts:
        raise ValueError("The cohort registry must contain a non-empty 'cohorts' mapping.")
    required = {"title", "access", "role", "adapter", "prepared_path"}
    for cohort, spec in cohorts.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Cohort '{cohort}' must be a mapping.")
        missing = sorted(required.difference(spec))
        if missing:
            raise ValueError(f"Cohort '{cohort}' is missing: {', '.join(missing)}")
        access = str(spec["access"]).lower()
        if access not in {"public", "controlled"}:
            raise ValueError(f"Cohort '{cohort}' has unsupported access mode '{access}'.")
        if access == "public" and not spec.get("sources"):
            raise ValueError(f"Public cohort '{cohort}' must define at least one source file.")
    registry["_registry_path"] = str(registry_path)
    registry["_project_root"] = str(registry_path.parent.parent)
    return registry


def build_registry_status(registry: dict[str, Any]) -> pd.DataFrame:
    """Report source, preparation, access, and donor-inference readiness."""
    root = Path(registry["_project_root"])
    rows: list[dict[str, object]] = []
    for cohort, spec in registry["cohorts"].items():
        source_rows = []
        for source in spec.get("sources", []):
            source_path = _resolve(root, spec["source_directory"]) / source["filename"]
            source_rows.append(_is_complete_file(source_path, source.get("expected_size_bytes")))
        prepared_path = _resolve(root, spec["prepared_path"])
        prepared_present = prepared_path.exists()
        if prepared_path.parent == _resolve(root, spec.get("source_directory", prepared_path.parent)):
            matching_source = next(
                (source for source in spec.get("sources", []) if source["filename"] == prepared_path.name),
                None,
            )
            if matching_source:
                prepared_present = _is_complete_file(
                    prepared_path, matching_source.get("expected_size_bytes")
                )
        rows.append(
            {
                "cohort": cohort,
                "title": spec["title"],
                "access": spec["access"],
                "role": spec["role"],
                "tissue": spec.get("tissue"),
                "nominal_donors": spec.get("nominal_donors"),
                "all_sources_present": bool(source_rows and all(source_rows)),
                "prepared_object_present": prepared_present,
                "prepared_path": str(prepared_path),
                "donor_inference_eligible": bool(spec.get("donor_inference_eligible", False)),
                "donor_ineligibility_reason": spec.get("donor_ineligibility_reason", ""),
                "access_note": spec.get("access_note", ""),
            }
        )
    return pd.DataFrame(rows)


def download_public_cohort(
    registry: dict[str, Any],
    cohort: str,
    chunk_size: int = 8 * 1024 * 1024,
    timeout_seconds: int = 120,
) -> list[DownloadRecord]:
    """Download public processed sources with resume support and provenance."""
    spec = _cohort_spec(registry, cohort)
    if str(spec["access"]).lower() != "public":
        raise PermissionError(spec.get("access_note") or f"Cohort '{cohort}' is controlled access.")
    root = Path(registry["_project_root"])
    source_dir = _resolve(root, spec["source_directory"])
    source_dir.mkdir(parents=True, exist_ok=True)
    records: list[DownloadRecord] = []
    for source in spec["sources"]:
        path = source_dir / source["filename"]
        _download_with_resume(
            source["url"],
            path,
            expected_size=source.get("expected_size_bytes"),
            chunk_size=chunk_size,
            timeout_seconds=timeout_seconds,
        )
        records.append(
            DownloadRecord(
                cohort=cohort,
                url=source["url"],
                path=path,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                status="verified",
            )
        )
    _write_download_manifest(source_dir / "download_manifest.json", records)
    return records


def prepare_registered_cohort(registry: dict[str, Any], cohort: str, overwrite: bool = False) -> Path:
    """Dispatch cohort preparation to an explicit, provenance-recorded adapter."""
    spec = _cohort_spec(registry, cohort)
    root = Path(registry["_project_root"])
    output_path = _resolve(root, spec["prepared_path"])
    if output_path.exists() and not overwrite:
        return output_path
    adapter = spec["adapter"]
    if adapter == "gse138852_csv":
        return prepare_gse138852(registry, cohort, overwrite=overwrite)
    if adapter == "geo_10x_tar_marker_annotation" and cohort == "GSE157827":
        return prepare_gse157827(registry, cohort, overwrite=overwrite)
    if adapter == "geo_10x_tar_marker_annotation" and cohort == "GSE237718":
        return prepare_gse237718(registry, cohort, overwrite=overwrite)
    if adapter == "gse268599_astrocyte_tar":
        return prepare_gse268599(registry, cohort, overwrite=overwrite)
    if adapter == "gse269906_stereo_cellbin_gef":
        return prepare_gse269906(registry, cohort, overwrite=overwrite)
    if adapter == "seaad_h5ad":
        return prepare_seaad_mtg(registry, cohort, overwrite=overwrite)
    if adapter == "rosmap_synapse_h5ad":
        if not output_path.exists():
            raise FileNotFoundError(f"Expected H5AD for {cohort} was not found: {output_path}")
        return output_path
    raise NotImplementedError(
        f"Preparation adapter '{adapter}' requires inspection of the downloaded archive before conversion."
    )


def prepare_seaad_mtg(registry: dict[str, Any], cohort: str, overwrite: bool = False) -> Path:
    """Create a compact raw-UMI object with author SEA-AD donor and taxonomy labels."""
    spec = _cohort_spec(registry, cohort)
    root = Path(registry["_project_root"])
    source_dir = _resolve(root, spec["source_directory"])
    source_path = source_dir / "SEAAD_MTG_RNAseq_final-nuclei.2026-06-22.h5ad"
    output_path = _resolve(root, spec["prepared_path"])
    if output_path.exists() and not overwrite:
        return output_path
    if not source_path.exists():
        raise FileNotFoundError(f"SEA-AD source H5AD was not found: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    retained_genes = list(registry["candidate_genes"])
    for genes in registry["programs"].values():
        retained_genes.extend(genes)
    retained_genes = list(dict.fromkeys(retained_genes))
    matrix, var = extract_h5ad_csr_columns(
        source_path,
        "layers/UMIs",
        retained_genes,
        rows_per_block=2000,
    )
    targeted_total = np.asarray(matrix.sum(axis=1)).reshape(-1)
    keep_targeted = targeted_total > 0
    targeted_zero_rows = int((~keep_targeted).sum())
    matrix = matrix[keep_targeted].tocsr()
    with h5py.File(source_path, "r") as handle:
        obs_group = handle["obs"]
        source_columns = {
            "donor_id": "Donor ID",
            "analysis_unit": "load_name",
            "pathology": "Overall AD neuropathological Change",
            "source_class": "Class",
            "source_subclass": "Subclass",
            "subpopulation": "Supertype",
            "brain_region": "Brain Region",
            "cognitive_status": "Cognitive Status",
            "apoe_genotype": "APOE Genotype",
            "sex": "Sex",
            "age_at_death": "Age at Death",
            "source_total_umis": "Number of UMIs",
        }
        obs_data = {
            target: read_h5ad_dataframe_column(obs_group, source)[keep_targeted]
            for target, source in source_columns.items()
        }
        index_value = obs_group.attrs.get("_index", "_index")
        index_name = index_value.decode("utf-8") if isinstance(index_value, bytes) else str(index_value)
        source_index = read_h5ad_dataframe_column(obs_group, index_name)[keep_targeted]
    obs = pd.DataFrame(obs_data, index=pd.Index(source_index.astype(str), name="cell_id"))
    obs["cell_type"] = map_seaad_cell_types(obs["source_class"].to_numpy(), obs["source_subclass"].to_numpy())
    obs["cohort"] = cohort
    obs["donor_inference_eligible"] = True
    obs = _safe_h5ad_obs(obs)

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.uns["matrix_semantics"] = {
        "X": "author raw UMI counts from source layers/UMIs, restricted to preregistered genes",
        "normalization": "none in stored X; analysis copies apply normalization/log1p",
        "feature_scope": "targeted candidate, program, and lineage-identity panel; not transcriptome-wide",
        "cell_type_annotation": "author Class/Subclass collapsed to broad populations; Immune maps to Microglia and retains source subclass/supertype",
        "pathology": "author Overall AD neuropathological Change with High, Intermediate, Low, and Not AD categories",
        "targeted_zero_row_filter": f"{targeted_zero_rows} source nuclei with zero counts across the retained targeted panel were removed",
    }
    adata.uns["cohort_provenance_json"] = json.dumps(
        _provenance_payload(registry, cohort, source_dir), sort_keys=True
    )
    adata.write_h5ad(output_path, compression="gzip")
    _write_preparation_summary(adata, output_path, spec)
    return output_path


def prepare_gse138852(registry: dict[str, Any], cohort: str, overwrite: bool = False) -> Path:
    """Convert GSE138852 count/covariate CSV files to a canonical sparse AnnData object."""
    spec = _cohort_spec(registry, cohort)
    root = Path(registry["_project_root"])
    source_dir = _resolve(root, spec["source_directory"])
    output_path = _resolve(root, spec["prepared_path"])
    if output_path.exists() and not overwrite:
        return output_path
    counts_path = source_dir / "GSE138852_counts.csv.gz"
    covariates_path = source_dir / "GSE138852_covariates.csv.gz"
    for path in (counts_path, covariates_path):
        if not path.exists():
            raise FileNotFoundError(f"Required source file was not found: {path}")

    feature_names: list[str] = []
    blocks: list[sparse.csr_matrix] = []
    cell_names: list[str] | None = None
    for chunk in pd.read_csv(counts_path, index_col=0, chunksize=500):
        if cell_names is None:
            cell_names = chunk.columns.astype(str).tolist()
        elif cell_names != chunk.columns.astype(str).tolist():
            raise ValueError("GSE138852 count-matrix columns changed between chunks.")
        feature_names.extend(chunk.index.astype(str).tolist())
        blocks.append(sparse.csr_matrix(chunk.to_numpy(dtype=np.float32, copy=False)))
    if not blocks or cell_names is None:
        raise ValueError("GSE138852 count matrix was empty.")
    matrix = sparse.vstack(blocks, format="csr").transpose().tocsr()

    covariates = pd.read_csv(covariates_path, index_col=0)
    covariates.index = covariates.index.astype(str)
    missing = pd.Index(cell_names).difference(covariates.index)
    if not missing.empty:
        raise ValueError(f"GSE138852 covariates are missing {missing.size} count-matrix cells.")
    obs = covariates.loc[cell_names].copy()
    raw_cell_type = obs["oupSample.cellType"].astype(str)
    pool = pd.Index(cell_names).to_series(index=obs.index).str.split("_", n=1).str[1]
    obs["donor_id"] = "unavailable"
    obs["analysis_unit"] = pool.fillna("unknown_pool").to_numpy()
    obs["pathology"] = obs["oupSample.batchCond"].astype(str).to_numpy()
    obs["cell_type"] = raw_cell_type.str.lower().map(GSE138852_CELL_TYPE_MAP).fillna(raw_cell_type)
    obs["subpopulation"] = obs["oupSample.subclustID"].astype(str)
    obs["brain_region"] = "entorhinal cortex"
    obs["cohort"] = cohort
    obs["donor_inference_eligible"] = False
    obs["donor_ineligibility_reason"] = spec["donor_ineligibility_reason"]
    for column in CANONICAL_OBS_COLUMNS:
        obs[column] = obs[column].astype(str)

    var = pd.DataFrame(index=pd.Index(feature_names, name="gene_symbol"))
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.var_names_make_unique()
    adata.uns["cohort_provenance_json"] = json.dumps(
        _provenance_payload(registry, cohort, source_dir), sort_keys=True
    )
    adata.uns["matrix_semantics"] = {
        "X": "author-provided integer counts",
        "normalization": "none during preparation",
        "donor_inference": "not eligible because individual donors were pooled before library generation",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path, compression="gzip")
    _write_preparation_summary(adata, output_path, spec)
    return output_path


def prepare_gse157827(registry: dict[str, Any], cohort: str, overwrite: bool = False) -> Path:
    """Prepare donor-separated GSE157827 matrices with transparent marker annotation."""
    spec = _cohort_spec(registry, cohort)
    root = Path(registry["_project_root"])
    source_dir = _resolve(root, spec["source_directory"])
    output_path = _resolve(root, spec["prepared_path"])
    if output_path.exists() and not overwrite:
        return output_path
    archive_path = source_dir / "GSE157827_RAW.tar"
    if not archive_path.exists():
        raise FileNotFoundError(f"Required source file was not found: {archive_path}")
    adata = read_geo_10x_triplet_tar(archive_path, cohort=cohort, brain_region="prefrontal cortex")
    adata, qc_summary = apply_basic_nucleus_qc(adata)
    annotation_summary = annotate_major_brain_cell_types(adata)
    for column in CANONICAL_OBS_COLUMNS:
        adata.obs[column] = adata.obs[column].astype(str)
    adata.uns["cohort_provenance_json"] = json.dumps(
        _provenance_payload(registry, cohort, source_dir), sort_keys=True
    )
    adata.uns["matrix_semantics"] = {
        "X": "author-provided integer counts after conservative nucleus QC",
        "normalization": "none in stored X; log-normalization is applied only to analysis copies",
        "cell_type_annotation": "prespecified marker-module maximum score with explicit confidence fields",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qc_summary.to_csv(output_path.with_suffix(".qc.csv"), index=False)
    annotation_summary.to_csv(output_path.with_suffix(".annotation.csv"), index=False)
    adata.write_h5ad(output_path, compression="gzip")
    _write_preparation_summary(adata, output_path, spec)
    return output_path


def prepare_gse237718(registry: dict[str, Any], cohort: str, overwrite: bool = False) -> Path:
    """Prepare donor-separated temporal-cortex data with GEO APOE/pathology metadata."""
    spec = _cohort_spec(registry, cohort)
    root = Path(registry["_project_root"])
    source_dir = _resolve(root, spec["source_directory"])
    output_path = _resolve(root, spec["prepared_path"])
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / "GSE237718_RAW.tar"
    miniml_path = source_dir / "GSE237718_family.xml.tgz"
    for path in (archive_path, miniml_path):
        if not path.exists():
            raise FileNotFoundError(f"Required source file was not found: {path}")
    metadata = parse_geo_miniml_metadata(miniml_path)
    metadata["donor_id"] = metadata["sample_title"].str.replace(r"^GEX_", "", regex=True)
    metadata["pathology"] = metadata["geo_pathological_diagnosis"]
    metadata["apoe_genotype"] = metadata["geo_genotype"]
    metadata["sex"] = metadata["geo_sex"]
    metadata.to_csv(output_path.with_suffix(".sample_metadata.csv"), index=False)
    adata = read_nested_geo_10x_tar(
        archive_path,
        sample_metadata=metadata,
        cohort=cohort,
        brain_region="temporal cortex",
    )
    adata, qc_summary = apply_basic_nucleus_qc(adata)
    annotation_summary = annotate_major_brain_cell_types(adata)
    for column in CANONICAL_OBS_COLUMNS:
        adata.obs[column] = adata.obs[column].astype(str)
    adata.uns["cohort_provenance_json"] = json.dumps(
        _provenance_payload(registry, cohort, source_dir), sort_keys=True
    )
    adata.uns["matrix_semantics"] = {
        "X": "author-provided integer counts after conservative nucleus QC",
        "normalization": "none in stored X; log-normalization is applied only to analysis copies",
        "cell_type_annotation": "prespecified marker-module maximum score with explicit confidence fields",
        "donor_metadata": "GEO MINiML pathological diagnosis, APOE genotype, and sex",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qc_summary.to_csv(output_path.with_suffix(".qc.csv"), index=False)
    annotation_summary.to_csv(output_path.with_suffix(".annotation.csv"), index=False)
    adata.write_h5ad(output_path, compression="gzip")
    _write_preparation_summary(adata, output_path, spec)
    return output_path


def prepare_gse268599(registry: dict[str, Any], cohort: str, overwrite: bool = False) -> Path:
    """Stream the multiregion archive into a targeted, astrocyte-assigned canonical object."""
    spec = _cohort_spec(registry, cohort)
    root = Path(registry["_project_root"])
    source_dir = _resolve(root, spec["source_directory"])
    output_path = _resolve(root, spec["prepared_path"])
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / "GSE268599_RAW.tar"
    miniml_path = source_dir / "GSE268599_family.xml.tgz"
    supplement_path = source_dir / "GSE268599_supplementary_data1.xlsx"
    for path in (archive_path, miniml_path, supplement_path):
        if not path.exists():
            raise FileNotFoundError(f"Required source file was not found: {path}")

    geo = parse_geo_miniml_metadata(miniml_path)
    geo["source_sample"] = geo["geo_supplementary_files"].str.extract(r"(6289-MW-\d+)")
    if geo["source_sample"].isna().any():
        raise ValueError("Could not derive all GSE268599 source-sample IDs from GEO metadata.")
    qc = pd.read_excel(supplement_path, sheet_name="Quality Control Data")
    donors = pd.read_excel(supplement_path, sheet_name="Supplementary Table 1")
    donor_columns = ["Donor.ID", "Braak", "CERAD", "Pathology Stage", "APOE", "Sex", "Age"]
    metadata = qc.merge(donors[donor_columns], on="Donor.ID", how="left", validate="many_to_one")
    metadata = metadata.rename(
        columns={
            "SampleName": "source_sample",
            "Donor.ID": "donor_id",
            "region": "brain_region",
            "Pathology Stage": "pathology",
            "Braak": "braak_stage",
            "CERAD": "cerad_stage",
            "APOE": "apoe_genotype",
            "Sex": "sex",
            "Age": "age",
            "Number_of_astrocytes": "author_reported_astrocytes",
        }
    )
    metadata = metadata.merge(
        geo[["source_sample", "geo_accession", "sample_title", "geo_isolate", "geo_tissue"]],
        on="source_sample",
        how="left",
        validate="one_to_one",
    )
    if metadata["geo_accession"].isna().any():
        raise ValueError("Peer-reviewed GSE268599 metadata did not fully match GEO sample records.")
    metadata["analysis_unit"] = metadata["source_sample"]
    metadata.columns = _safe_unique_column_names(metadata.columns)
    metadata.to_csv(output_path.with_suffix(".sample_metadata.csv"), index=False)
    retained_genes = list(registry["candidate_genes"])
    for genes in registry["programs"].values():
        retained_genes.extend(genes)
    for genes in MAJOR_BRAIN_CELL_MARKERS.values():
        retained_genes.extend(genes)
    adata, sample_summary = build_targeted_cell_type_object_from_geo_tar(
        archive_path,
        sample_metadata=metadata,
        retained_genes=retained_genes,
        target_cell_type="Astrocytes",
        cohort=cohort,
        target_count_column="author_reported_astrocytes",
    )
    targeted_totals = np.asarray(adata.X.sum(axis=1)).reshape(-1)
    nonzero_targeted = targeted_totals > 0
    zero_targeted_rows = int((~nonzero_targeted).sum())
    if zero_targeted_rows:
        adata = adata[nonzero_targeted].copy()
    sample_summary["targeted_panel_zero_rows_removed_total"] = zero_targeted_rows
    for column in CANONICAL_OBS_COLUMNS:
        adata.obs[column] = adata.obs[column].astype(str)
    adata.uns["cohort_provenance_json"] = json.dumps(
        _provenance_payload(registry, cohort, source_dir), sort_keys=True
    )
    adata.uns["matrix_semantics"] = {
        "X": "author-provided counts restricted after full-matrix QC and two-tier astrocyte selection",
        "feature_scope": "preregistered candidate, program, and cell-identity genes; not transcriptome-wide",
        "normalization": "none in stored X; log-normalization is applied only to analysis copies",
        "targeted_zero_row_filter": f"{zero_targeted_rows} cells with zero counts across the retained targeted panel were removed",
        "pathology_metadata": "peer-reviewed Supplementary Data 1, DOI 10.1038/s41593-024-01791-4",
        "cell_type_selection": "strict marker assignments retained first; remaining author-reported per-sample astrocyte quota filled by target-versus-other-lineage marker-score margin; tier stored per nucleus",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_summary.to_csv(output_path.with_suffix(".sample_retention.csv"), index=False)
    adata.write_h5ad(output_path, compression="gzip")
    _write_preparation_summary(adata, output_path, spec)
    return output_path


def prepare_gse269906(registry: dict[str, Any], cohort: str, overwrite: bool = False) -> Path:
    """Prepare a targeted spatial object from donor-separated Stereo-seq GEF files."""
    spec = _cohort_spec(registry, cohort)
    root = Path(registry["_project_root"])
    source_dir = _resolve(root, spec["source_directory"])
    output_path = _resolve(root, spec["prepared_path"])
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / "GSE269906_RAW.tar"
    miniml_path = source_dir / "GSE269906_family.xml.tgz"
    for path in (archive_path, miniml_path):
        if not path.exists():
            raise FileNotFoundError(f"Required spatial source file was not found: {path}")
    geo = parse_geo_miniml_metadata(miniml_path)
    geo["pathology"] = np.where(
        geo["sample_title"].str.contains(r"\bAD\b", case=False, regex=True), "AD", "Control"
    )
    geo.to_csv(output_path.with_suffix(".sample_metadata.csv"), index=False)
    retained_genes = list(registry["candidate_genes"])
    for genes in registry["programs"].values():
        retained_genes.extend(genes)
    for genes in MAJOR_BRAIN_CELL_MARKERS.values():
        retained_genes.extend(genes)
    gef_paths = extract_cellbin_gef_files(archive_path, source_dir / "extracted_cellbin")
    adata, sample_summary = prepare_spatial_cohort(
        gef_paths=gef_paths,
        sample_metadata=geo,
        retained_genes=retained_genes,
        cohort=cohort,
    )
    adata.uns["cohort_provenance_json"] = json.dumps(
        _provenance_payload(registry, cohort, source_dir), sort_keys=True
    )
    sample_summary.to_csv(output_path.with_suffix(".sample_retention.csv"), index=False)
    adata.write_h5ad(output_path, compression="gzip")
    _write_preparation_summary(adata, output_path, spec)
    return output_path


def canonical_metadata_summary(path: str | Path) -> pd.DataFrame:
    """Summarize canonical metadata and RHBDF2 support without loading all expression."""
    adata = ad.read_h5ad(Path(path), backed="r")
    try:
        rows = []
        for column in CANONICAL_OBS_COLUMNS:
            present = column in adata.obs.columns
            rows.append(
                {
                    "column": column,
                    "present": present,
                    "non_missing": int(adata.obs[column].notna().sum()) if present else 0,
                    "unique_values": int(adata.obs[column].nunique(dropna=True)) if present else 0,
                }
            )
        rows.append(
            {
                "column": "RHBDF2_gene",
                "present": "RHBDF2" in adata.var_names,
                "non_missing": int(adata.n_obs),
                "unique_values": np.nan,
            }
        )
        return pd.DataFrame(rows)
    finally:
        adata.file.close()


def parse_geo_miniml_metadata(path: str | Path) -> pd.DataFrame:
    """Parse sample titles and tagged characteristics from a GEO MINiML archive."""
    path = Path(path)
    namespace = {"m": "http://www.ncbi.nlm.nih.gov/geo/info/MINiML"}
    with tarfile.open(path, mode="r:gz") as archive:
        xml_members = [member for member in archive.getmembers() if member.name.endswith(".xml")]
        if len(xml_members) != 1:
            raise ValueError(f"Expected one MINiML XML member in {path}, found {len(xml_members)}.")
        handle = archive.extractfile(xml_members[0])
        if handle is None:
            raise FileNotFoundError(f"Could not read {xml_members[0].name} from {path}.")
        root = ET.parse(handle).getroot()
    rows: list[dict[str, str]] = []
    for sample in root.findall("m:Sample", namespace):
        row = {
            "geo_accession": str(sample.get("iid", "")).strip(),
            "sample_title": str(sample.findtext("m:Title", default="", namespaces=namespace)).strip(),
        }
        channel = sample.find("m:Channel", namespace)
        if channel is not None:
            for characteristic in channel.findall("m:Characteristics", namespace):
                tag = str(characteristic.get("tag", "characteristic")).strip().lower()
                key = "geo_" + "_".join(tag.replace("/", " ").split())
                row[key] = str(characteristic.text or "").strip()
        supplementary = sample.findall("m:Supplementary-Data", namespace)
        row["geo_supplementary_files"] = ";".join(
            str(item.text or "").strip() for item in supplementary if str(item.text or "").strip()
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty or result["geo_accession"].duplicated().any():
        raise ValueError(f"MINiML metadata in {path} was empty or had duplicate accessions.")
    return result


def build_canonical_cell_type_eligibility(
    path: str | Path,
    output_path: str | Path,
    minimum_cells: int,
    minimum_detected_cells: int,
    minimum_donors: int,
    minimum_donors_with_detection: int,
) -> pd.DataFrame:
    """Build cell-type eligibility without pretending unavailable donors are replicates."""
    adata = ad.read_h5ad(Path(path), backed="r")
    try:
        required = {"cell_type", "donor_id", "pathology"}
        missing = required.difference(adata.obs.columns)
        if missing:
            raise ValueError(f"Canonical H5AD is missing obs columns: {', '.join(sorted(missing))}")
        donor_eligible = (
            "donor_inference_eligible" in adata.obs.columns
            and bool(adata.obs["donor_inference_eligible"].astype(bool).all())
        )
        rhbdf2 = extract_expression_vector(adata, "RHBDF2")
        obs = adata.obs[["cell_type", "donor_id", "pathology"]].copy()
    finally:
        adata.file.close()
    obs["_rhbdf2"] = rhbdf2
    rows: list[dict[str, object]] = []
    for cell_type, frame in obs.groupby("cell_type", observed=True):
        detected = frame["_rhbdf2"] > 0
        n_cells = int(frame.shape[0])
        n_detected = int(detected.sum())
        n_donors = int(frame["donor_id"].nunique()) if donor_eligible else 0
        donors_detected = int(frame.loc[detected, "donor_id"].nunique()) if donor_eligible else 0
        cell_eligible = n_cells >= minimum_cells and n_detected >= minimum_detected_cells
        donor_model_eligible = (
            cell_eligible
            and donor_eligible
            and n_donors >= minimum_donors
            and donors_detected >= minimum_donors_with_detection
        )
        reasons = []
        if n_cells < minimum_cells:
            reasons.append(f"fewer than {minimum_cells} cells")
        if n_detected < minimum_detected_cells:
            reasons.append(f"fewer than {minimum_detected_cells} RHBDF2-detected cells")
        if not donor_eligible:
            reasons.append("cell-to-donor assignment unavailable")
        elif n_donors < minimum_donors:
            reasons.append(f"fewer than {minimum_donors} donors")
        elif donors_detected < minimum_donors_with_detection:
            reasons.append(f"fewer than {minimum_donors_with_detection} donors with RHBDF2 detection")
        rows.append(
            {
                "cell_type": str(cell_type),
                "n_cells": n_cells,
                "n_donors": n_donors,
                "rhbdf2_detected_cells": n_detected,
                "rhbdf2_detection_fraction": n_detected / n_cells if n_cells else np.nan,
                "donors_with_rhbdf2_detection": donors_detected,
                "pathology_levels": int(frame["pathology"].nunique()),
                "cell_level_continuous_eligible": cell_eligible,
                "donor_continuous_pseudobulk_eligible": donor_model_eligible,
                "eligibility_note": "; ".join(reasons),
            }
        )
    result = pd.DataFrame(rows).sort_values("n_cells", ascending=False).reset_index(drop=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute a streaming SHA-256 checksum."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_with_resume(
    url: str,
    path: Path,
    expected_size: int | None,
    chunk_size: int,
    timeout_seconds: int,
) -> None:
    existing = path.stat().st_size if path.exists() else 0
    if expected_size and existing == int(expected_size) and not _aria2_sidecar(path).exists():
        return
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    mode = "ab" if existing else "wb"
    with requests.get(url, stream=True, headers=headers, timeout=timeout_seconds) as response:
        response.raise_for_status()
        if existing and response.status_code != 206:
            existing = 0
            mode = "wb"
        with path.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    if expected_size and path.stat().st_size != int(expected_size):
        raise IOError(
            f"Downloaded size mismatch for {path.name}: {path.stat().st_size} != {expected_size}."
        )


def _write_download_manifest(path: Path, records: list[DownloadRecord]) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [
            {
                "cohort": record.cohort,
                "url": record.url,
                "path": str(record.path),
                "size_bytes": record.size_bytes,
                "sha256": record.sha256,
                "status": record.status,
            }
            for record in records
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_preparation_summary(adata: ad.AnnData, output_path: Path, spec: dict[str, Any]) -> None:
    summary = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "cell_types": {str(k): int(v) for k, v in adata.obs["cell_type"].value_counts().items()},
        "pathology": {str(k): int(v) for k, v in adata.obs["pathology"].value_counts().items()},
        "analysis_units": int(adata.obs["analysis_unit"].nunique()),
        "donor_inference_eligible": bool(spec.get("donor_inference_eligible", False)),
    }
    output_path.with_suffix(".preparation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def _provenance_payload(registry: dict[str, Any], cohort: str, source_dir: Path) -> dict[str, Any]:
    spec = _cohort_spec(registry, cohort)
    files = []
    for source in spec.get("sources", []):
        path = source_dir / source["filename"]
        files.append(
            {
                "url": source["url"],
                "filename": source["filename"],
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256_file(path) if path.exists() else None,
            }
        )
    return {
        "cohort": cohort,
        "title": spec["title"],
        "role": spec["role"],
        "registry_version": registry.get("registry_version"),
        "registry_path": registry["_registry_path"],
        "source_files": files,
    }


def _cohort_spec(registry: dict[str, Any], cohort: str) -> dict[str, Any]:
    try:
        return registry["cohorts"][cohort]
    except KeyError as exc:
        available = ", ".join(sorted(registry["cohorts"]))
        raise KeyError(f"Unknown cohort '{cohort}'. Available cohorts: {available}") from exc


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _aria2_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".aria2")


def _safe_unique_column_names(columns: Any) -> list[str]:
    """Create deterministic HDF5-safe observation column names."""
    used: dict[str, int] = {}
    result = []
    for value in columns:
        base = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip()).strip("_").lower()
        base = base or "unnamed"
        count = used.get(base, 0)
        used[base] = count + 1
        result.append(base if count == 0 else f"{base}_{count + 1}")
    return result


def _safe_h5ad_obs(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize observation names and string columns for portable HDF5 writes."""
    result = frame.copy()
    result.columns = _safe_unique_column_names(result.columns)
    for column in result.columns:
        if pd.api.types.is_object_dtype(result[column]) or isinstance(result[column].dtype, pd.StringDtype):
            result[column] = pd.Categorical(result[column].fillna("missing").astype(str))
    return result


def _is_complete_file(path: Path, expected_size: int | None = None) -> bool:
    if not path.exists() or _aria2_sidecar(path).exists():
        return False
    if expected_size is not None:
        return path.stat().st_size == int(expected_size)
    return path.stat().st_size > 0
