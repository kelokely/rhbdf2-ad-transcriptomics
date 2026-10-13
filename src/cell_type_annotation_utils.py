"""Transparent major-cell-type annotation for public brain 10x count archives."""

from __future__ import annotations

from pathlib import Path
import gzip
from io import BytesIO
import re
import tarfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


MAJOR_BRAIN_CELL_MARKERS = {
    "Microglia": ["P2RY12", "CX3CR1", "CSF1R", "TYROBP", "FCER1G", "AIF1", "LST1", "TREM2"],
    "Astrocytes": ["AQP4", "ALDH1L1", "SLC1A3", "SLC1A2", "GFAP", "GJA1", "GLUL"],
    "Oligodendrocytes": ["MBP", "PLP1", "MOG", "MOBP", "MAG", "SOX10"],
    "OPCs": ["PDGFRA", "CSPG4", "VCAN", "OLIG1", "OLIG2"],
    "Excitatory neurons": ["SLC17A7", "SATB2", "CAMK2A", "CUX1", "CUX2", "RORB"],
    "Inhibitory neurons": ["GAD1", "GAD2", "SLC6A1", "DLX1", "DLX2"],
    "Endothelial cells": ["CLDN5", "PECAM1", "VWF", "KDR", "FLT1", "EMCN"],
    "Pericytes": ["RGS5", "PDGFRB", "CSPG4", "COL4A1", "COL4A2", "MCAM"],
}


def read_geo_10x_triplet_tar(
    path: str | Path,
    cohort: str,
    brain_region: str,
) -> ad.AnnData:
    """Read donor-separated, gzipped 10x triplets directly from a GEO tar archive."""
    path = Path(path)
    matrices: list[sparse.csr_matrix] = []
    obs_frames: list[pd.DataFrame] = []
    reference_features: pd.DataFrame | None = None
    with tarfile.open(path, mode="r") as archive:
        names = archive.getnames()
        sample_prefixes = sorted(name[: -len("_matrix.mtx.gz")] for name in names if name.endswith("_matrix.mtx.gz"))
        if not sample_prefixes:
            raise ValueError(f"No gzipped 10x matrix triplets were found in {path}.")
        for prefix in sample_prefixes:
            matrix = _read_mtx_member(archive, f"{prefix}_matrix.mtx.gz").transpose().tocsr()
            features = _read_tsv_member(archive, f"{prefix}_features.tsv.gz")
            barcodes = _read_tsv_member(archive, f"{prefix}_barcodes.tsv.gz").iloc[:, 0].astype(str)
            if matrix.shape != (barcodes.shape[0], features.shape[0]):
                raise ValueError(
                    f"10x shape mismatch for {prefix}: {matrix.shape}, "
                    f"barcodes={barcodes.shape[0]}, features={features.shape[0]}."
                )
            if reference_features is None:
                reference_features = features.copy()
            elif not features.iloc[:, :2].equals(reference_features.iloc[:, :2]):
                raise ValueError(f"Feature order differs between 10x libraries at {prefix}.")
            sample = _sample_label(prefix)
            pathology = "AD" if sample.upper().startswith("AD") else "Control"
            obs = pd.DataFrame(index=pd.Index(prefix + ":" + barcodes, name="cell_id"))
            obs["donor_id"] = sample
            obs["analysis_unit"] = sample
            obs["pathology"] = pathology
            obs["subpopulation"] = "not_assigned"
            obs["brain_region"] = brain_region
            obs["cohort"] = cohort
            obs["donor_inference_eligible"] = True
            matrices.append(matrix)
            obs_frames.append(obs)
    assert reference_features is not None
    feature_symbols = reference_features.iloc[:, 1].astype(str)
    var = pd.DataFrame(index=pd.Index(feature_symbols, name="gene_symbol"))
    var["gene_id"] = reference_features.iloc[:, 0].astype(str).to_numpy()
    if reference_features.shape[1] >= 3:
        var["feature_type"] = reference_features.iloc[:, 2].astype(str).to_numpy()
    result = ad.AnnData(
        X=sparse.vstack(matrices, format="csr"),
        obs=pd.concat(obs_frames, axis=0),
        var=var,
    )
    result.var_names_make_unique()
    return result


def read_nested_geo_10x_tar(
    path: str | Path,
    sample_metadata: pd.DataFrame,
    cohort: str,
    brain_region: str,
) -> ad.AnnData:
    """Read one nested 10x archive per donor and attach frozen GEO metadata."""
    required = {"sample_title", "donor_id", "pathology"}
    missing = required.difference(sample_metadata.columns)
    if missing:
        raise ValueError(f"Sample metadata is missing: {', '.join(sorted(missing))}")
    metadata = sample_metadata.set_index("sample_title", verify_integrity=True)
    matrices: list[sparse.csr_matrix] = []
    obs_frames: list[pd.DataFrame] = []
    reference_features: pd.DataFrame | None = None
    with tarfile.open(Path(path), mode="r") as outer:
        members = sorted(
            (member for member in outer.getmembers() if member.isfile() and member.name.endswith(".tar.gz")),
            key=lambda member: member.name,
        )
        if not members:
            raise ValueError(f"No nested 10x archives were found in {path}.")
        for member in members:
            sample_title = re.sub(r"^GSM\d+_", "", Path(member.name).name[: -len(".tar.gz")])
            if sample_title not in metadata.index:
                raise KeyError(f"No GEO metadata row matched nested sample '{sample_title}'.")
            extracted = outer.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(f"Could not read nested archive member {member.name}.")
            with tarfile.open(fileobj=BytesIO(extracted.read()), mode="r:gz") as inner:
                matrix = _read_mtx_member(inner, _member_ending(inner, "matrix.mtx.gz")).transpose().tocsr()
                features = _read_tsv_member(inner, _member_ending(inner, "features.tsv.gz"))
                barcodes = _read_tsv_member(inner, _member_ending(inner, "barcodes.tsv.gz")).iloc[:, 0].astype(str)
            if matrix.shape != (barcodes.shape[0], features.shape[0]):
                raise ValueError(f"10x shape mismatch for nested sample {sample_title}: {matrix.shape}.")
            if reference_features is None:
                reference_features = features.copy()
            elif not features.iloc[:, :2].equals(reference_features.iloc[:, :2]):
                raise ValueError(f"Feature order differs at nested sample {sample_title}.")
            sample_row = metadata.loc[sample_title]
            obs = pd.DataFrame(index=pd.Index(sample_title + ":" + barcodes, name="cell_id"))
            obs["donor_id"] = str(sample_row["donor_id"])
            obs["analysis_unit"] = str(sample_row["donor_id"])
            obs["pathology"] = str(sample_row["pathology"])
            obs["subpopulation"] = "not_assigned"
            obs["brain_region"] = brain_region
            obs["cohort"] = cohort
            obs["donor_inference_eligible"] = True
            for column in sample_metadata.columns:
                if column not in obs.columns and column != "sample_title":
                    obs[column] = str(sample_row[column])
            matrices.append(matrix)
            obs_frames.append(obs)
    assert reference_features is not None
    feature_symbols = ad.utils.make_index_unique(pd.Index(reference_features.iloc[:, 1].astype(str), name="gene_symbol"))
    var = pd.DataFrame(index=feature_symbols)
    var["gene_id"] = reference_features.iloc[:, 0].astype(str).to_numpy()
    if reference_features.shape[1] >= 3:
        var["feature_type"] = reference_features.iloc[:, 2].astype(str).to_numpy()
    return ad.AnnData(X=sparse.vstack(matrices, format="csr"), obs=pd.concat(obs_frames), var=var)


def build_targeted_cell_type_object_from_geo_tar(
    path: str | Path,
    sample_metadata: pd.DataFrame,
    retained_genes: list[str],
    target_cell_type: str,
    cohort: str,
    target_count_column: str | None = None,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Stream a large GEO tar and retain a transparent targeted cell-type panel.

    If frozen author-reported cell counts are available but barcode labels are not,
    strict marker assignments are retained first and the remaining quota is filled
    by the target-versus-other-lineage score margin. Every filled cell is flagged.
    """
    required = {"source_sample", "donor_id", "pathology", "brain_region"}
    missing = required.difference(sample_metadata.columns)
    if missing:
        raise ValueError(f"Sample metadata is missing: {', '.join(sorted(missing))}")
    metadata = sample_metadata.set_index("source_sample", verify_integrity=True)
    requested = list(dict.fromkeys(retained_genes))
    matrices: list[sparse.csr_matrix] = []
    obs_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    retained_var: pd.DataFrame | None = None
    with tarfile.open(Path(path), mode="r") as archive:
        names = archive.getnames()
        prefixes = sorted(name[: -len("_matrix.mtx.gz")] for name in names if name.endswith("_matrix.mtx.gz"))
        if not prefixes:
            raise ValueError(f"No gzipped 10x matrix triplets were found in {path}.")
        for prefix in prefixes:
            source_sample = _sample_label(prefix).split("_", 1)[-1]
            if source_sample not in metadata.index:
                raise KeyError(f"No frozen sample metadata matched '{source_sample}'.")
            matrix = _read_mtx_member(archive, f"{prefix}_matrix.mtx.gz").transpose().tocsr()
            features = _read_tsv_member(archive, f"{prefix}_features.tsv.gz")
            barcodes = _read_tsv_member(archive, f"{prefix}_barcodes.tsv.gz").iloc[:, 0].astype(str)
            symbols = ad.utils.make_index_unique(pd.Index(features.iloc[:, 1].astype(str), name="gene_symbol"))
            var = pd.DataFrame(index=symbols)
            var["gene_id"] = features.iloc[:, 0].astype(str).to_numpy()
            sample_obs = pd.DataFrame(index=pd.Index(prefix + ":" + barcodes, name="cell_id"))
            sample = ad.AnnData(X=matrix, obs=sample_obs, var=var)
            sample, qc = apply_basic_nucleus_qc(sample)
            annotate_major_brain_cell_types(sample)
            strict_mask = sample.obs["cell_type"].astype(str).eq(target_cell_type).to_numpy()
            cell_mask = strict_mask.copy()
            selection_method = "strict_marker_assignment"
            requested_target_count = np.nan
            sample_row = metadata.loc[source_sample]
            if target_count_column and target_count_column in sample_row.index:
                requested_target_count = pd.to_numeric(sample_row[target_count_column], errors="coerce")
                if pd.notna(requested_target_count):
                    target_count = min(int(requested_target_count), sample.n_obs)
                    cell_mask, specificity = calibrated_target_selection(
                        sample.obs, target_cell_type, target_count
                    )
                    sample.obs["target_cell_type_specificity"] = specificity
                    selection_method = "strict_then_author_count_calibrated_marker_rank"
            present = [gene for gene in requested if gene in sample.var_names]
            if retained_var is None:
                retained_var = sample.var.loc[present].copy()
            elif present != retained_var.index.tolist():
                raise ValueError(f"Retained gene order differs at sample {source_sample}.")
            selected_obs = sample.obs.loc[cell_mask].copy()
            selected_obs["donor_id"] = str(sample_row["donor_id"])
            selected_obs["analysis_unit"] = str(sample_row.get("analysis_unit", source_sample))
            selected_obs["pathology"] = str(sample_row["pathology"])
            selected_obs["cell_type"] = target_cell_type
            selected_obs["astrocyte_selection_tier"] = np.where(
                strict_mask[cell_mask], "strict_marker", "author_count_calibrated_fill"
            )
            selected_obs["subpopulation"] = selected_obs["astrocyte_selection_tier"]
            selected_obs["cell_type_selection_method"] = selection_method
            selected_obs["brain_region"] = str(sample_row["brain_region"])
            selected_obs["cohort"] = cohort
            selected_obs["donor_inference_eligible"] = True
            for column in sample_metadata.columns:
                if column not in selected_obs.columns and column != "source_sample":
                    selected_obs[column] = str(sample_row[column])
            matrices.append(sample[cell_mask, present].X.copy().tocsr())
            obs_frames.append(selected_obs)
            summaries.append(
                {
                    "source_sample": source_sample,
                    "donor_id": sample_row["donor_id"],
                    "brain_region": sample_row["brain_region"],
                    "cells_before_qc": int(qc.loc[0, "cells_before"]),
                    "cells_after_qc": int(qc.loc[0, "cells_after"]),
                    "target_cells_retained": int(cell_mask.sum()),
                    "strict_marker_target_cells": int(strict_mask.sum()),
                    "author_reported_astrocytes": sample_row.get("author_reported_astrocytes"),
                    "requested_author_target_count": requested_target_count,
                    "selection_method": selection_method,
                }
            )
            del sample, matrix
    if retained_var is None or not matrices:
        raise ValueError(f"No {target_cell_type} cells were retained from {path}.")
    result = ad.AnnData(
        X=sparse.vstack(matrices, format="csr"),
        obs=pd.concat(obs_frames, axis=0),
        var=retained_var,
    )
    return result, pd.DataFrame(summaries)


def calibrated_target_selection(
    obs: pd.DataFrame,
    target_cell_type: str,
    target_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain strict calls first, then fill an author-count quota by score margin."""
    strict_mask = obs["cell_type"].astype(str).eq(target_cell_type).to_numpy()
    target_score_column = f"marker_score_{_safe_name(target_cell_type)}"
    other_score_columns = [
        column
        for column in obs.columns
        if column.startswith("marker_score_") and column != target_score_column
    ]
    if target_score_column not in obs.columns or not other_score_columns:
        raise ValueError(f"Marker-score columns are incomplete for target '{target_cell_type}'.")
    target_score = obs[target_score_column].to_numpy(dtype=float)
    other_score = obs[other_score_columns].max(axis=1).to_numpy(dtype=float)
    specificity = target_score - other_score
    target_count = min(max(int(target_count), 0), obs.shape[0])
    strict_indices = np.flatnonzero(strict_mask)
    if strict_indices.size >= target_count:
        keep = strict_indices[np.argsort(specificity[strict_indices])[::-1][:target_count]]
    else:
        remaining = np.flatnonzero(~strict_mask)
        fill = remaining[np.argsort(specificity[remaining])[::-1][: target_count - strict_indices.size]]
        keep = np.concatenate([strict_indices, fill])
    selected = np.zeros(obs.shape[0], dtype=bool)
    selected[keep] = True
    return selected, specificity


def apply_basic_nucleus_qc(
    adata: ad.AnnData,
    minimum_genes: int = 200,
    maximum_genes: int = 10_000,
    maximum_mitochondrial_fraction: float = 0.20,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Apply conservative count-based QC while preserving raw counts."""
    matrix = adata.X
    n_genes = np.asarray((matrix > 0).sum(axis=1)).reshape(-1)
    total_counts = np.asarray(matrix.sum(axis=1)).reshape(-1)
    mitochondrial = adata.var_names.str.upper().str.startswith("MT-")
    if mitochondrial.any():
        mitochondrial_counts = np.asarray(matrix[:, mitochondrial].sum(axis=1)).reshape(-1)
        mitochondrial_fraction = np.divide(
            mitochondrial_counts,
            total_counts,
            out=np.zeros_like(total_counts, dtype=float),
            where=total_counts > 0,
        )
    else:
        mitochondrial_fraction = np.zeros(adata.n_obs, dtype=float)
    keep = (
        (n_genes >= minimum_genes)
        & (n_genes <= maximum_genes)
        & (mitochondrial_fraction <= maximum_mitochondrial_fraction)
        & (total_counts > 0)
    )
    adata.obs["n_genes_by_counts"] = n_genes
    adata.obs["total_counts"] = total_counts
    adata.obs["pct_counts_mt"] = mitochondrial_fraction * 100.0
    summary = pd.DataFrame(
        [
            {
                "cells_before": int(adata.n_obs),
                "cells_after": int(keep.sum()),
                "cells_removed": int((~keep).sum()),
                "minimum_genes": minimum_genes,
                "maximum_genes": maximum_genes,
                "maximum_mitochondrial_fraction": maximum_mitochondrial_fraction,
            }
        ]
    )
    return adata[keep].copy(), summary


def annotate_major_brain_cell_types(
    adata: ad.AnnData,
    marker_sets: dict[str, list[str]] | None = None,
    minimum_detected_markers: int = 2,
    high_confidence_margin: float = 0.15,
) -> pd.DataFrame:
    """Assign major cell types from prespecified marker modules with explicit confidence."""
    marker_sets = marker_sets or MAJOR_BRAIN_CELL_MARKERS
    requested = list(dict.fromkeys(gene for genes in marker_sets.values() for gene in genes))
    present = [gene for gene in requested if gene in adata.var_names]
    if not present:
        raise ValueError("No major-cell-type marker genes were present in the matrix.")
    indices = [adata.var_names.get_loc(gene) for gene in present]
    values = adata.X[:, indices]
    values = values.toarray().astype(np.float32) if sparse.issparse(values) else np.asarray(values, dtype=np.float32)
    library_size = np.asarray(adata.X.sum(axis=1)).reshape(-1).astype(np.float32)
    scale = np.divide(10_000.0, library_size, out=np.zeros_like(library_size), where=library_size > 0)
    normalized = np.log1p(values * scale[:, None])
    means = normalized.mean(axis=0)
    standard_deviations = normalized.std(axis=0)
    standard_deviations[standard_deviations == 0] = 1.0
    standardized = (normalized - means) / standard_deviations
    position = {gene: index for index, gene in enumerate(present)}

    score_columns: dict[str, np.ndarray] = {}
    detection_columns: dict[str, np.ndarray] = {}
    for cell_type, genes in marker_sets.items():
        available = [position[gene] for gene in genes if gene in position]
        if not available:
            score_columns[cell_type] = np.full(adata.n_obs, -np.inf)
            detection_columns[cell_type] = np.zeros(adata.n_obs, dtype=int)
            continue
        score_columns[cell_type] = standardized[:, available].mean(axis=1)
        detection_columns[cell_type] = (values[:, available] > 0).sum(axis=1)

    labels = list(marker_sets)
    score_matrix = np.column_stack([score_columns[label] for label in labels])
    detection_matrix = np.column_stack([detection_columns[label] for label in labels])
    order = np.argsort(score_matrix, axis=1)
    best_index = order[:, -1]
    second_index = order[:, -2]
    row_index = np.arange(adata.n_obs)
    best_score = score_matrix[row_index, best_index]
    second_score = score_matrix[row_index, second_index]
    best_detection = detection_matrix[row_index, best_index]
    margin = best_score - second_score
    assigned = np.asarray(labels, dtype=object)[best_index]
    insufficient = best_detection < minimum_detected_markers
    assigned[insufficient] = "Unassigned"
    confidence = np.where(
        insufficient,
        "unassigned",
        np.where(margin >= high_confidence_margin, "high", "moderate"),
    )
    adata.obs["cell_type"] = assigned.astype(str)
    adata.obs["cell_type_annotation_method"] = "prespecified_marker_module"
    adata.obs["cell_type_annotation_confidence"] = confidence.astype(str)
    adata.obs["cell_type_score_margin"] = margin.astype(float)
    adata.obs["cell_type_detected_markers"] = best_detection.astype(int)
    for label in labels:
        adata.obs[f"marker_score_{_safe_name(label)}"] = score_columns[label]

    rows = []
    for label in labels + ["Unassigned"]:
        mask = adata.obs["cell_type"].astype(str).to_numpy() == label
        rows.append(
            {
                "cell_type": label,
                "n_cells": int(mask.sum()),
                "fraction": float(mask.mean()),
                "high_confidence_cells": int(
                    np.sum(mask & (adata.obs["cell_type_annotation_confidence"].to_numpy() == "high"))
                ),
                "markers_requested": ";".join(marker_sets.get(label, [])),
                "markers_present": ";".join(gene for gene in marker_sets.get(label, []) if gene in present),
            }
        )
    return pd.DataFrame(rows)


def _read_mtx_member(archive: tarfile.TarFile, name: str) -> sparse.coo_matrix:
    extracted = archive.extractfile(name)
    if extracted is None:
        raise FileNotFoundError(f"Tar member was not found: {name}")
    with gzip.GzipFile(fileobj=extracted, mode="rb") as handle:
        return mmread(handle).tocoo()


def _read_tsv_member(archive: tarfile.TarFile, name: str) -> pd.DataFrame:
    extracted = archive.extractfile(name)
    if extracted is None:
        raise FileNotFoundError(f"Tar member was not found: {name}")
    with gzip.GzipFile(fileobj=extracted, mode="rb") as handle:
        return pd.read_csv(handle, sep="\t", header=None)


def _sample_label(prefix: str) -> str:
    match = re.match(r"GSM\d+_(.+)$", prefix)
    return match.group(1) if match else prefix


def _member_ending(archive: tarfile.TarFile, suffix: str) -> str:
    matches = [member.name for member in archive.getmembers() if member.isfile() and member.name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one inner archive member ending in '{suffix}', found {len(matches)}.")
    return matches[0]


def _safe_name(value: str) -> str:
    return "_".join(value.lower().replace("/", " ").split())
