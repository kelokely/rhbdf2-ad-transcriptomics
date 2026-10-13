"""Targeted reader and descriptive analysis helpers for Stereo-seq cell-bin GEF."""

from __future__ import annotations

from pathlib import Path
import shutil
import tarfile
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial import cKDTree

from .cell_type_annotation_utils import annotate_major_brain_cell_types


def extract_cellbin_gef_files(archive_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Extract only cell-bin GEF members with basename-only destinations."""
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with tarfile.open(archive_path, mode="r") as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(".cellbin.gef")]
        if not members:
            raise ValueError(f"No .cellbin.gef members were found in {archive_path}.")
        for member in sorted(members, key=lambda item: item.name):
            destination = output_dir / Path(member.name).name
            if not destination.exists() or destination.stat().st_size != member.size:
                source = archive.extractfile(member)
                if source is None:
                    raise FileNotFoundError(f"Could not read tar member {member.name}.")
                with destination.open("wb") as handle:
                    shutil.copyfileobj(source, handle, length=8 * 1024 * 1024)
            paths.append(destination)
    return paths


def read_cellbin_gef_targeted(path: str | Path, requested_genes: list[str]) -> ad.AnnData:
    """Read selected genes and spatial centroids from an official cell-bin GEF file."""
    path = Path(path)
    genes = list(dict.fromkeys(str(gene) for gene in requested_genes))
    with h5py.File(path, mode="r") as handle:
        cell_group = _find_group(handle, "cellBin")
        cell = cell_group["cell"][:]
        gene = cell_group["gene"][:]
        gene_exp = cell_group["geneExp"]
        gene_names = _gene_names(gene)
        gene_position = {symbol: index for index, symbol in enumerate(gene_names)}
        rows: list[np.ndarray] = []
        columns: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for output_column, symbol in enumerate(genes):
            gene_index = gene_position.get(symbol)
            if gene_index is None:
                continue
            record = gene[gene_index]
            offset = int(_field(record, ("offset",)))
            count = int(_field(record, ("cellCount", "count")))
            entries = gene_exp[offset : offset + count]
            cell_ids = np.asarray(_structured_field(entries, ("cellID", "cellId", "cell_id")), dtype=np.int64)
            expression = np.asarray(_structured_field(entries, ("count", "MIDCount", "expCount")), dtype=np.float32)
            row_indices = _cell_row_indices(cell, cell_ids)
            valid = (row_indices >= 0) & (row_indices < len(cell)) & (expression > 0)
            rows.append(row_indices[valid])
            columns.append(np.full(int(valid.sum()), output_column, dtype=np.int32))
            values.append(expression[valid])
        if rows:
            matrix = sparse.coo_matrix(
                (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                shape=(len(cell), len(genes)),
                dtype=np.float32,
            ).tocsr()
        else:
            matrix = sparse.csr_matrix((len(cell), len(genes)), dtype=np.float32)
        x = np.asarray(_structured_field(cell, ("x", "centerX")), dtype=np.float32)
        y = np.asarray(_structured_field(cell, ("y", "centerY")), dtype=np.float32)
        obs = pd.DataFrame(index=pd.Index([f"cell_{i}" for i in range(len(cell))], name="cell_id"))
        for target, alternatives in {
            "full_gene_count": ("geneCount",),
            "full_exp_count": ("expCount", "MIDCount"),
            "dnb_count": ("dnbCount",),
            "cell_area": ("area",),
        }.items():
            try:
                obs[target] = _structured_field(cell, alternatives)
            except KeyError:
                obs[target] = np.nan
    result = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene_symbol")))
    result.obsm["spatial"] = np.column_stack([x, y])
    result.uns["source_gef"] = str(path)
    return result


def prepare_spatial_cohort(
    gef_paths: list[Path],
    sample_metadata: pd.DataFrame,
    retained_genes: list[str],
    cohort: str,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Prepare and annotate all spatial donors while retaining source coordinates."""
    metadata = sample_metadata.set_index("geo_accession", verify_integrity=True)
    objects = []
    summaries = []
    for path in gef_paths:
        accession = path.name.split("_", 1)[0]
        if accession not in metadata.index:
            raise KeyError(f"No GEO metadata matched spatial file {path.name}.")
        row = metadata.loc[accession]
        sample = read_cellbin_gef_targeted(path, retained_genes)
        before = sample.n_obs
        if "full_gene_count" in sample.obs:
            keep = pd.to_numeric(sample.obs["full_gene_count"], errors="coerce").fillna(0).ge(200).to_numpy()
            sample = sample[keep].copy()
        annotate_major_brain_cell_types(sample)
        sample.obs_names = pd.Index(accession + ":" + sample.obs_names.astype(str), name="cell_id")
        sample.obs["donor_id"] = accession
        sample.obs["analysis_unit"] = accession
        sample.obs["pathology"] = str(row["pathology"])
        sample.obs["subpopulation"] = sample.obs["cell_type_annotation_confidence"].astype(str)
        sample.obs["brain_region"] = "prefrontal cortex BA10"
        sample.obs["cohort"] = cohort
        sample.obs["donor_inference_eligible"] = True
        sample.obs["source_gef"] = path.name
        objects.append(sample)
        summaries.append(
            {
                "geo_accession": accession,
                "pathology": row["pathology"],
                "cells_before_qc": before,
                "cells_after_qc": sample.n_obs,
                "rhbdf2_detected_cells": int((np.asarray(sample[:, "RHBDF2"].X.toarray()).reshape(-1) > 0).sum())
                if "RHBDF2" in sample.var_names else 0,
            }
        )
    # Every donor is read against the same preregistered gene list; an inner join
    # is the AnnData-compatible equivalent of the intended exact feature join.
    combined = ad.concat(objects, axis=0, join="inner", merge="same", index_unique=None)
    combined.uns["matrix_semantics"] = {
        "X": "Stereo-seq cell-bin MID counts restricted to preregistered candidate, program, and lineage-marker genes",
        "feature_scope": "targeted; not transcriptome-wide",
        "spatial_coordinates": "author-provided cell centroid x-y coordinates in obsm['spatial']",
        "cell_type_annotation": "prespecified marker-module maximum score",
    }
    return combined, pd.DataFrame(summaries)


def spatial_donor_detection_summary(adata: ad.AnnData, gene: str = "RHBDF2") -> pd.DataFrame:
    expression = np.asarray(adata[:, gene].X.toarray()).reshape(-1)
    frame = adata.obs[["donor_id", "pathology", "cell_type"]].copy()
    frame["expression"] = expression
    return frame.groupby(["donor_id", "pathology", "cell_type"], observed=True).agg(
        n_cells=("expression", "size"),
        detected_cells=("expression", lambda values: int(np.sum(values > 0))),
        detection_fraction=("expression", lambda values: float(np.mean(values > 0))),
        mean_targeted_count=("expression", "mean"),
    ).reset_index()


def compare_spatial_detection_by_pathology(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell_type, frame in summary.groupby("cell_type", observed=True):
        case = frame.loc[frame["pathology"].eq("AD"), "detection_fraction"].to_numpy(dtype=float)
        control = frame.loc[frame["pathology"].eq("Control"), "detection_fraction"].to_numpy(dtype=float)
        statistic, pvalue = stats.ttest_ind(case, control, equal_var=False) if len(case) >= 2 and len(control) >= 2 else (np.nan, np.nan)
        rows.append(
            {
                "cell_type": cell_type,
                "ad_donors": len(case),
                "control_donors": len(control),
                "ad_mean_detection_fraction": np.mean(case) if len(case) else np.nan,
                "control_mean_detection_fraction": np.mean(control) if len(control) else np.nan,
                "difference_ad_minus_control": np.mean(case) - np.mean(control) if len(case) and len(control) else np.nan,
                "welch_t": statistic,
                "pval": pvalue,
            }
        )
    result = pd.DataFrame(rows)
    result["pvals_adj"] = _fdr_bh(result["pval"])
    return result


def spatial_neighbor_composition(
    adata: ad.AnnData, gene: str = "RHBDF2", neighbors: int = 15
) -> pd.DataFrame:
    """Summarize cell-type composition around RHBDF2-detected cells without LR inference."""
    expression = np.asarray(adata[:, gene].X.toarray()).reshape(-1)
    rows = []
    for donor, indices in adata.obs.groupby("donor_id", observed=True).indices.items():
        indices = np.asarray(indices, dtype=int)
        detected_local = np.flatnonzero(expression[indices] > 0)
        if detected_local.size == 0 or indices.size <= neighbors:
            continue
        coordinates = np.asarray(adata.obsm["spatial"])[indices]
        tree = cKDTree(coordinates)
        _, neighbor_index = tree.query(coordinates[detected_local], k=min(neighbors + 1, indices.size))
        neighbor_index = np.atleast_2d(neighbor_index)[:, 1:]
        anchor_types = adata.obs.iloc[indices[detected_local]]["cell_type"].astype(str).to_numpy()
        all_types = adata.obs.iloc[indices]["cell_type"].astype(str).to_numpy()
        for anchor_type in np.unique(anchor_types):
            selected_neighbors = neighbor_index[anchor_types == anchor_type].reshape(-1)
            counts = pd.Series(all_types[selected_neighbors]).value_counts()
            total = int(counts.sum())
            for neighbor_type, count in counts.items():
                rows.append(
                    {
                        "donor_id": donor,
                        "pathology": str(adata.obs.iloc[indices[0]]["pathology"]),
                        "anchor_cell_type": anchor_type,
                        "neighbor_cell_type": neighbor_type,
                        "anchor_detected_cells": int(np.sum(anchor_types == anchor_type)),
                        "neighbor_observations": int(count),
                        "neighbor_fraction": float(count / total),
                        "k_neighbors": neighbors,
                        "interpretation": "descriptive_spatial_neighborhood_not_ligand_receptor_inference",
                    }
                )
    return pd.DataFrame(rows)


def _find_group(handle: h5py.File, name: str) -> h5py.Group:
    if name in handle:
        return handle[name]
    matches = []
    handle.visititems(lambda path, item: matches.append(item) if isinstance(item, h5py.Group) and path.split("/")[-1] == name else None)
    if len(matches) != 1:
        raise KeyError(f"Expected one '{name}' group, found {len(matches)}.")
    return matches[0]


def _gene_names(gene: np.ndarray) -> list[str]:
    values = _structured_field(gene, ("geneName", "gene", "geneID"))
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _field(record: np.void, alternatives: tuple[str, ...]) -> Any:
    for field in alternatives:
        if record.dtype.names and field in record.dtype.names:
            return record[field]
    raise KeyError(f"None of the fields {alternatives} were present in {record.dtype.names}.")


def _structured_field(array: np.ndarray, alternatives: tuple[str, ...]) -> np.ndarray:
    for field in alternatives:
        if array.dtype.names and field in array.dtype.names:
            return array[field]
    raise KeyError(f"None of the fields {alternatives} were present in {array.dtype.names}.")


def _cell_row_indices(cell: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
    if cell_ids.size == 0:
        return cell_ids
    if cell_ids.min() >= 0 and cell_ids.max() < len(cell):
        return cell_ids
    ids = np.asarray(_structured_field(cell, ("id", "cellID", "cellId")), dtype=np.int64)
    mapping = {value: index for index, value in enumerate(ids)}
    return np.asarray([mapping.get(value, -1) for value in cell_ids], dtype=np.int64)


def _fdr_bh(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(numeric.shape, np.nan)
    valid = np.isfinite(numeric)
    if not valid.any():
        return adjusted
    p = numeric[valid]
    order = np.argsort(p)
    ranked = p[order]
    corrected = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty_like(corrected)
    restored[order] = np.minimum(corrected, 1.0)
    adjusted[valid] = restored
    return adjusted
