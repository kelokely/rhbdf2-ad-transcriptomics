"""Utilities for the GSE243292 multi-cell Alzheimer extension."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse
from scipy.stats import f as f_distribution
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests

from .continuous_evidence_utils import run_continuous_evidence
from .gene_utils import extract_expression_vector, gene_presence_table


CELL_TYPE_LABELS = {
    "Mic": "Microglia",
    "Ast": "Astrocytes",
    "Oli": "Oligodendrocytes",
    "Opc": "OPCs",
    "Ex": "Excitatory neurons",
    "In": "Inhibitory neurons",
    "End": "Endothelial cells",
}


@dataclass
class DonorPseudobulkResult:
    """Continuous donor-level pseudobulk outputs for one population."""

    sample_metadata: pd.DataFrame
    log_cpm: pd.DataFrame
    full_results: pd.DataFrame
    candidate_results: pd.DataFrame
    status: str
    reason: str


def load_backed_subset(path: Path, mask: np.ndarray) -> ad.AnnData:
    """Load only X, obs, and var for a backed H5AD subset, excluding large auxiliary layers."""
    backed = sc.read_h5ad(path, backed="r")
    try:
        positions = np.flatnonzero(np.asarray(mask, dtype=bool))
        matrix = backed.X[positions]
        if hasattr(matrix, "to_memory"):
            matrix = matrix.to_memory()
        if sparse.issparse(matrix):
            matrix = matrix.tocsr()
        else:
            matrix = np.asarray(matrix)
        obs = backed.obs.iloc[positions].copy()
        var = backed.var.copy()
    finally:
        backed.file.close()

    result = ad.AnnData(X=matrix, obs=obs, var=var)
    result.obs_names = _stable_obs_names(result.obs)
    return result


def load_backed_population(path: Path, column: str, value: str) -> ad.AnnData:
    """Load one annotated population from a backed H5AD object."""
    backed = sc.read_h5ad(path, backed="r")
    try:
        if column not in backed.obs.columns:
            raise ValueError(f"Population column '{column}' was not found in {path}.")
        mask = backed.obs[column].astype(str).to_numpy() == str(value)
    finally:
        backed.file.close()
    return load_backed_subset(path, mask)


def build_cell_type_eligibility(
    full_path: Path,
    output_path: Path,
    gene: str,
    cell_type_column: str,
    donor_column: str,
    pathology_column: str,
    minimum_cells: int,
    minimum_detected_cells: int,
    minimum_donors_with_detection: int,
) -> pd.DataFrame:
    """Summarize expression and metadata coverage for every annotated population."""
    backed = sc.read_h5ad(full_path, backed="r")
    try:
        expression = _extract_backed_gene(backed, gene)
        obs = backed.obs[[cell_type_column, donor_column, pathology_column]].copy()
    finally:
        backed.file.close()

    obs["_expression"] = expression
    rows: list[dict[str, object]] = []
    for cell_type, frame in obs.groupby(cell_type_column, observed=True):
        pathology_donors = (
            frame[[donor_column, pathology_column]]
            .drop_duplicates()
            .groupby(pathology_column, observed=True)[donor_column]
            .nunique()
            .to_dict()
        )
        detected = frame["_expression"] > 0
        n_cells = int(frame.shape[0])
        n_detected = int(detected.sum())
        n_donors_detected = int(frame.loc[detected, donor_column].nunique())
        eligible = (
            n_cells >= minimum_cells
            and n_detected >= minimum_detected_cells
            and n_donors_detected >= minimum_donors_with_detection
        )
        reasons: list[str] = []
        if n_cells < minimum_cells:
            reasons.append(f"fewer than {minimum_cells} cells")
        if n_detected < minimum_detected_cells:
            reasons.append(f"fewer than {minimum_detected_cells} RHBDF2-detected cells")
        if n_donors_detected < minimum_donors_with_detection:
            reasons.append(f"fewer than {minimum_donors_with_detection} donors with detection")
        rows.append(
            {
                "cell_type_code": str(cell_type),
                "cell_type": CELL_TYPE_LABELS.get(str(cell_type), str(cell_type)),
                "n_cells": n_cells,
                "n_donors": int(frame[donor_column].nunique()),
                "rhbdf2_detected_cells": n_detected,
                "rhbdf2_detection_fraction": n_detected / n_cells if n_cells else np.nan,
                "donors_with_rhbdf2_detection": n_donors_detected,
                "pathology_levels": int(frame[pathology_column].nunique()),
                "pathology_donor_coverage": json.dumps({str(k): int(v) for k, v in pathology_donors.items()}, sort_keys=True),
                "cell_level_continuous_eligible": eligible,
                "donor_continuous_pseudobulk_eligible": (
                    eligible and int(frame[donor_column].nunique()) >= 5
                ),
                "analysis_status": "eligible" if eligible else "descriptive_only",
                "eligibility_note": "; ".join(reasons),
            }
        )

    rows.append(
        {
            "cell_type_code": "Pericyte",
            "cell_type": "Pericytes",
            "n_cells": 0,
            "n_donors": 0,
            "rhbdf2_detected_cells": 0,
            "rhbdf2_detection_fraction": np.nan,
            "donors_with_rhbdf2_detection": 0,
            "pathology_levels": 0,
            "pathology_donor_coverage": "{}",
            "cell_level_continuous_eligible": False,
            "donor_continuous_pseudobulk_eligible": False,
            "analysis_status": "not_annotated",
            "eligibility_note": "Pericytes are not separately annotated in the supplied full-brain object.",
        }
    )
    result = pd.DataFrame(rows).sort_values("n_cells", ascending=False, na_position="last").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def build_subpopulation_eligibility(
    adata: ad.AnnData,
    output_path: Path,
    gene: str,
    subgroup_column: str,
    donor_column: str,
    pathology_column: str,
    excluded_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize RHBDF2 support for clean and excluded astrocyte subpopulations."""
    expression = extract_expression_vector(adata, gene)
    obs = adata.obs[[subgroup_column, donor_column, pathology_column]].copy()
    obs["_expression"] = expression
    excluded = set(excluded_labels or [])
    rows: list[dict[str, object]] = []
    for subgroup, frame in obs.groupby(subgroup_column, observed=True):
        detected = frame["_expression"] > 0
        pathology_donors = (
            frame[[donor_column, pathology_column]]
            .drop_duplicates()
            .groupby(pathology_column, observed=True)[donor_column]
            .nunique()
            .to_dict()
        )
        rows.append(
            {
                "subpopulation": str(subgroup),
                "primary_analysis": str(subgroup) not in excluded,
                "n_cells": int(frame.shape[0]),
                "n_donors": int(frame[donor_column].nunique()),
                "rhbdf2_detected_cells": int(detected.sum()),
                "rhbdf2_detection_fraction": float(detected.mean()),
                "donors_with_rhbdf2_detection": int(frame.loc[detected, donor_column].nunique()),
                "pathology_donor_coverage": json.dumps({str(k): int(v) for k, v in pathology_donors.items()}, sort_keys=True),
                "note": "excluded annotated doublet group" if str(subgroup) in excluded else "clean astrocyte subpopulation",
            }
        )
    result = pd.DataFrame(rows).sort_values("n_cells", ascending=False).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def normalize_log1p_copy(adata: ad.AnnData, target_sum: float = 10000.0) -> ad.AnnData:
    """Normalize and log-transform a sparse analysis copy without densifying it."""
    result = adata.copy()
    sc.pp.normalize_total(result, target_sum=target_sum)
    sc.pp.log1p(result)
    return result


def filter_myeloid_like_cells(
    adata: ad.AnnData,
    marker_genes: list[str],
    output_path: Path,
    minimum_markers: int = 2,
) -> ad.AnnData:
    """Conservatively remove non-microglial cells expressing multiple myeloid identity markers."""
    present = [gene for gene in marker_genes if gene in adata.var_names]
    if not present:
        raise ValueError("None of the configured myeloid QC markers were present.")
    marker_matrix = adata[:, present].X
    if sparse.issparse(marker_matrix):
        detected_counts = np.asarray((marker_matrix > 0).sum(axis=1)).reshape(-1)
    else:
        detected_counts = (np.asarray(marker_matrix) > 0).sum(axis=1)
    remove = detected_counts >= minimum_markers
    rhbdf2 = extract_expression_vector(adata, "RHBDF2")
    summary = pd.DataFrame(
        [
            {
                "cells_before": int(adata.n_obs),
                "cells_removed": int(remove.sum()),
                "cells_after": int((~remove).sum()),
                "rhbdf2_detected_before": int((rhbdf2 > 0).sum()),
                "rhbdf2_detected_removed": int(((rhbdf2 > 0) & remove).sum()),
                "minimum_myeloid_markers": int(minimum_markers),
                "markers_present": ";".join(present),
            }
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return adata[~remove].copy()


def run_cell_level_continuous_layer(
    raw_adata: ad.AnnData,
    output_dir: Path,
    candidate_genes: list[str],
    donor_column: str,
    target_sum: float = 10000.0,
    anchor_gene: str = "RHBDF2",
) -> object:
    """Run exploratory threshold-free and donor-adjusted hurdle models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_key = _gene_key(anchor_gene)
    presence = gene_presence_table(raw_adata, list(dict.fromkeys([anchor_gene, *candidate_genes])))
    presence.to_csv(output_dir / "candidate_gene_presence.csv", index=False)
    processed = normalize_log1p_copy(raw_adata, target_sum=target_sum)
    result = run_continuous_evidence(
        adata=processed,
        gene_of_interest=anchor_gene,
        candidate_genes=candidate_genes,
        continuous_config={"enabled": True, "top_n_export": 100, "min_positive_cells": 25, "donor_column": donor_column},
        output_dir=output_dir,
        fallback_donor_column=donor_column,
    )
    hurdle = result.candidate_hurdle_summary.copy()
    detection_coefficient = f"detection_coef_per_log1p_{anchor_key}"
    positive_coefficient = f"positive_expression_coef_per_log1p_{anchor_key}"
    hurdle["detection_practical_support"] = (
        hurdle["detection_padj"].fillna(1.0).lt(0.05)
        & hurdle[detection_coefficient].abs().ge(0.05)
    )
    hurdle["positive_expression_practical_support"] = (
        hurdle["positive_expression_padj"].fillna(1.0).lt(0.05)
        & hurdle[positive_coefficient].abs().ge(0.05)
    )
    suffix = "" if anchor_key == "rhbdf2" else f"_{anchor_key}"
    hurdle.to_csv(output_dir / f"candidate_hurdle_interpretable_summary{suffix}.csv", index=False)
    result.candidate_hurdle_summary = hurdle
    return result


def run_continuous_donor_pseudobulk(
    adata: ad.AnnData,
    output_dir: Path,
    candidate_genes: list[str],
    donor_column: str,
    pathology_column: str,
    minimum_donors: int = 5,
    adjust_for_pathology: bool = True,
    normalization_target_sum: float = 1_000_000.0,
    anchor_gene: str = "RHBDF2",
    minimum_cells_per_donor: int = 20,
    covariance_method: str = "OLS",
    adjust_for_library_depth: bool = False,
    minimum_residual_degrees_freedom: int = 2,
) -> DonorPseudobulkResult:
    """Aggregate one sample per donor and regress expression on donor anchor detection rate."""
    output_dir.mkdir(parents=True, exist_ok=True)
    empty = pd.DataFrame()
    anchor_key = _gene_key(anchor_gene)
    predictor_column = f"{anchor_key}_detection_rate_log1p_per_1000"
    if adata.n_obs == 0:
        return _write_ineligible_pseudobulk(output_dir, "No cells were available.", anchor_gene=anchor_gene)

    aggregated, metadata = _aggregate_by_donor(
        adata=adata,
        donor_column=donor_column,
        pathology_column=pathology_column,
        anchor_gene=anchor_gene,
    )
    if minimum_cells_per_donor < 1:
        raise ValueError("minimum_cells_per_donor must be positive.")
    retained = metadata["n_cells"].ge(minimum_cells_per_donor)
    metadata.assign(retained_for_model=retained).to_csv(output_dir / f"donor_cell_qc_{anchor_key}.csv", index=False)
    aggregated = aggregated.loc[retained.to_numpy()].reset_index(drop=True)
    metadata = metadata.loc[retained].reset_index(drop=True)
    if metadata.shape[0] < minimum_donors:
        return _write_ineligible_pseudobulk(
            output_dir,
            f"Only {metadata.shape[0]} donors were available; minimum is {minimum_donors}.",
            metadata=metadata,
            anchor_gene=anchor_gene,
        )
    predictor = metadata[predictor_column].to_numpy(dtype=float)
    if np.unique(predictor).size < 3 or np.isclose(np.std(predictor), 0.0):
        return _write_ineligible_pseudobulk(
            output_dir,
            f"Donor-level {anchor_gene} detection had insufficient variation.",
            metadata=metadata,
            anchor_gene=anchor_gene,
        )

    log_cpm = _normalize_log_cpm(
        aggregated,
        normalization_target_sum,
        library_sizes=metadata["normalization_library_size"],
    )
    covariates = pd.DataFrame(index=metadata.index)
    if adjust_for_pathology and metadata[pathology_column].nunique() > 1:
        pathology = pd.get_dummies(metadata[pathology_column].astype(str), prefix="pathology", drop_first=True, dtype=float)
        covariates = pd.concat([covariates, pathology], axis=1)
    if adjust_for_library_depth:
        covariates["log_library_per_cell"] = np.log(metadata["normalization_library_size"] / metadata["n_cells"])
    results = _fit_continuous_matrix_model(
        values=log_cpm.to_numpy(dtype=float),
        feature_names=log_cpm.columns.astype(str).tolist(),
        predictor=predictor,
        covariates=covariates,
        effect_column=f"effect_per_log1p_{anchor_key}_detection_per_1000_cells",
        covariance_method=covariance_method,
        minimum_residual_degrees_freedom=minimum_residual_degrees_freedom,
    )
    results["n_donors"] = int(metadata.shape[0])
    results["pathology_adjusted"] = bool(not covariates.empty)
    results["anchor_gene"] = anchor_gene
    results["self_association"] = results["names"] == anchor_gene
    results["minimum_cells_per_donor"] = minimum_cells_per_donor
    results["library_depth_adjusted"] = adjust_for_library_depth

    present_candidates = [gene for gene in dict.fromkeys(candidate_genes) if gene in results["names"].values]
    candidate_results = results[results["names"].isin(present_candidates)].copy()
    candidate_results = candidate_results.sort_values(["pvals_adj", "pvals", "names"]).reset_index(drop=True)
    suffix = "" if anchor_key == "rhbdf2" else f"_{anchor_key}"
    metadata.to_csv(output_dir / f"donor_pseudobulk_metadata{suffix}.csv", index=False)
    log_cpm[present_candidates].to_csv(output_dir / f"donor_candidate_log1p_cpm{suffix}.csv", index=False)
    results.to_csv(output_dir / f"donor_continuous_pseudobulk_all_genes{suffix}.csv", index=False)
    candidate_results.to_csv(output_dir / f"donor_continuous_pseudobulk_candidates{suffix}.csv", index=False)
    pd.DataFrame([{"status": "completed", "reason": "", "anchor_gene": anchor_gene}]).to_csv(
        output_dir / f"status{suffix}.csv", index=False
    )
    return DonorPseudobulkResult(metadata, log_cpm, results, candidate_results, "completed", "")


def run_program_associations(
    donor_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    program_genes: dict[str, list[str]],
    output_path: Path,
    pathology_column: str,
    anchor_gene: str = "RHBDF2",
) -> pd.DataFrame:
    """Test donor-level program scores against anchor detection within each cell type."""
    anchor_key = _gene_key(anchor_gene)
    predictor_column = f"{anchor_key}_detection_rate_log1p_per_1000"
    rows: list[dict[str, object]] = []
    for cell_type, (metadata, log_cpm) in donor_tables.items():
        predictor = metadata[predictor_column].to_numpy(dtype=float)
        pathology = pd.get_dummies(metadata[pathology_column].astype(str), prefix="pathology", drop_first=True, dtype=float)
        for program, requested_genes in program_genes.items():
            present = [gene for gene in requested_genes if gene in log_cpm.columns and gene != anchor_gene]
            if len(present) < 2:
                rows.append(
                    {
                        "cell_type": cell_type,
                        "program": program,
                        "n_genes_present": len(present),
                        "present_genes": ";".join(present),
                        "effect": np.nan,
                        "pval": np.nan,
                        "status": "not_eligible",
                        "reason": "fewer than two program genes were present",
                    }
                )
                continue
            gene_values = log_cpm[present].to_numpy(dtype=float)
            means = gene_values.mean(axis=0)
            std = gene_values.std(axis=0, ddof=1)
            std[std == 0] = 1.0
            score = ((gene_values - means) / std).mean(axis=1)
            fit = _fit_single_continuous_model(score, predictor, pathology)
            rows.append(
                {
                    "cell_type": cell_type,
                    "anchor_gene": anchor_gene,
                    "program": program,
                    "n_genes_present": len(present),
                    "present_genes": ";".join(present),
                    "effect": fit["effect"],
                    "t_statistic": fit["t_statistic"],
                    "pval": fit["pval"],
                    "n_donors": int(metadata.shape[0]),
                    "status": "completed",
                    "reason": "",
                }
            )
    result = pd.DataFrame(rows)
    result["pvals_adj_global"] = _adjust_available(result["pval"])
    result["pvals_adj_within_cell_type"] = np.nan
    for _, indices in result.groupby("cell_type", observed=True).groups.items():
        result.loc[indices, "pvals_adj_within_cell_type"] = _adjust_available(result.loc[indices, "pval"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def summarize_program_specificity(program_results: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Classify FDR-supported programs as microglial, glial, shared, or unsupported."""
    rows: list[dict[str, object]] = []
    for program, frame in program_results.groupby("program", observed=True):
        supported = frame[frame["pvals_adj_global"].fillna(1.0) < 0.05].copy()
        positive = supported.loc[supported["effect"] > 0, "cell_type"].astype(str).tolist()
        negative = supported.loc[supported["effect"] < 0, "cell_type"].astype(str).tolist()
        types = set(supported["cell_type"].astype(str))
        glia = {"Microglia", "Astrocytes", "Oligodendrocytes", "OPCs"}
        if not types:
            classification = "unsupported_at_donor_fdr"
        elif types == {"Microglia"}:
            classification = "microglia_specific"
        elif len(types) >= 2 and types.issubset(glia):
            classification = "shared_glial"
        elif len(types) >= 2:
            classification = "broad_multicell"
        else:
            classification = "single_nonmicroglial_cell_type"
        rows.append(
            {
                "program": program,
                "classification": classification,
                "n_supported_cell_types": len(types),
                "supported_cell_types": ";".join(sorted(types)),
                "positive_cell_types": ";".join(sorted(positive)),
                "negative_cell_types": ";".join(sorted(negative)),
            }
        )
    result = pd.DataFrame(rows).sort_values(["n_supported_cell_types", "program"], ascending=[False, True])
    result.to_csv(output_path, index=False)
    return result


def build_donor_cross_cell_table(
    donor_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine donor-level RHBDF2 detection and compute cross-cell-type correlations."""
    rows: list[pd.DataFrame] = []
    for cell_type, (metadata, _) in donor_tables.items():
        frame = metadata.copy()
        frame["cell_type"] = cell_type
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "donor_rhbdf2_by_cell_type.csv", index=False)
    pivot = combined.pivot(index="donor_id", columns="cell_type", values="rhbdf2_detection_fraction")
    correlation = pivot.corr(method="spearman")
    correlation.to_csv(output_dir / "donor_rhbdf2_detection_spearman.csv")
    return combined, correlation


def test_donor_rhbdf2_pathology_association(
    donor_table: pd.DataFrame,
    pathology_column: str,
    output_path: Path,
) -> pd.DataFrame:
    """Test donor-level RHBDF2 detection differences across pathology groups per cell type."""
    rows: list[dict[str, object]] = []
    for cell_type, frame in donor_table.groupby("cell_type", observed=True):
        frame = frame.dropna(subset=["rhbdf2_detection_rate_log1p_per_1000", pathology_column]).copy()
        y = frame["rhbdf2_detection_rate_log1p_per_1000"].to_numpy(dtype=float)
        groups = frame[pathology_column].astype(str)
        dummies = pd.get_dummies(groups, drop_first=True, dtype=float)
        reduced = np.ones((y.shape[0], 1))
        full = np.column_stack([np.ones(y.shape[0]), dummies.to_numpy(dtype=float)])
        reduced_residual = y - reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
        full_residual = y - full @ np.linalg.lstsq(full, y, rcond=None)[0]
        reduced_sse = float(np.square(reduced_residual).sum())
        full_sse = float(np.square(full_residual).sum())
        numerator_df = int(full.shape[1] - reduced.shape[1])
        denominator_df = int(y.shape[0] - np.linalg.matrix_rank(full))
        if numerator_df <= 0 or denominator_df <= 0 or full_sse <= 0:
            f_statistic = np.nan
            pvalue = np.nan
        else:
            f_statistic = max(0.0, ((reduced_sse - full_sse) / numerator_df) / (full_sse / denominator_df))
            pvalue = float(f_distribution.sf(f_statistic, numerator_df, denominator_df))
        rows.append(
            {
                "cell_type": str(cell_type),
                "n_donors": int(frame.shape[0]),
                "n_pathology_groups": int(groups.nunique()),
                "f_statistic": f_statistic,
                "pval": pvalue,
                "variance_explained_r_squared": 1.0 - (full_sse / reduced_sse) if reduced_sse > 0 else np.nan,
                "group_means_log1p_detection_per_1000": json.dumps(
                    {str(k): float(v) for k, v in frame.groupby(pathology_column, observed=True)["rhbdf2_detection_rate_log1p_per_1000"].mean().items()},
                    sort_keys=True,
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["pvals_adj"] = _adjust_available(result["pval"])
    result = result.sort_values(["pvals_adj", "pval", "cell_type"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


# This is an analysis helper retained under its historical public name, not a pytest test.
test_donor_rhbdf2_pathology_association.__test__ = False


def evaluate_ligand_receptor_gate(
    program_results: pd.DataFrame,
    astro_validation: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Open ligand-receptor analysis only after replicated donor-level program support."""
    significant = program_results[program_results["pvals_adj_global"].fillna(1.0) < 0.05]
    astro_program_supported = not significant[significant["cell_type"] == "Astrocytes"].empty
    astro_supported = astro_validation[
        (astro_validation.get("pvals_adj", pd.Series(dtype=float)).fillna(1.0) < 0.05)
        & (~astro_validation.get("self_association", pd.Series(False, index=astro_validation.index)).fillna(False).astype(bool))
        & (astro_validation.get("names", pd.Series("", index=astro_validation.index)).astype(str) != "RHBDF2")
    ] if not astro_validation.empty else astro_validation
    other_nonmicroglial = significant[
        ~significant["cell_type"].isin(["Astrocytes", "Microglia"])
    ]
    passed = astro_program_supported and not astro_supported.empty and not other_nonmicroglial.empty
    reason = (
        "Astrocytes, an independent astrocyte stratum, and another non-microglial population showed donor-level global-FDR support."
        if passed
        else "The dedicated astrocyte and cross-cell results did not meet the predeclared non-self global-FDR stability gate; ligand-receptor analysis was not run."
    )
    result = pd.DataFrame(
        [
            {
                "status": "eligible" if passed else "skipped",
                "gate_passed": passed,
                "global_fdr_supported_cell_types": int(significant["cell_type"].nunique()),
                "astro_program_global_fdr_support": bool(astro_program_supported),
                "astro_validation_fdr_hits": int(astro_supported.shape[0]),
                "other_nonmicroglial_global_fdr_hits": int(other_nonmicroglial.shape[0]),
                "reason": reason,
            }
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def save_extension_plots(
    eligibility: pd.DataFrame,
    donor_correlation: pd.DataFrame,
    program_results: pd.DataFrame,
    output_dir: Path,
    dpi: int = 300,
) -> list[Path]:
    """Save compact figures with labels sized to avoid overlap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    paths: list[Path] = []

    plot_data = eligibility[eligibility["n_cells"] > 0].sort_values("rhbdf2_detection_fraction")
    percentages = plot_data["rhbdf2_detection_fraction"] * 100
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    bars = ax.barh(plot_data["cell_type"], percentages, color="#2b6f6d")
    ax.set_xscale("log")
    for bar, value in zip(bars, percentages, strict=True):
        ax.text(value * 1.08, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontsize=11)
    ax.set_xlim(max(0.2, percentages.min() * 0.7), percentages.max() * 1.55)
    ax.set_xlabel("RHBDF2-detected cells (%)")
    ax.set_ylabel("")
    ax.set_title("RHBDF2 detection by annotated brain cell type")
    path = output_dir / "rhbdf2_detection_by_cell_type.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    sns.heatmap(donor_correlation, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Donor-level RHBDF2 detection correlations")
    ax.set_xlabel("")
    ax.set_ylabel("")
    path = output_dir / "donor_rhbdf2_detection_correlation_heatmap.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    completed = program_results[program_results["status"] == "completed"].copy()
    if not completed.empty:
        matrix = completed.pivot(index="program", columns="cell_type", values="effect")
        significance = completed.pivot(index="program", columns="cell_type", values="pvals_adj_global").reindex_like(matrix)
        annotations = matrix.map(lambda value: f"{value:.2f}" if pd.notna(value) else "")
        annotations = annotations + significance.map(lambda value: "*" if pd.notna(value) and value < 0.05 else "")
        fig, ax = plt.subplots(figsize=(11, max(6.5, 0.55 * matrix.shape[0] + 2)), constrained_layout=True)
        sns.heatmap(matrix, cmap="vlag", center=0, annot=annotations, fmt="", ax=ax)
        ax.set_title("Pathology-adjusted donor-level program associations\n* denotes global FDR < 0.05")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=35)
        path = output_dir / "program_association_heatmap.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _aggregate_by_donor(
    adata: ad.AnnData,
    donor_column: str,
    pathology_column: str,
    anchor_gene: str = "RHBDF2",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if donor_column not in adata.obs.columns or pathology_column not in adata.obs.columns:
        raise ValueError("Donor and pathology columns are required for donor pseudobulk.")
    anchor_key = _gene_key(anchor_gene)
    anchor_expression = extract_expression_vector(adata, anchor_gene)
    rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    donor_order: list[str] = []
    for donor, positions in adata.obs.groupby(donor_column, observed=True).indices.items():
        positions_array = np.asarray(positions, dtype=int)
        matrix = adata.X[positions_array]
        summed = np.asarray(matrix.sum(axis=0)).reshape(-1) if sparse.issparse(matrix) else np.asarray(matrix).sum(axis=0)
        donor_expression = anchor_expression[positions_array]
        detected = int(np.sum(donor_expression > 0))
        n_cells = int(positions_array.size)
        pathology_values = adata.obs.iloc[positions_array][pathology_column].dropna().astype(str).unique()
        pathology = pathology_values[0] if pathology_values.size == 1 else "mixed"
        retained_panel_library_size = float(np.sum(summed))
        normalization_library_size = retained_panel_library_size
        normalization_library_source = "summed_adata_x"
        for full_library_column in ("source_library_counts", "source_total_umis", "total_counts"):
            if full_library_column not in adata.obs.columns:
                continue
            source_totals = pd.to_numeric(adata.obs.iloc[positions_array][full_library_column], errors="coerce")
            if source_totals.isna().any() or (source_totals <= 0).any():
                raise ValueError(f"Invalid full-library counts in {full_library_column} for donor {donor}.")
            normalization_library_size = float(source_totals.sum())
            normalization_library_source = f"summed_{full_library_column}"
            break
        donor_order.append(str(donor))
        rows.append(np.asarray(summed, dtype=float))
        metadata_rows.append(
            {
                "donor_id": str(donor),
                pathology_column: pathology,
                "n_cells": n_cells,
                "raw_library_size": retained_panel_library_size,
                "normalization_library_size": normalization_library_size,
                "normalization_library_source": normalization_library_source,
                f"{anchor_key}_detected_cells": detected,
                f"{anchor_key}_detection_fraction": detected / n_cells if n_cells else np.nan,
                f"{anchor_key}_detection_rate_log1p_per_1000": math.log1p(1000.0 * detected / n_cells) if n_cells else np.nan,
                f"{anchor_key}_summed_counts": float(np.sum(donor_expression)),
            }
        )
    metadata = pd.DataFrame(metadata_rows).sort_values("donor_id").reset_index(drop=True)
    aggregated = pd.DataFrame(
        np.vstack(rows),
        index=pd.Index(donor_order, name="donor_id"),
        columns=adata.var_names.astype(str),
    )
    aggregated = aggregated.loc[metadata["donor_id"]].reset_index(drop=True)
    return aggregated, metadata


def _normalize_log_cpm(
    counts: pd.DataFrame,
    target_sum: float,
    library_sizes: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    if library_sizes is None:
        library = counts.sum(axis=1)
    else:
        library = pd.Series(np.asarray(library_sizes, dtype=float), index=counts.index)
    library = library.replace(0, np.nan)
    return np.log1p(counts.div(library, axis=0).fillna(0.0) * target_sum)


def _fit_continuous_matrix_model(
    values: np.ndarray,
    feature_names: list[str],
    predictor: np.ndarray,
    covariates: pd.DataFrame,
    effect_column: str,
    covariance_method: str = "OLS",
    minimum_residual_degrees_freedom: int = 2,
) -> pd.DataFrame:
    design_parts = [np.ones((predictor.shape[0], 1)), predictor.reshape(-1, 1)]
    if not covariates.empty:
        design_parts.append(covariates.to_numpy(dtype=float))
    design = np.column_stack(design_parts)
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise ValueError("Continuous donor model design is rank deficient.")
    degrees_freedom = int(design.shape[0] - rank)
    if degrees_freedom < minimum_residual_degrees_freedom:
        raise ValueError(f"Continuous donor model has fewer than {minimum_residual_degrees_freedom} residual degrees of freedom.")
    inverse = np.linalg.pinv(design.T @ design)
    coefficients = inverse @ design.T @ values
    residuals = values - design @ coefficients
    sigma2 = np.square(residuals).sum(axis=0) / degrees_freedom
    if covariance_method == "OLS":
        standard_error = np.sqrt(np.maximum(sigma2 * inverse[1, 1], 0.0))
    elif covariance_method == "HC3":
        influence = design @ inverse
        leverage = np.sum(influence * design, axis=1)
        if np.any(leverage >= 1 - 1e-10):
            raise ValueError("HC3 covariance is undefined for unit-leverage donor profiles.")
        weights = np.square(influence[:, 1] / (1 - leverage))
        standard_error = np.sqrt(weights @ np.square(residuals))
    else:
        raise ValueError("covariance_method must be OLS or HC3.")
    effects = coefficients[1]
    t_statistics = np.divide(effects, standard_error, out=np.zeros_like(effects), where=standard_error > 0)
    pvalues = 2.0 * student_t.sf(np.abs(t_statistics), df=degrees_freedom)
    pvalues = np.where(np.isfinite(pvalues), pvalues, 1.0)
    adjusted = multipletests(pvalues, method="fdr_bh")[1]
    result = pd.DataFrame(
        {
            "names": feature_names,
            effect_column: effects,
            "t_statistic": t_statistics,
            "pvals": pvalues,
            "pvals_adj": adjusted,
            "residual_degrees_freedom": degrees_freedom,
            "standard_error": standard_error,
            "ci95_low": effects - student_t.ppf(0.975, degrees_freedom) * standard_error,
            "ci95_high": effects + student_t.ppf(0.975, degrees_freedom) * standard_error,
            "covariance_method": covariance_method,
        }
    )
    return result.sort_values(["pvals_adj", "pvals", effect_column], ascending=[True, True, False]).reset_index(drop=True)


def _fit_single_continuous_model(values: np.ndarray, predictor: np.ndarray, covariates: pd.DataFrame) -> dict[str, float]:
    result = _fit_continuous_matrix_model(
        values=np.asarray(values, dtype=float).reshape(-1, 1),
        feature_names=["program"],
        predictor=np.asarray(predictor, dtype=float),
        covariates=covariates,
        effect_column="effect",
    ).iloc[0]
    return {"effect": float(result["effect"]), "t_statistic": float(result["t_statistic"]), "pval": float(result["pvals"])}


def _write_ineligible_pseudobulk(
    output_dir: Path,
    reason: str,
    metadata: pd.DataFrame | None = None,
    anchor_gene: str = "RHBDF2",
) -> DonorPseudobulkResult:
    anchor_key = _gene_key(anchor_gene)
    suffix = "" if anchor_key == "rhbdf2" else f"_{anchor_key}"
    metadata = metadata if metadata is not None else pd.DataFrame()
    metadata.to_csv(output_dir / f"donor_pseudobulk_metadata{suffix}.csv", index=False)
    pd.DataFrame([{"status": "not_eligible", "reason": reason, "anchor_gene": anchor_gene}]).to_csv(
        output_dir / f"status{suffix}.csv", index=False
    )
    return DonorPseudobulkResult(metadata, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "not_eligible", reason)


def _extract_backed_gene(adata, gene: str) -> np.ndarray:
    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' was not found in the active matrix.")
    matrix = adata[:, adata.var_names.get_loc(gene)].X
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray()).reshape(-1).astype(float)
    return np.asarray(matrix).reshape(-1).astype(float)


def _adjust_available(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    mask = numeric.notna().to_numpy()
    adjusted = np.full(numeric.shape[0], np.nan)
    if mask.any():
        adjusted[mask] = multipletests(numeric.to_numpy()[mask], method="fdr_bh")[1]
    return adjusted


def _stable_obs_names(obs: pd.DataFrame) -> pd.Index:
    donor = obs["sampleID"].astype(str) if "sampleID" in obs.columns else pd.Series("sample", index=obs.index)
    return pd.Index(donor.to_numpy() + "__" + obs.index.astype(str).to_numpy()).astype(str)


def _gene_key(gene: str) -> str:
    """Return a stable lowercase key for anchor-specific outputs."""
    key = re.sub(r"[^A-Za-z0-9]+", "_", str(gene)).strip("_").lower()
    if not key:
        raise ValueError("Anchor gene must contain at least one alphanumeric character.")
    return key
