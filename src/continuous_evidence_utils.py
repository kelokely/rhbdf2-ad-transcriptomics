"""Threshold-free continuous and hurdle-style evidence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as student_t

try:
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
except ImportError:
    sm = None
    multipletests = None
    PerfectSeparationWarning = Warning

from .gene_utils import check_gene_presence, extract_expression_vector


@dataclass
class ContinuousEvidenceResult:
    """Continuous association and candidate hurdle outputs."""

    full_results: pd.DataFrame
    positive_results: pd.DataFrame
    negative_results: pd.DataFrame
    candidate_hurdle_summary: pd.DataFrame
    full_results_path: Path
    positive_path: Path
    negative_path: Path
    candidate_hurdle_path: Path


def run_continuous_evidence(
    adata,
    gene_of_interest: str,
    candidate_genes: list[str],
    continuous_config: dict[str, object],
    output_dir: Path,
    fallback_donor_column: str | None,
) -> ContinuousEvidenceResult | None:
    """Run threshold-free transcriptome-wide and candidate hurdle models."""
    if not continuous_config.get("enabled", True):
        return None
    if sm is None or multipletests is None:
        raise ImportError(
            "statsmodels is required for continuous and hurdle-style modeling. "
            "Install the full dependency set to enable this optional layer."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    top_n_export = int(continuous_config.get("top_n_export", 100))
    min_positive_cells = int(continuous_config.get("min_positive_cells", 25))
    donor_column = continuous_config.get("donor_column") or fallback_donor_column
    anchor_key = _anchor_key(gene_of_interest)
    slope_column = f"slope_per_log1p_{anchor_key}"

    full_results = _run_transcriptome_wide_linear_association(adata=adata, gene_of_interest=gene_of_interest)
    positive_results = (
        full_results[full_results[slope_column] > 0]
        .sort_values([slope_column, "pvals_adj"], ascending=[False, True], na_position="last")
        .head(top_n_export)
        .reset_index(drop=True)
    )
    negative_results = (
        full_results[full_results[slope_column] < 0]
        .sort_values([slope_column, "pvals_adj"], ascending=[True, True], na_position="last")
        .head(top_n_export)
        .reset_index(drop=True)
    )

    suffix = "" if anchor_key == "rhbdf2" else f"_{anchor_key}"
    full_results_path = output_dir / f"continuous_association_with_{anchor_key}.csv"
    positive_path = output_dir / f"continuous_top_positive_genes{suffix}.csv"
    negative_path = output_dir / f"continuous_top_negative_genes{suffix}.csv"
    full_results.to_csv(full_results_path, index=False)
    positive_results.to_csv(positive_path, index=False)
    negative_results.to_csv(negative_path, index=False)

    candidate_hurdle_summary = _run_candidate_hurdle_models(
        adata=adata,
        gene_of_interest=gene_of_interest,
        candidate_genes=candidate_genes,
        donor_column=str(donor_column) if donor_column else None,
        min_positive_cells=min_positive_cells,
    )
    candidate_hurdle_path = output_dir / f"candidate_hurdle_summary{suffix}.csv"
    candidate_hurdle_summary.to_csv(candidate_hurdle_path, index=False)

    return ContinuousEvidenceResult(
        full_results=full_results,
        positive_results=positive_results,
        negative_results=negative_results,
        candidate_hurdle_summary=candidate_hurdle_summary,
        full_results_path=full_results_path,
        positive_path=positive_path,
        negative_path=negative_path,
        candidate_hurdle_path=candidate_hurdle_path,
    )


def _run_transcriptome_wide_linear_association(adata, gene_of_interest: str) -> pd.DataFrame:
    """Fit a fast transcriptome-wide linear association against a continuous anchor."""
    anchor_key = _anchor_key(gene_of_interest)
    slope_column = f"slope_per_log1p_{anchor_key}"
    x = extract_expression_vector(adata, gene_of_interest).astype(float)
    n_obs = int(x.shape[0])
    x_mean = float(np.mean(x))
    x_centered = x - x_mean
    ss_x = float(np.sum(np.square(x_centered)))
    if ss_x <= 0:
        raise ValueError(f"Gene of interest '{gene_of_interest}' has no variance after preprocessing.")

    matrix = adata.X
    if sparse.issparse(matrix):
        y_mean = np.asarray(matrix.mean(axis=0)).reshape(-1)
        y_sum_sq = np.asarray(matrix.power(2).sum(axis=0)).reshape(-1)
        numerator = np.asarray(matrix.T.dot(x_centered)).reshape(-1)
    else:
        values = np.asarray(matrix, dtype=float)
        y_mean = values.mean(axis=0)
        y_sum_sq = np.square(values).sum(axis=0)
        numerator = values.T @ x_centered

    ss_y = y_sum_sq - (n_obs * np.square(y_mean))
    slope = np.divide(
        numerator,
        ss_x,
        out=np.zeros_like(numerator, dtype=float),
        where=ss_x > 0,
    )

    denominator = np.sqrt(ss_x * ss_y)
    pearson_r = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
    pearson_r = np.clip(pearson_r, -0.999999999999, 0.999999999999)

    degrees_freedom = max(n_obs - 2, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistic = pearson_r * np.sqrt(degrees_freedom / (1.0 - np.square(pearson_r)))
    pvals = 2.0 * student_t.sf(np.abs(t_statistic), df=degrees_freedom)
    pvals = np.where(np.isfinite(pvals), pvals, 1.0)
    pvals_adj = multipletests(pvals, method="fdr_bh")[1]

    intercept = y_mean - (slope * x_mean)
    results = pd.DataFrame(
        {
            "names": adata.var_names.astype(str).tolist(),
            "anchor_gene": gene_of_interest,
            slope_column: slope.astype(float),
            "intercept": intercept.astype(float),
            "pearson_r": pearson_r.astype(float),
            "t_statistic": np.where(np.isfinite(t_statistic), t_statistic, 0.0).astype(float),
            "pvals": pvals.astype(float),
            "pvals_adj": pvals_adj.astype(float),
            "mean_expression": y_mean.astype(float),
        }
    )
    return results.sort_values(
        [slope_column, "pvals_adj"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)


def _run_candidate_hurdle_models(
    adata,
    gene_of_interest: str,
    candidate_genes: list[str],
    donor_column: str | None,
    min_positive_cells: int,
) -> pd.DataFrame:
    """Fit detection and positive-expression models for the candidate panel."""
    anchor_key = _anchor_key(gene_of_interest)
    predictor_column = f"{anchor_key}_expression"
    detection_coefficient = f"detection_coef_per_log1p_{anchor_key}"
    positive_coefficient = f"positive_expression_coef_per_log1p_{anchor_key}"
    x = extract_expression_vector(adata, gene_of_interest).astype(float)
    design = pd.DataFrame({"intercept": np.ones(adata.n_obs, dtype=float), predictor_column: x})
    donor_groups: np.ndarray | None = None
    donor_adjustment_method = "none"
    if donor_column and donor_column in adata.obs.columns:
        donors = adata.obs[donor_column].astype(str).to_numpy()
        if adata.n_obs >= 100_000 and np.unique(donors).size >= 2:
            donor_groups = donors
            donor_adjustment_method = "cluster_robust_by_donor"
        else:
            donor_dummies = pd.get_dummies(
                donors,
                prefix="donor",
                drop_first=True,
                dtype=float,
            )
            if not donor_dummies.empty:
                design = pd.concat([design.reset_index(drop=True), donor_dummies.reset_index(drop=True)], axis=1)
                donor_adjustment_method = "donor_fixed_effects"

    rows: list[dict[str, object]] = []
    unique_genes = [gene for gene in dict.fromkeys(candidate_genes) if check_gene_presence(adata, gene)]
    for gene in unique_genes:
        y = extract_expression_vector(adata, gene).astype(float)
        detection = (y > 0).astype(float)
        row = {
            "gene": gene,
            "anchor_gene": gene_of_interest,
            "n_cells": int(y.shape[0]),
            "n_positive_cells": int(np.sum(detection)),
            detection_coefficient: np.nan,
            "detection_pval": np.nan,
            positive_coefficient: np.nan,
            "positive_expression_pval": np.nan,
            "donor_adjustment_method": donor_adjustment_method,
            "note": "",
        }

        detection_note = _fit_detection_component(
            row=row,
            design=design,
            detection=detection,
            gene=gene,
            gene_of_interest=gene_of_interest,
            predictor_column=predictor_column,
            coefficient_column=detection_coefficient,
            donor_groups=donor_groups,
        )
        positive_note = _fit_positive_component(
            row=row,
            design=design,
            expression=y,
            detection=detection,
            min_positive_cells=min_positive_cells,
            gene=gene,
            gene_of_interest=gene_of_interest,
            predictor_column=predictor_column,
            coefficient_column=positive_coefficient,
            donor_groups=donor_groups,
        )
        row["note"] = "; ".join(note for note in [detection_note, positive_note] if note)
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["detection_padj"] = _adjust_pvalues(summary["detection_pval"])
        summary["positive_expression_padj"] = _adjust_pvalues(summary["positive_expression_pval"])
    return summary.sort_values("gene").reset_index(drop=True)


def _fit_detection_component(
    row: dict[str, object],
    design: pd.DataFrame,
    detection: np.ndarray,
    gene: str,
    gene_of_interest: str,
    predictor_column: str,
    coefficient_column: str,
    donor_groups: np.ndarray | None = None,
) -> str:
    if gene == gene_of_interest:
        return "self-gene detection model skipped"
    if np.unique(detection).size < 2:
        return "detection model skipped because the gene had no detection variance"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PerfectSeparationWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fit_kwargs: dict[str, object] = {"disp": False}
            if donor_groups is not None:
                fit_kwargs.update(
                    {"cov_type": "cluster", "cov_kwds": {"groups": donor_groups}}
                )
            model = sm.GLM(detection, design, family=sm.families.Binomial()).fit(**fit_kwargs)
        row[coefficient_column] = float(model.params[predictor_column])
        row["detection_pval"] = float(model.pvalues[predictor_column])
        return ""
    except Exception:
        return "detection model failed or encountered separation"


def _fit_positive_component(
    row: dict[str, object],
    design: pd.DataFrame,
    expression: np.ndarray,
    detection: np.ndarray,
    min_positive_cells: int,
    gene: str,
    gene_of_interest: str,
    predictor_column: str,
    coefficient_column: str,
    donor_groups: np.ndarray | None = None,
) -> str:
    positive_mask = detection > 0
    if gene == gene_of_interest:
        return "self-gene positive-expression model skipped"
    if int(np.sum(positive_mask)) < min_positive_cells:
        return f"positive-expression model skipped because fewer than {min_positive_cells} cells expressed the gene"

    y_positive = expression[positive_mask]
    if np.unique(y_positive).size < 2:
        return "positive-expression model skipped because positive cells had no variance"

    try:
        model = sm.OLS(y_positive, design.loc[positive_mask].reset_index(drop=True))
        if donor_groups is not None:
            model = model.fit(cov_type="cluster", cov_kwds={"groups": donor_groups[positive_mask]})
        else:
            model = model.fit()
        row[coefficient_column] = float(model.params[predictor_column])
        row["positive_expression_pval"] = float(model.pvalues[predictor_column])
        return ""
    except Exception:
        return "positive-expression model failed"


def _adjust_pvalues(values: pd.Series) -> np.ndarray:
    mask = values.notna().to_numpy()
    adjusted = np.full(values.shape[0], np.nan, dtype=float)
    if np.any(mask):
        adjusted[mask] = multipletests(values[mask].to_numpy(dtype=float), method="fdr_bh")[1]
    return adjusted


def _anchor_key(gene: str) -> str:
    """Return a stable lowercase key for anchor-specific columns and files."""
    key = re.sub(r"[^A-Za-z0-9]+", "_", str(gene)).strip("_").lower()
    if not key:
        raise ValueError("Anchor gene must contain at least one alphanumeric character.")
    return key
