"""Transcript-based pathway activity proxy scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, ttest_1samp

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None

from .gene_utils import extract_expression_vector


DEFAULT_PROXY_GENE_SETS: dict[str, list[str]] = {
    "trem2_tyrobp_syk_response": ["TREM2", "TYROBP", "SYK", "ITGAX", "CLEC7A", "CST7", "LPL", "FCER1G"],
    "inflammatory_nfkb": ["TNF", "IL6R", "RELA", "NFKBIA", "IL1B", "JUN", "FOS", "CCL2"],
    "phagocytosis_myeloid": ["TREM2", "TYROBP", "ITGAX", "CLEC7A", "LPL", "CST7", "CTSB", "AXL"],
    "lysosomal_activation": ["CST7", "LPL", "CTSB", "CTSD", "LAMP1", "LAMP2", "PSAP", "GPNMB"],
    "sting_tbk1_irf3_response": ["TMEM173", "TBK1", "IRF3", "ISG15", "IFIT1", "OAS1", "IFI27", "MX1"],
}


@dataclass
class PathwayActivityProxyResult:
    """Outputs for transcript-based pathway activity proxy scoring."""

    summary: pd.DataFrame
    cell_scores: pd.DataFrame
    summary_path: Path
    cell_scores_path: Path
    gene_sets_path: Path
    top_proxy_modules: list[str]


def run_pathway_activity_proxies(
    adata,
    gene_of_interest: str,
    state_column: str,
    high_label: str,
    low_label: str,
    proxy_config: dict[str, Any],
    output_dir: Path,
    fallback_donor_column: str | None,
) -> PathwayActivityProxyResult | None:
    """Score predefined transcript modules as pathway-activity proxies."""
    if not proxy_config.get("enabled", True):
        return None

    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "scanpy is required for transcript-based pathway activity proxies."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    donor_column = proxy_config.get("donor_column") or fallback_donor_column
    min_genes_present = int(proxy_config.get("min_genes_present", 2))
    gene_sets = proxy_config.get("gene_sets") or DEFAULT_PROXY_GENE_SETS
    if not isinstance(gene_sets, dict) or not gene_sets:
        raise ValueError("'pathway_activity_proxy.gene_sets' must be a non-empty mapping.")

    work_adata = adata.copy()
    rhbdf2_expression = extract_expression_vector(work_adata, gene_of_interest).astype(float)
    state_series = work_adata.obs[state_column].astype(str)
    high_mask = state_series == high_label
    low_mask = state_series == low_label

    rows: list[dict[str, object]] = []
    score_columns: list[str] = []
    gene_set_export: dict[str, dict[str, object]] = {}

    for module_name, requested_genes in gene_sets.items():
        unique_requested = [str(gene) for gene in dict.fromkeys(requested_genes)]
        present_genes = [gene for gene in unique_requested if gene in work_adata.var_names]
        missing_genes = [gene for gene in unique_requested if gene not in work_adata.var_names]
        score_column = f"{module_name}_score"
        note = ""

        if len(present_genes) < min_genes_present:
            scores = pd.Series(np.nan, index=work_adata.obs_names, dtype=float)
            note = f"Skipped because fewer than {min_genes_present} genes were present in the matrix."
        else:
            sc.tl.score_genes(
                work_adata,
                gene_list=present_genes,
                score_name=score_column,
                use_raw=False,
            )
            score_columns.append(score_column)
            scores = pd.Series(work_adata.obs[score_column].astype(float).to_numpy(), index=work_adata.obs_names)

        cell_pval = np.nan
        high_mean = np.nan
        low_mean = np.nan
        high_median = np.nan
        low_median = np.nan
        delta_high_minus_low = np.nan
        pearson_r_value = np.nan
        spearman_r_value = np.nan
        donor_paired_pval = np.nan
        donor_paired_mean_delta = np.nan
        donor_paired_median_delta = np.nan
        donor_fraction_positive = np.nan
        donor_paired_n = 0

        if scores.notna().any():
            high_scores = scores.loc[high_mask.to_numpy()]
            low_scores = scores.loc[low_mask.to_numpy()]
            if not high_scores.empty and not low_scores.empty:
                high_mean = float(high_scores.mean())
                low_mean = float(low_scores.mean())
                high_median = float(high_scores.median())
                low_median = float(low_scores.median())
                delta_high_minus_low = high_mean - low_mean
                try:
                    _, cell_pval = mannwhitneyu(
                        high_scores.to_numpy(dtype=float),
                        low_scores.to_numpy(dtype=float),
                        alternative="two-sided",
                    )
                except ValueError:
                    cell_pval = np.nan

            valid_corr = scores.notna().to_numpy() & np.isfinite(rhbdf2_expression)
            if int(np.sum(valid_corr)) >= 3:
                corr_scores = scores.to_numpy(dtype=float)[valid_corr]
                corr_anchor = rhbdf2_expression[valid_corr]
                if np.std(corr_scores) > 0 and np.std(corr_anchor) > 0:
                    pearson_r_value = float(pearsonr(corr_scores, corr_anchor).statistic)
                    spearman_r_value = float(spearmanr(corr_scores, corr_anchor).statistic)

            if donor_column and donor_column in work_adata.obs.columns:
                donor_frame = pd.DataFrame(
                    {
                        "donor_id": work_adata.obs[donor_column].astype(str).to_numpy(),
                        "state": state_series.to_numpy(),
                        "score": scores.to_numpy(dtype=float),
                    },
                    index=work_adata.obs_names,
                ).dropna(subset=["score"])
                if not donor_frame.empty:
                    donor_means = donor_frame.groupby(["donor_id", "state"], observed=True)["score"].mean().reset_index()
                    pivot = donor_means.pivot(index="donor_id", columns="state", values="score")
                    if high_label in pivot.columns and low_label in pivot.columns:
                        deltas = (pivot[high_label] - pivot[low_label]).dropna()
                        donor_paired_n = int(len(deltas))
                        if donor_paired_n:
                            donor_paired_mean_delta = float(deltas.mean())
                            donor_paired_median_delta = float(deltas.median())
                            donor_fraction_positive = float((deltas > 0).mean())
                            if donor_paired_n >= 2 and np.std(deltas.to_numpy(dtype=float)) > 0:
                                donor_paired_pval = float(ttest_1samp(deltas.to_numpy(dtype=float), popmean=0.0).pvalue)

        rows.append(
            {
                "module_name": module_name,
                "requested_genes": ";".join(unique_requested),
                "present_genes": ";".join(present_genes),
                "missing_genes": ";".join(missing_genes),
                "n_genes_requested": len(unique_requested),
                "n_genes_present": len(present_genes),
                "cell_level_high_mean_score": high_mean,
                "cell_level_low_mean_score": low_mean,
                "cell_level_high_median_score": high_median,
                "cell_level_low_median_score": low_median,
                "cell_level_delta_high_minus_low": delta_high_minus_low,
                "cell_level_pval": cell_pval,
                "pearson_r_with_RHBDF2": pearson_r_value,
                "spearman_r_with_RHBDF2": spearman_r_value,
                "donor_paired_n": donor_paired_n,
                "donor_paired_mean_delta": donor_paired_mean_delta,
                "donor_paired_median_delta": donor_paired_median_delta,
                "donor_paired_fraction_positive": donor_fraction_positive,
                "donor_paired_pval": donor_paired_pval,
                "note": note,
            }
        )
        gene_set_export[module_name] = {
            "requested_genes": unique_requested,
            "present_genes": present_genes,
            "missing_genes": missing_genes,
        }

    summary = pd.DataFrame(rows)
    if not summary.empty and multipletests is not None:
        summary["cell_level_padj"] = _adjust_pvalues(summary["cell_level_pval"])
        summary["donor_paired_padj"] = _adjust_pvalues(summary["donor_paired_pval"])
    else:
        summary["cell_level_padj"] = np.nan
        summary["donor_paired_padj"] = np.nan

    cell_scores = pd.DataFrame(index=work_adata.obs_names)
    cell_scores["state"] = state_series.to_numpy()
    if donor_column and donor_column in work_adata.obs.columns:
        cell_scores["donor_id"] = work_adata.obs[donor_column].astype(str).to_numpy()
    for score_column in score_columns:
        cell_scores[score_column] = work_adata.obs[score_column].astype(float).to_numpy()
    cell_scores = cell_scores.reset_index(names="cell_id")

    summary_path = output_dir / "pathway_activity_proxy_summary.csv"
    cell_scores_path = output_dir / "pathway_activity_proxy_cell_scores.csv"
    gene_sets_path = output_dir / "pathway_activity_proxy_gene_sets.yaml"
    summary.to_csv(summary_path, index=False)
    cell_scores.to_csv(cell_scores_path, index=False)
    gene_sets_path.write_text(yaml.safe_dump(gene_set_export, sort_keys=False), encoding="utf-8")

    sortable = summary.sort_values(
        ["donor_paired_padj", "cell_level_padj", "module_name"],
        ascending=[True, True, True],
        na_position="last",
    )
    top_modules = [
        f"{row.module_name} (delta={float(row.cell_level_delta_high_minus_low):.3f})"
        for row in sortable.head(5).itertuples(index=False)
        if pd.notna(row.cell_level_delta_high_minus_low)
    ]

    return PathwayActivityProxyResult(
        summary=summary,
        cell_scores=cell_scores,
        summary_path=summary_path,
        cell_scores_path=cell_scores_path,
        gene_sets_path=gene_sets_path,
        top_proxy_modules=top_modules,
    )


def _adjust_pvalues(values: pd.Series) -> np.ndarray:
    adjusted = np.full(values.shape[0], np.nan, dtype=float)
    mask = values.notna().to_numpy()
    if np.any(mask):
        adjusted[mask] = multipletests(values[mask].to_numpy(dtype=float), method="fdr_bh")[1]
    return adjusted
