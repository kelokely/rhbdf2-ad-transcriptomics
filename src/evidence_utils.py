"""Evidence-layer helpers for candidate genes and metadata associations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def build_candidate_evidence_table(
    candidate_genes: list[str],
    candidate_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    cell_de_results: pd.DataFrame,
    continuous_results: pd.DataFrame | None,
    pseudobulk_de_results: pd.DataFrame | None,
    pseudobulk_candidate_expression: pd.DataFrame | None,
    pseudobulk_paired_candidate_deltas: pd.DataFrame | None,
    candidate_hurdle_summary: pd.DataFrame | None,
    subpopulation_pseudobulk_gene_summary: pd.DataFrame | None,
    protein_summary: pd.DataFrame | None,
    genetics_summary: pd.DataFrame | None,
    brain_context_summary: pd.DataFrame | None,
    proteomics_overlap_summary: pd.DataFrame | None,
    gene_of_interest: str,
    state_column: str,
    high_label: str,
    low_label: str,
    evidence_config: dict[str, object],
    output_path: Path,
) -> pd.DataFrame:
    """Merge multiple evidence layers into one candidate-gene summary table."""
    unique_genes = pd.DataFrame({"gene": list(dict.fromkeys(candidate_genes))})

    candidate_summary = candidate_summary.copy()
    mean_pivot = candidate_summary.pivot(index="gene", columns="group_name", values="mean_expression")
    frac_pivot = candidate_summary.pivot(index="gene", columns="group_name", values="fraction_expressing")
    mean_pivot = mean_pivot.rename(
        columns={
            low_label: f"mean_expression_{low_label}",
            high_label: f"mean_expression_{high_label}",
        }
    )
    frac_pivot = frac_pivot.rename(
        columns={
            low_label: f"fraction_expressing_{low_label}",
            high_label: f"fraction_expressing_{high_label}",
        }
    )
    summary_layer = mean_pivot.join(frac_pivot, how="outer").reset_index()

    correlation_layer = correlations.rename(
        columns={
            "pearson_r": "candidate_pearson_r_with_RHBDF2",
            "spearman_r": "candidate_spearman_r_with_RHBDF2",
        }
    )

    cell_layer = (
        cell_de_results[cell_de_results["names"].isin(unique_genes["gene"])]
        [["names", "scores", "logfoldchanges", "pvals_adj"]]
        .rename(
            columns={
                "names": "gene",
                "scores": "cell_level_score",
                "logfoldchanges": "cell_level_logfoldchange",
                "pvals_adj": "cell_level_padj",
            }
        )
    )

    evidence = (
        unique_genes
        .merge(summary_layer, on="gene", how="left")
        .merge(correlation_layer, on="gene", how="left")
        .merge(cell_layer, on="gene", how="left")
    )

    if continuous_results is not None and not continuous_results.empty:
        continuous_layer = (
            continuous_results[continuous_results["names"].isin(unique_genes["gene"])]
            [["names", "slope_per_log1p_rhbdf2", "pvals_adj"]]
            .rename(
                columns={
                    "names": "gene",
                    "slope_per_log1p_rhbdf2": "continuous_slope_per_log1p_rhbdf2",
                    "pvals_adj": "continuous_padj",
                }
            )
        )
        evidence = evidence.merge(continuous_layer, on="gene", how="left")

    if pseudobulk_de_results is not None:
        pseudobulk_layer = (
            pseudobulk_de_results[pseudobulk_de_results["names"].isin(unique_genes["gene"])]
            [["names", "effect_size_log1p_cpm", "pvals_adj"]]
            .rename(
                columns={
                    "names": "gene",
                    "effect_size_log1p_cpm": "pseudobulk_effect_log1p_cpm",
                    "pvals_adj": "pseudobulk_padj",
                }
            )
        )
        evidence = evidence.merge(pseudobulk_layer, on="gene", how="left")

    if pseudobulk_candidate_expression is not None and pseudobulk_paired_candidate_deltas is not None:
        pseudobulk_support = summarize_pseudobulk_candidate_support(
            candidate_expression=pseudobulk_candidate_expression,
            paired_candidate_deltas=pseudobulk_paired_candidate_deltas,
            gene_of_interest=gene_of_interest,
            candidate_genes=unique_genes["gene"].tolist(),
        )
        evidence = evidence.merge(pseudobulk_support, on="gene", how="left")

    if candidate_hurdle_summary is not None and not candidate_hurdle_summary.empty:
        evidence = evidence.merge(candidate_hurdle_summary, on="gene", how="left")

    if subpopulation_pseudobulk_gene_summary is not None and not subpopulation_pseudobulk_gene_summary.empty:
        evidence = evidence.merge(subpopulation_pseudobulk_gene_summary, on="gene", how="left")

    if protein_summary is not None and not protein_summary.empty:
        protein_columns = [
            "gene",
            "uniprot_found",
            "reactome_pathway_count",
            "annotated_ptm_feature_count",
            "candidate_panel_interaction_count",
            "candidate_panel_partners",
            "candidate_panel_ptm_out_count",
            "candidate_panel_ptm_in_count",
            "known_substrate_relationship_count",
            "known_cleavage_relationship_count",
            "literature_supported_context",
            "curated_literature_pmid_count",
        ]
        protein_layer = protein_summary[[column for column in protein_columns if column in protein_summary.columns]].copy()
        evidence = evidence.merge(protein_layer, on="gene", how="left")

    if genetics_summary is not None and not genetics_summary.empty:
        genetics_columns = [
            "gene",
            "gwas_matching_trait_count",
            "gwas_association_count",
            "gwas_min_p_value",
            "gwas_best_trait",
            "opentargets_matching_disease_count",
            "opentargets_max_score",
            "opentargets_best_disease",
            "combined_genetics_support",
        ]
        genetics_layer = genetics_summary[[column for column in genetics_columns if column in genetics_summary.columns]].copy()
        evidence = evidence.merge(genetics_layer, on="gene", how="left")

    if brain_context_summary is not None and not brain_context_summary.empty:
        evidence = evidence.merge(brain_context_summary, on="gene", how="left")

    if proteomics_overlap_summary is not None and not proteomics_overlap_summary.empty:
        overlap_columns = [
            "gene",
            "cell_level_vs_public_proteomics_direction_match",
            "pseudobulk_vs_public_proteomics_direction_match",
            "continuous_vs_public_proteomics_direction_match",
            "transcript_vs_public_proteomics_direction_match",
            "public_proteomics_overlap_supported",
        ]
        evidence = evidence.merge(
            proteomics_overlap_summary[[column for column in overlap_columns if column in proteomics_overlap_summary.columns]].copy(),
            on="gene",
            how="left",
        )

    evidence["cell_level_significant_fdr_0_05"] = evidence["cell_level_padj"].fillna(1.0) < 0.05
    if "continuous_padj" in evidence.columns:
        evidence["continuous_significant_fdr_0_05"] = evidence["continuous_padj"].fillna(1.0) < 0.05
    if "pseudobulk_padj" in evidence.columns:
        evidence["pseudobulk_significant_fdr_0_05"] = evidence["pseudobulk_padj"].fillna(1.0) < 0.05
    if "detection_padj" in evidence.columns:
        evidence["detection_significant_fdr_0_05"] = evidence["detection_padj"].fillna(1.0) < 0.05
    if "positive_expression_padj" in evidence.columns:
        evidence["positive_expression_significant_fdr_0_05"] = evidence["positive_expression_padj"].fillna(1.0) < 0.05
    if "subpopulation_pseudobulk_n_subgroups_significant" in evidence.columns:
        evidence["subpopulation_supported"] = (
            evidence["subpopulation_pseudobulk_n_subgroups_significant"].fillna(0).astype(float) > 0
        )

    protein_context_supported = (
        _as_bool_series(evidence, "uniprot_found")
        & (
            _positive_numeric_series(evidence, "reactome_pathway_count")
            | _positive_numeric_series(evidence, "annotated_ptm_feature_count")
            | _positive_numeric_series(evidence, "candidate_panel_interaction_count")
            | _positive_numeric_series(evidence, "candidate_panel_ptm_out_count")
            | _positive_numeric_series(evidence, "candidate_panel_ptm_in_count")
        )
    )
    evidence["protein_context_supported"] = protein_context_supported
    evidence["genetics_supported"] = _as_bool_series(evidence, "combined_genetics_support")
    evidence["brain_context_supported"] = (
        _positive_numeric_series(evidence, "hpa_microglia_nCPM")
        | _positive_numeric_series(evidence, "hpa_cns_macrophage_nCPM")
        | _as_bool_series(evidence, "proteomics_gene_detected")
    )
    evidence["public_proteomics_overlap_supported"] = _as_bool_series(
        evidence,
        "public_proteomics_overlap_supported",
    )

    weights = evidence_config.get("module_weights", {}) if isinstance(evidence_config, dict) else {}
    transcript_weight_map = {
        "cell_level": "cell_level_significant_fdr_0_05",
        "pseudobulk": "pseudobulk_significant_fdr_0_05",
        "continuous": "continuous_significant_fdr_0_05",
        "detection": "detection_significant_fdr_0_05",
        "positive_expression": "positive_expression_significant_fdr_0_05",
        "subpopulation_pseudobulk": "subpopulation_supported",
    }
    context_weight_map = {
        "protein_context": "protein_context_supported",
        "genetics": "genetics_supported",
        "brain_context": "brain_context_supported",
        "public_proteomics_overlap": "public_proteomics_overlap_supported",
    }
    evidence["transcript_support_count"] = _count_supports(evidence, transcript_weight_map.values())
    evidence["context_support_count"] = _count_supports(evidence, context_weight_map.values())
    evidence["evidence_support_count"] = evidence["transcript_support_count"] + evidence["context_support_count"]
    evidence["weighted_transcript_evidence_score"] = _weighted_support_score(
        evidence,
        transcript_weight_map,
        weights,
    )
    evidence["weighted_context_evidence_score"] = _weighted_support_score(
        evidence,
        context_weight_map,
        weights,
    )
    evidence["weighted_total_evidence_score"] = (
        evidence["weighted_transcript_evidence_score"] + evidence["weighted_context_evidence_score"]
    )

    evidence.to_csv(output_path, index=False)
    return evidence


def summarize_pseudobulk_candidate_support(
    candidate_expression: pd.DataFrame,
    paired_candidate_deltas: pd.DataFrame,
    gene_of_interest: str,
    candidate_genes: list[str],
) -> pd.DataFrame:
    """Summarize continuous and paired-delta candidate support in pseudobulk space."""
    rows: list[dict[str, object]] = []

    expression_columns = set(candidate_expression.columns)
    paired_group = None
    if not paired_candidate_deltas.empty:
        paired_group = paired_candidate_deltas.groupby("gene", observed=True)

    for gene in candidate_genes:
        pearson_r = np.nan
        spearman_r = np.nan
        n_pseudobulk_samples = 0
        if gene in expression_columns and gene_of_interest in expression_columns:
            if gene == gene_of_interest:
                series = candidate_expression[gene_of_interest].dropna()
                n_pseudobulk_samples = int(len(series))
                if n_pseudobulk_samples >= 2 and series.nunique() > 1:
                    pearson_r = 1.0
                    spearman_r = 1.0
            else:
                subset = candidate_expression[[gene_of_interest, gene]].dropna()
                n_pseudobulk_samples = int(len(subset))
                if n_pseudobulk_samples >= 2 and subset[gene_of_interest].nunique() > 1 and subset[gene].nunique() > 1:
                    pearson_r = float(subset[gene_of_interest].corr(subset[gene], method="pearson"))
                    spearman_r = float(subset[gene_of_interest].corr(subset[gene], method="spearman"))

        mean_delta = np.nan
        median_delta = np.nan
        fraction_positive = np.nan
        n_paired_donors = 0
        if paired_group is not None and gene in paired_group.groups:
            gene_deltas = paired_group.get_group(gene)["delta_high_minus_low"].dropna()
            n_paired_donors = int(len(gene_deltas))
            if n_paired_donors:
                mean_delta = float(gene_deltas.mean())
                median_delta = float(gene_deltas.median())
                fraction_positive = float((gene_deltas > 0).mean())

        rows.append(
            {
                "gene": gene,
                "pseudobulk_pearson_r_with_RHBDF2": pearson_r,
                "pseudobulk_spearman_r_with_RHBDF2": spearman_r,
                "pseudobulk_n_samples_for_correlation": n_pseudobulk_samples,
                "paired_delta_mean_high_minus_low": mean_delta,
                "paired_delta_median_high_minus_low": median_delta,
                "paired_delta_fraction_positive": fraction_positive,
                "paired_delta_n_donors": n_paired_donors,
            }
        )

    return pd.DataFrame(rows)


def summarize_state_metadata_associations(
    adata,
    state_column: str,
    metadata_columns: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    """Compute contingency summaries between the RHBDF2 state and key metadata columns."""
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_columns = [column for column in dict.fromkeys(metadata_columns) if column in adata.obs.columns]
    rows: list[dict[str, object]] = []
    written_paths: list[Path] = []

    for column in unique_columns:
        counts = pd.crosstab(adata.obs[column], adata.obs[state_column])
        counts_path = output_dir / f"state_association_counts_{column}.csv"
        row_pct_path = output_dir / f"state_association_row_pct_{column}.csv"
        counts.to_csv(counts_path)
        counts.div(counts.sum(axis=1), axis=0).to_csv(row_pct_path)
        written_paths.extend([counts_path, row_pct_path])

        if counts.shape[0] > 1 and counts.shape[1] > 1:
            chi2, p_value, _, _ = chi2_contingency(counts)
            n = counts.to_numpy().sum()
            cramers_v = float(np.sqrt(chi2 / (n * min(counts.shape[0] - 1, counts.shape[1] - 1))))
        else:
            p_value = np.nan
            cramers_v = np.nan

        rows.append(
            {
                "metadata_column": column,
                "n_levels": int(counts.shape[0]),
                "n_cells": int(counts.to_numpy().sum()),
                "chi_square_p_value": p_value,
                "cramers_v": cramers_v,
            }
        )

    summary = pd.DataFrame(rows).sort_values("chi_square_p_value", na_position="last").reset_index(drop=True)
    summary_path = output_dir / "state_metadata_association_summary.csv"
    summary.to_csv(summary_path, index=False)
    written_paths.append(summary_path)
    return summary, written_paths


def _count_supports(evidence: pd.DataFrame, columns: object) -> pd.Series:
    total = pd.Series(0, index=evidence.index, dtype=int)
    for column in columns:
        if column not in evidence.columns:
            continue
        total = total + evidence[column].fillna(False).astype(bool).astype(int)
    return total


def _weighted_support_score(
    evidence: pd.DataFrame,
    support_map: dict[str, str],
    weights: dict[str, object],
) -> pd.Series:
    score = pd.Series(0.0, index=evidence.index, dtype=float)
    for module_name, column_name in support_map.items():
        if column_name not in evidence.columns:
            continue
        weight = float(weights.get(module_name, 0.0))
        score = score + (evidence[column_name].fillna(False).astype(bool).astype(float) * weight)
    return score


def _as_bool_series(evidence: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in evidence.columns:
        return pd.Series(False, index=evidence.index, dtype=bool)
    values = evidence[column_name]
    return values.where(values.notna(), False).astype(bool)


def _positive_numeric_series(evidence: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in evidence.columns:
        return pd.Series(False, index=evidence.index, dtype=bool)
    return pd.to_numeric(evidence[column_name], errors="coerce").fillna(0.0) > 0
