"""Helpers for comparing transcript-level signals against public brain proteomics context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PublicProteomicsOverlapResult:
    """Candidate-level overlap between transcript results and public proteomics context."""

    summary: pd.DataFrame
    summary_path: Path
    top_concordant_genes: list[str]


def summarize_public_proteomics_overlap(
    candidate_genes: list[str],
    cell_de_results: pd.DataFrame,
    pseudobulk_de_results: pd.DataFrame | None,
    continuous_results: pd.DataFrame | None,
    brain_context_summary: pd.DataFrame | None,
    output_dir: Path,
) -> PublicProteomicsOverlapResult | None:
    """Summarize direction concordance between transcript layers and public proteomics evidence."""
    if brain_context_summary is None or brain_context_summary.empty:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    unique_genes = pd.DataFrame({"gene": list(dict.fromkeys(candidate_genes))})

    summary = unique_genes.copy()
    summary = summary.merge(
        _compact_cell_layer(cell_de_results),
        on="gene",
        how="left",
    )
    if pseudobulk_de_results is not None and not pseudobulk_de_results.empty:
        summary = summary.merge(
            _compact_pseudobulk_layer(pseudobulk_de_results),
            on="gene",
            how="left",
        )
    if continuous_results is not None and not continuous_results.empty:
        summary = summary.merge(
            _compact_continuous_layer(continuous_results),
            on="gene",
            how="left",
        )

    proteomics_columns = [
        "gene",
        "proteomics_context_label",
        "proteomics_case_labels",
        "proteomics_control_labels",
        "proteomics_gene_detected",
        "proteomics_regions_profiled",
        "proteomics_regions_with_matched_case_control",
        "proteomics_mean_log2_fc_case_vs_control",
        "proteomics_median_log2_fc_case_vs_control",
        "proteomics_fraction_regions_higher_in_case",
        "proteomics_strongest_case_region",
    ]
    summary = summary.merge(
        brain_context_summary[[column for column in proteomics_columns if column in brain_context_summary.columns]].copy(),
        on="gene",
        how="left",
    )
    proteomics_direction = _numeric_series(summary, "proteomics_mean_log2_fc_case_vs_control")

    summary["cell_level_vs_public_proteomics_direction_match"] = _direction_match(
        summary.get("cell_level_logfoldchange"),
        proteomics_direction,
    )
    summary["pseudobulk_vs_public_proteomics_direction_match"] = _direction_match(
        summary.get("pseudobulk_effect_log1p_cpm"),
        proteomics_direction,
    )
    summary["continuous_vs_public_proteomics_direction_match"] = _direction_match(
        summary.get("continuous_slope_per_log1p_rhbdf2"),
        proteomics_direction,
    )

    summary["best_transcript_effect"] = _best_available_effect(summary)
    summary["transcript_vs_public_proteomics_direction_match"] = _direction_match(
        summary.get("best_transcript_effect"),
        proteomics_direction,
    )
    summary["public_proteomics_overlap_supported"] = (
        summary.get("proteomics_gene_detected", pd.Series(False, index=summary.index)).fillna(False).astype(bool)
        & summary["transcript_vs_public_proteomics_direction_match"].fillna(False).astype(bool)
    )

    summary_path = output_dir / "public_proteomics_overlap_summary.csv"
    summary.to_csv(summary_path, index=False)

    sortable = summary[
        summary["public_proteomics_overlap_supported"].fillna(False).astype(bool)
    ].sort_values(
        ["proteomics_regions_with_matched_case_control", "gene"],
        ascending=[False, True],
    )
    top_concordant = [
        f"{row.gene} (proteomics_log2_fc={float(row.proteomics_mean_log2_fc_case_vs_control):.2f})"
        for row in sortable.head(5).itertuples(index=False)
        if pd.notna(row.proteomics_mean_log2_fc_case_vs_control)
    ]

    return PublicProteomicsOverlapResult(
        summary=summary,
        summary_path=summary_path,
        top_concordant_genes=top_concordant,
    )


def _compact_cell_layer(cell_de_results: pd.DataFrame) -> pd.DataFrame:
    return cell_de_results[["names", "logfoldchanges", "pvals_adj"]].rename(
        columns={
            "names": "gene",
            "logfoldchanges": "cell_level_logfoldchange",
            "pvals_adj": "cell_level_padj",
        }
    )


def _compact_pseudobulk_layer(pseudobulk_de_results: pd.DataFrame) -> pd.DataFrame:
    return pseudobulk_de_results[["names", "effect_size_log1p_cpm", "pvals_adj"]].rename(
        columns={
            "names": "gene",
            "effect_size_log1p_cpm": "pseudobulk_effect_log1p_cpm",
            "pvals_adj": "pseudobulk_padj",
        }
    )


def _compact_continuous_layer(continuous_results: pd.DataFrame) -> pd.DataFrame:
    return continuous_results[["names", "slope_per_log1p_rhbdf2", "pvals_adj"]].rename(
        columns={
            "names": "gene",
            "slope_per_log1p_rhbdf2": "continuous_slope_per_log1p_rhbdf2",
            "pvals_adj": "continuous_padj",
        }
    )


def _direction_match(left, right) -> pd.Series:
    left_values = pd.to_numeric(left, errors="coerce") if left is not None else pd.Series(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce") if right is not None else pd.Series(dtype=float)
    if left_values.empty or right_values.empty:
        return pd.Series(dtype=bool)
    left_sign = np.sign(left_values.to_numpy(dtype=float))
    right_sign = np.sign(right_values.to_numpy(dtype=float))
    valid = np.isfinite(left_sign) & np.isfinite(right_sign) & (left_sign != 0) & (right_sign != 0)
    matched = np.zeros(len(left_values), dtype=bool)
    matched[valid] = left_sign[valid] == right_sign[valid]
    return pd.Series(matched, index=left_values.index)


def _best_available_effect(summary: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=summary.index, dtype=float)
    for column in [
        "pseudobulk_effect_log1p_cpm",
        "continuous_slope_per_log1p_rhbdf2",
        "cell_level_logfoldchange",
    ]:
        if column not in summary.columns:
            continue
        values = pd.to_numeric(summary[column], errors="coerce")
        result = result.where(result.notna(), values)
    return result


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")
