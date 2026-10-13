"""Helpers for merging protein-context evidence across UniProt/PTM and public proteomics layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ProteinContextSummaryResult:
    """Merged protein-context evidence summary."""

    summary: pd.DataFrame
    summary_path: Path
    top_supported_genes: list[str]


def build_protein_context_summary(
    candidate_genes: list[str],
    protein_summary: pd.DataFrame | None,
    brain_context_summary: pd.DataFrame | None,
    proteomics_overlap_summary: pd.DataFrame | None,
    output_dir: Path,
) -> ProteinContextSummaryResult | None:
    """Merge protein/PTM, literature, and public proteomics context into one table."""
    if protein_summary is None and brain_context_summary is None and proteomics_overlap_summary is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame({"gene": list(dict.fromkeys(candidate_genes))})

    def _bool_series(column: str) -> pd.Series:
        values = summary.get(column)
        if values is None:
            return pd.Series(False, index=summary.index, dtype=bool)
        return values.fillna(False).astype(bool)

    def _numeric_series(column: str) -> pd.Series:
        values = summary.get(column)
        if values is None:
            return pd.Series(0.0, index=summary.index, dtype=float)
        return pd.to_numeric(values, errors="coerce").fillna(0.0)

    if protein_summary is not None and not protein_summary.empty:
        protein_columns = [
            "gene",
            "uniprot_found",
            "protein_name",
            "annotation_score",
            "reactome_pathway_count",
            "annotated_ptm_feature_count",
            "candidate_panel_interaction_count",
            "candidate_panel_ptm_out_count",
            "candidate_panel_ptm_in_count",
            "known_substrate_relationship_count",
            "known_cleavage_relationship_count",
            "literature_supported_context",
            "curated_literature_pmid_count",
            "function_literature_pmid_count",
            "disease_literature_pmid_count",
            "ptm_literature_pmid_count",
        ]
        summary = summary.merge(
            protein_summary[[column for column in protein_columns if column in protein_summary.columns]].copy(),
            on="gene",
            how="left",
        )

    if brain_context_summary is not None and not brain_context_summary.empty:
        brain_columns = [
            "gene",
            "hpa_microglia_nCPM",
            "hpa_cns_macrophage_nCPM",
            "proteomics_context_label",
            "proteomics_case_labels",
            "proteomics_control_labels",
            "proteomics_gene_detected",
            "proteomics_mean_log2_fc_case_vs_control",
            "proteomics_fraction_regions_higher_in_case",
            "proteomics_strongest_case_region",
        ]
        summary = summary.merge(
            brain_context_summary[[column for column in brain_columns if column in brain_context_summary.columns]].copy(),
            on="gene",
            how="left",
        )

    if proteomics_overlap_summary is not None and not proteomics_overlap_summary.empty:
        overlap_columns = [
            "gene",
            "transcript_vs_public_proteomics_direction_match",
            "pseudobulk_vs_public_proteomics_direction_match",
            "continuous_vs_public_proteomics_direction_match",
            "public_proteomics_overlap_supported",
        ]
        summary = summary.merge(
            proteomics_overlap_summary[[column for column in overlap_columns if column in proteomics_overlap_summary.columns]].copy(),
            on="gene",
            how="left",
        )

    summary["protein_context_evidence_supported"] = _bool_series("uniprot_found") & (
        (_numeric_series("annotated_ptm_feature_count") > 0)
        | (_numeric_series("candidate_panel_interaction_count") > 0)
        | (_numeric_series("known_substrate_relationship_count") > 0)
        | _bool_series("public_proteomics_overlap_supported")
    )
    if "public_proteomics_overlap_supported" not in summary.columns:
        summary["public_proteomics_overlap_supported"] = False

    summary_path = output_dir / "protein_context_evidence_summary.csv"
    summary.to_csv(summary_path, index=False)

    sortable = summary.sort_values(
        [
            "protein_context_evidence_supported",
            "public_proteomics_overlap_supported",
            "curated_literature_pmid_count",
            "annotated_ptm_feature_count",
            "gene",
        ],
        ascending=[False, False, False, False, True],
        na_position="last",
    )
    top_supported = [
        f"{row.gene} (pmids={int(getattr(row, 'curated_literature_pmid_count', 0) or 0)})"
        for row in sortable.head(5).itertuples(index=False)
    ]

    return ProteinContextSummaryResult(
        summary=summary,
        summary_path=summary_path,
        top_supported_genes=top_supported,
    )
