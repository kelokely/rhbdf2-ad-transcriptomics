"""Markdown report generation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .brain_context_utils import BrainContextEvidenceResult
from .activity_proxy_utils import PathwayActivityProxyResult
from .continuous_evidence_utils import ContinuousEvidenceResult
from .enrich_utils import EnrichmentBundleResult
from .genetics_evidence_utils import GeneticsEvidenceResult
from .pseudobulk_utils import PseudobulkResult, SubpopulationPseudobulkResult
from .protein_context_utils import ProteinContextSummaryResult
from .protein_evidence_utils import ProteinEvidenceResult
from .public_proteomics_overlap_utils import PublicProteomicsOverlapResult
from .state_utils import StateAssignmentResult


def generate_run_report(
    config: dict[str, object],
    run_dir: Path,
    report_copy_path: Path,
    config_snapshot_path: Path,
    dataset_summary: dict[str, object],
    metadata_validation: pd.DataFrame,
    metadata_suggestions: pd.DataFrame,
    gene_presence: pd.DataFrame,
    state_result: StateAssignmentResult,
    state_summary: pd.DataFrame,
    preprocessing_summary: dict[str, object],
    de_results: pd.DataFrame,
    pseudobulk_result: PseudobulkResult | None,
    gprofiler_result: EnrichmentBundleResult | None,
    pseudobulk_gprofiler_result: EnrichmentBundleResult | None,
    continuous_result: ContinuousEvidenceResult | None,
    continuous_gprofiler_result: EnrichmentBundleResult | None,
    enrichr_result: EnrichmentBundleResult | None,
    pseudobulk_enrichr_result: EnrichmentBundleResult | None,
    continuous_enrichr_result: EnrichmentBundleResult | None,
    subpopulation_pseudobulk_result: SubpopulationPseudobulkResult | None,
    metadata_association_summary: pd.DataFrame | None,
    correlations: pd.DataFrame,
    protein_evidence_result: ProteinEvidenceResult | None,
    genetics_evidence_result: GeneticsEvidenceResult | None,
    brain_context_result: BrainContextEvidenceResult | None,
    protein_context_result: ProteinContextSummaryResult | None,
    public_proteomics_overlap_result: PublicProteomicsOverlapResult | None,
    pathway_activity_proxy_result: PathwayActivityProxyResult | None,
    module_status_summary: pd.DataFrame,
    plot_status_summary: pd.DataFrame,
    output_files: list[str],
) -> list[Path]:
    """Write a markdown report in the run folder and mirrored reports folder."""
    report_text = build_report_text(
        config=config,
        config_snapshot_path=config_snapshot_path,
        dataset_summary=dataset_summary,
        metadata_validation=metadata_validation,
        metadata_suggestions=metadata_suggestions,
        gene_presence=gene_presence,
        state_result=state_result,
        state_summary=state_summary,
        preprocessing_summary=preprocessing_summary,
        de_results=de_results,
        pseudobulk_result=pseudobulk_result,
        gprofiler_result=gprofiler_result,
        pseudobulk_gprofiler_result=pseudobulk_gprofiler_result,
        continuous_result=continuous_result,
        continuous_gprofiler_result=continuous_gprofiler_result,
        enrichr_result=enrichr_result,
        pseudobulk_enrichr_result=pseudobulk_enrichr_result,
        continuous_enrichr_result=continuous_enrichr_result,
        subpopulation_pseudobulk_result=subpopulation_pseudobulk_result,
        metadata_association_summary=metadata_association_summary,
        correlations=correlations,
        protein_evidence_result=protein_evidence_result,
        genetics_evidence_result=genetics_evidence_result,
        brain_context_result=brain_context_result,
        protein_context_result=protein_context_result,
        public_proteomics_overlap_result=public_proteomics_overlap_result,
        pathway_activity_proxy_result=pathway_activity_proxy_result,
        module_status_summary=module_status_summary,
        plot_status_summary=plot_status_summary,
        output_files=output_files,
    )

    run_report_path = run_dir / "run_summary.md"
    run_report_path.write_text(report_text, encoding="utf-8")
    report_copy_path.write_text(report_text, encoding="utf-8")
    return [run_report_path, report_copy_path]


def build_report_text(
    config: dict[str, object],
    config_snapshot_path: Path,
    dataset_summary: dict[str, object],
    metadata_validation: pd.DataFrame,
    metadata_suggestions: pd.DataFrame,
    gene_presence: pd.DataFrame,
    state_result: StateAssignmentResult,
    state_summary: pd.DataFrame,
    preprocessing_summary: dict[str, object],
    de_results: pd.DataFrame,
    pseudobulk_result: PseudobulkResult | None,
    gprofiler_result: EnrichmentBundleResult | None,
    pseudobulk_gprofiler_result: EnrichmentBundleResult | None,
    continuous_result: ContinuousEvidenceResult | None,
    continuous_gprofiler_result: EnrichmentBundleResult | None,
    enrichr_result: EnrichmentBundleResult | None,
    pseudobulk_enrichr_result: EnrichmentBundleResult | None,
    continuous_enrichr_result: EnrichmentBundleResult | None,
    subpopulation_pseudobulk_result: SubpopulationPseudobulkResult | None,
    metadata_association_summary: pd.DataFrame | None,
    correlations: pd.DataFrame,
    protein_evidence_result: ProteinEvidenceResult | None,
    genetics_evidence_result: GeneticsEvidenceResult | None,
    brain_context_result: BrainContextEvidenceResult | None,
    protein_context_result: ProteinContextSummaryResult | None,
    public_proteomics_overlap_result: PublicProteomicsOverlapResult | None,
    pathway_activity_proxy_result: PathwayActivityProxyResult | None,
    module_status_summary: pd.DataFrame,
    plot_status_summary: pd.DataFrame,
    output_files: list[str],
) -> str:
    """Compose a compact markdown run summary."""
    found_metadata = metadata_validation[metadata_validation["exists"]]["column_name"].dropna().tolist()
    metadata_lines = [f"- `{column}`" for column in found_metadata] or [
        "- No configured metadata columns were set yet."
    ]
    metadata_suggestion_lines = _build_metadata_suggestion_lines(metadata_suggestions)

    if not de_results.empty and "logfoldchanges" in de_results.columns:
        positive_genes = (
            de_results[de_results["logfoldchanges"] > 0]
            .sort_values("logfoldchanges", ascending=False, na_position="last")["names"]
            .head(10)
            .astype(str)
            .tolist()
        )
        negative_genes = (
            de_results[de_results["logfoldchanges"] < 0]
            .sort_values("logfoldchanges", ascending=True, na_position="last")["names"]
            .head(10)
            .astype(str)
            .tolist()
        )
    else:
        positive_genes = []
        negative_genes = []

    top_positive = _format_gene_list(positive_genes)
    top_negative = _format_gene_list(negative_genes)
    preprocessing_lines = [
        f"- Matrix source: `{preprocessing_summary.get('matrix_source', 'X')}`",
        f"- Matrix source detail: {preprocessing_summary.get('matrix_source_detail', 'Using adata.X as the downstream analysis matrix.')}",
        f"- Normalize total applied: `{preprocessing_summary.get('normalize_total')}`",
        f"- Target sum: `{preprocessing_summary.get('target_sum')}`",
        f"- log1p applied: `{preprocessing_summary.get('log1p')}`",
        f"- Median total counts before preprocessing: `{preprocessing_summary.get('input_total_counts_median', 'n/a')}`",
    ]

    pseudobulk_lines = ["- Pseudobulk comparison was not run."]
    if pseudobulk_result is not None:
        pseudobulk_positive = _format_gene_list(
            pseudobulk_result.positive_results["names"].head(10).astype(str).tolist()
        )
        pseudobulk_negative = _format_gene_list(
            pseudobulk_result.negative_results["names"].head(10).astype(str).tolist()
        )
        pseudobulk_lines = [
            f"- Pseudobulk samples retained: `{pseudobulk_result.n_pseudobulk_samples}`",
            f"- Paired donors retained: `{pseudobulk_result.n_paired_donors}`",
            f"- Positive direction: {pseudobulk_positive}",
            f"- Negative direction: {pseudobulk_negative}",
        ]

    gprofiler_lines = _build_enrichment_lines(gprofiler_result, heading_prefix="Cell-level g:Profiler")
    pseudobulk_gprofiler_lines = _build_enrichment_lines(
        pseudobulk_gprofiler_result,
        heading_prefix="Pseudobulk g:Profiler",
    )
    continuous_gprofiler_lines = _build_enrichment_lines(
        continuous_gprofiler_result,
        heading_prefix="Continuous-model g:Profiler",
    )
    enrichr_lines = _build_enrichment_lines(enrichr_result, heading_prefix="Cell-level Enrichr")
    pseudobulk_enrichr_lines = _build_enrichment_lines(
        pseudobulk_enrichr_result,
        heading_prefix="Pseudobulk Enrichr",
    )
    continuous_enrichr_lines = _build_enrichment_lines(
        continuous_enrichr_result,
        heading_prefix="Continuous-model Enrichr",
    )

    association_lines = ["- Metadata association tests were not run."]
    if metadata_association_summary is not None and not metadata_association_summary.empty:
        association_lines = [
            f"- `{row.metadata_column}`: chi-square p=`{row.chi_square_p_value:.3e}`; Cramer's V=`{row.cramers_v:.3f}`"
            if pd.notna(row.chi_square_p_value)
            else f"- `{row.metadata_column}`: insufficient diversity for chi-square test"
            for row in metadata_association_summary.head(5).itertuples(index=False)
        ]

    correlation_lines = [
        f"- `{row.gene}`: Pearson `{row.pearson_r:.3f}`, Spearman `{row.spearman_r:.3f}`"
        if pd.notna(row.pearson_r) and pd.notna(row.spearman_r)
        else f"- `{row.gene}`: insufficient variance for correlation"
        for row in correlations.head(10).itertuples(index=False)
    ] or ["- No candidate-gene correlations were available."]

    state_lines = [
        f"- `{getattr(row, state_result.column_name)}`: {row.count}"
        for row in state_summary.itertuples(index=False)
    ]

    gene_presence_lines = [
        f"- `{row.gene}`: {'present' if row.present else 'missing'}"
        for row in gene_presence.itertuples(index=False)
    ]

    protein_lines = _build_protein_evidence_lines(protein_evidence_result)
    protein_context_lines = _build_protein_context_lines(protein_context_result)
    public_proteomics_lines = _build_public_proteomics_lines(public_proteomics_overlap_result)
    genetics_lines = _build_genetics_evidence_lines(genetics_evidence_result)
    continuous_lines = _build_continuous_lines(continuous_result)
    subpopulation_lines = _build_subpopulation_lines(subpopulation_pseudobulk_result)
    activity_proxy_lines = _build_activity_proxy_lines(pathway_activity_proxy_result)
    brain_context_lines = _build_brain_context_lines(brain_context_result, config["gene_of_interest"])
    module_status_lines = _build_status_lines(module_status_summary, "module_name")
    plot_status_lines = _build_status_lines(plot_status_summary, "plot_name")
    output_lines = [f"- `{path}`" for path in output_files] or ["- No outputs were generated."]

    lines = [
        f"# Run Summary: {config['dataset_name']}",
        "",
        "## Overview",
        f"- Dataset name: `{config['dataset_name']}`",
        f"- Requested config: `{config['_config_path']}`",
        f"- Effective config snapshot: `{config_snapshot_path}`",
        f"- File path analyzed: `{config['input_path']}`",
        f"- Matrix source used downstream: `{preprocessing_summary.get('matrix_source', 'X')}`",
        f"- Cells: `{dataset_summary['n_cells']}`",
        f"- Genes: `{dataset_summary['n_genes']}`",
        "",
        "## Metadata Columns Found",
        *metadata_lines,
        "",
        "## Metadata Suggestions",
        *metadata_suggestion_lines,
        "",
        "## Gene Presence Check",
        *gene_presence_lines,
        "",
        "## Preprocessing",
        *preprocessing_lines,
        "",
        "## State Definition",
        f"- Method: `{state_result.threshold_method}`",
        f"- Threshold value: `{state_result.threshold_value:.6f}`",
        f"- New column: `{state_result.column_name}`",
        f"- Note: {state_result.note}",
        "",
        "## State Counts",
        *state_lines,
        "",
        "## Top Differential Expression Genes",
        f"- Positive direction: {top_positive}",
        f"- Negative direction: {top_negative}",
        "",
        "## Donor-Aware Pseudobulk Comparison",
        *pseudobulk_lines,
        "",
        "## Continuous And Hurdle Modeling",
        *continuous_lines,
        "",
        "## Within-Subpopulation Pseudobulk",
        *subpopulation_lines,
        "",
        "## Database Enrichment",
        *gprofiler_lines,
        *pseudobulk_gprofiler_lines,
        *continuous_gprofiler_lines,
        *enrichr_lines,
        *pseudobulk_enrichr_lines,
        *continuous_enrichr_lines,
        "",
        "## State Associations",
        *association_lines,
        "",
        "## Candidate Correlation Summary",
        *correlation_lines,
        "",
        "## Protein And PTM Evidence",
        *protein_lines,
        "",
        "## Protein-Level Context",
        *protein_context_lines,
        "",
        "## Public Proteomics Overlap",
        *public_proteomics_lines,
        "",
        "## Human Genetics Evidence",
        *genetics_lines,
        "",
        "## Brain Atlas And External Proteomics",
        *brain_context_lines,
        "",
        "## Transcript-Based Pathway Activity Proxies",
        *activity_proxy_lines,
        "",
        "## Module Status",
        *module_status_lines,
        "",
        "## Plot Status",
        *plot_status_lines,
        "",
        "## Interpretation Guardrail",
        "- Results in this run should be interpreted as transcriptomic associations and state-linked patterns.",
        "- iRhom2/RHBDF2 biology is strongly post-translational, so these outputs are hypothesis-generating context rather than direct mechanism proof.",
        "- Cell-level DE remains exploratory; donor-aware pseudobulk, continuous-model, and within-subpopulation outputs should carry more weight when making transcript-level association claims.",
        "- The binary RHBDF2 state is retained for comparability, but continuous and hurdle-style outputs reduce dependence on a zero-versus-nonzero split.",
        "- Transcript-based pathway activity proxy scores are indirect readouts consistent with pathway activation and should not be described as direct protein activity measurements.",
        "- Differential expression and enrichment-ready gene lists should not be treated as causal evidence.",
        "",
        "## Generated Outputs",
        *output_lines,
        "",
    ]
    return "\n".join(lines)


def _format_gene_list(genes: list[str]) -> str:
    return ", ".join(f"`{gene}`" for gene in genes) if genes else "none exported"


def _build_metadata_suggestion_lines(metadata_suggestions: pd.DataFrame) -> list[str]:
    if metadata_suggestions.empty:
        return ["- No likely metadata columns were detected from column-name heuristics."]
    lines = []
    grouped = metadata_suggestions.groupby("suggested_role", sort=True)
    for suggested_role, group in grouped:
        top = group.iloc[0]
        lines.append(
            f"- `{suggested_role}` candidate: `{top['column_name']}` "
            f"(keyword hits: `{top['keyword_hits']}`, unique values: `{int(top['n_unique'])}`)"
        )
    return lines


def _build_enrichment_lines(
    result: EnrichmentBundleResult | None,
    heading_prefix: str,
) -> list[str]:
    if result is None:
        return [f"- {heading_prefix} enrichment was not run."]

    lines = [
        f"- {heading_prefix} up top terms: "
        + (_format_term_list(result.up.top_terms) if result.up.top_terms else "no significant terms"),
        f"- {heading_prefix} down top terms: "
        + (_format_term_list(result.down.top_terms) if result.down.top_terms else "no significant terms"),
    ]
    return lines


def _format_term_list(terms: list[str]) -> str:
    return "; ".join(terms)


def _build_protein_evidence_lines(
    protein_evidence_result: ProteinEvidenceResult | None,
) -> list[str]:
    if protein_evidence_result is None:
        return ["- Protein/PTM evidence was not run."]

    summary = protein_evidence_result.summary
    found_count = int(summary["uniprot_found"].fillna(False).sum()) if not summary.empty else 0
    total_count = int(len(summary))
    ptm_positive = summary[summary["annotated_ptm_feature_count"].fillna(0) > 0]

    lines = [
        f"- UniProt reviewed proteins found: `{found_count}/{total_count}`",
        "- Candidate proteins with annotated PTM features: "
        + (", ".join(f"`{item}`" for item in protein_evidence_result.top_ptm_genes) if protein_evidence_result.top_ptm_genes else "none detected"),
        "- Candidate-panel OmniPath interactions: "
        + (", ".join(f"`{item}`" for item in protein_evidence_result.top_candidate_edges) if protein_evidence_result.top_candidate_edges else "none detected"),
        "- Candidate-panel PTM or cleavage edges: "
        + (
            ", ".join(f"`{item}`" for item in protein_evidence_result.top_candidate_panel_ptm_edges)
            if protein_evidence_result.top_candidate_panel_ptm_edges
            else "none detected"
        ),
    ]
    if not ptm_positive.empty:
        max_ptm_gene = ptm_positive.sort_values("annotated_ptm_feature_count", ascending=False).iloc[0]
        lines.append(
            f"- Highest PTM annotation burden in the current panel: `{max_ptm_gene['gene']}` with `{int(max_ptm_gene['annotated_ptm_feature_count'])}` UniProt PTM features"
        )
    return lines


def _build_protein_context_lines(
    protein_context_result: ProteinContextSummaryResult | None,
) -> list[str]:
    if protein_context_result is None:
        return ["- Protein-context evidence summary was not generated."]
    summary = protein_context_result.summary
    supported = summary[summary["protein_context_evidence_supported"].fillna(False).astype(bool)] if not summary.empty else pd.DataFrame()
    return [
        f"- Candidate genes with merged protein-context support: `{int(len(supported))}/{int(len(summary)) if not summary.empty else 0}`",
        "- Top supported genes: "
        + (", ".join(f"`{item}`" for item in protein_context_result.top_supported_genes) if protein_context_result.top_supported_genes else "none detected"),
    ]


def _build_public_proteomics_lines(
    public_proteomics_overlap_result: PublicProteomicsOverlapResult | None,
) -> list[str]:
    if public_proteomics_overlap_result is None:
        return ["- Public proteomics overlap summary was not generated."]
    summary = public_proteomics_overlap_result.summary
    detected = summary["proteomics_gene_detected"].fillna(False).astype(bool).sum() if not summary.empty and "proteomics_gene_detected" in summary.columns else 0
    concordant = summary["public_proteomics_overlap_supported"].fillna(False).astype(bool).sum() if not summary.empty and "public_proteomics_overlap_supported" in summary.columns else 0
    context_label = "public proteomics"
    if not summary.empty and "proteomics_context_label" in summary.columns:
        labels = summary["proteomics_context_label"].dropna().astype(str)
        if not labels.empty and labels.iloc[0].strip():
            context_label = labels.iloc[0].strip()
    return [
        f"- Candidate genes detected in the {context_label} dataset: `{int(detected)}`",
        f"- Candidate genes with directionally concordant transcript/{context_label} overlap: `{int(concordant)}`",
        "- Top concordant genes: "
        + (", ".join(f"`{item}`" for item in public_proteomics_overlap_result.top_concordant_genes) if public_proteomics_overlap_result.top_concordant_genes else "none detected"),
    ]


def _build_genetics_evidence_lines(
    genetics_evidence_result: GeneticsEvidenceResult | None,
) -> list[str]:
    if genetics_evidence_result is None:
        return ["- Human genetics evidence was not run."]

    summary = genetics_evidence_result.summary
    supported = summary[summary["combined_genetics_support"].fillna(False)]
    supported_count = int(len(supported))
    total_count = int(len(summary))
    disease_focus_label = "configured disease"
    if not summary.empty and "disease_focus_label" in summary.columns:
        first_label = str(summary["disease_focus_label"].dropna().iloc[0]) if summary["disease_focus_label"].dropna().shape[0] else ""
        if first_label:
            disease_focus_label = first_label

    lines = [
        f"- Candidate genes with {disease_focus_label}-relevant human genetics support: `{supported_count}/{total_count}`",
        "- Top supported genes: "
        + (", ".join(f"`{item}`" for item in genetics_evidence_result.top_genes) if genetics_evidence_result.top_genes else "none detected"),
    ]
    if not supported.empty:
        best_row = supported.sort_values(["opentargets_max_score", "gwas_association_count"], ascending=[False, False]).iloc[0]
        lines.append(
            f"- Strongest genetics signal in the current panel: `{best_row['gene']}` with Open Targets score `{float(best_row['opentargets_max_score']):.3f}` and `{int(best_row['gwas_association_count'])}` matching GWAS associations"
        )
    return lines


def _build_continuous_lines(
    continuous_result: ContinuousEvidenceResult | None,
) -> list[str]:
    if continuous_result is None:
        return ["- Continuous and hurdle-style modeling was not run."]

    positive = _format_gene_list(
        continuous_result.positive_results["names"].head(10).astype(str).tolist()
    )
    negative = _format_gene_list(
        continuous_result.negative_results["names"].head(10).astype(str).tolist()
    )
    hurdle = continuous_result.candidate_hurdle_summary.copy()
    hurdle["best_padj"] = hurdle[
        ["detection_padj", "positive_expression_padj"]
    ].min(axis=1, skipna=True)
    hurdle = hurdle.sort_values("best_padj", ascending=True, na_position="last")
    top_hurdle = [
        f"`{row.gene}` (detect_padj={_format_optional_pvalue(row.detection_padj)}, positive_padj={_format_optional_pvalue(row.positive_expression_padj)})"
        for row in hurdle.head(5).itertuples(index=False)
    ]
    return [
        f"- Transcriptome-wide positive direction: {positive}",
        f"- Transcriptome-wide negative direction: {negative}",
        "- Top candidate hurdle signals: " + (", ".join(top_hurdle) if top_hurdle else "none exported"),
    ]


def _build_subpopulation_lines(
    subpopulation_pseudobulk_result: SubpopulationPseudobulkResult | None,
) -> list[str]:
    if subpopulation_pseudobulk_result is None:
        return ["- Within-subpopulation pseudobulk was not run or no subgroup column was available."]

    run_summary = subpopulation_pseudobulk_result.run_summary
    completed = run_summary[run_summary["status"] == "completed"] if not run_summary.empty else pd.DataFrame()
    gene_summary = subpopulation_pseudobulk_result.gene_summary
    top_support = []
    if not gene_summary.empty:
        sortable = gene_summary.sort_values(
            ["subpopulation_pseudobulk_n_subgroups_significant", "subpopulation_pseudobulk_n_subgroups_positive", "gene"],
            ascending=[False, False, True],
        )
        top_support = [
            f"`{row.gene}` (significant_subgroups={int(row.subpopulation_pseudobulk_n_subgroups_significant)}, best=`{row.subpopulation_pseudobulk_best_subgroup}`)"
            for row in sortable.head(5).itertuples(index=False)
        ]
    return [
        f"- Subgroups completed: `{int(completed.shape[0])}`",
        f"- Subgroups attempted: `{int(run_summary.shape[0])}`",
        "- Genes with the strongest within-subpopulation support: "
        + (", ".join(top_support) if top_support else "none exported"),
    ]


def _build_brain_context_lines(
    brain_context_result: BrainContextEvidenceResult | None,
    gene_of_interest: str,
) -> list[str]:
    if brain_context_result is None:
        return ["- Brain atlas and external proteomics evidence was not run."]

    summary = brain_context_result.combined_summary
    rhbdf2_row = summary[summary["gene"] == gene_of_interest]
    rhbdf2_lines = []
    if not rhbdf2_row.empty:
        row = rhbdf2_row.iloc[0]
        rhbdf2_lines.append(
            f"- `{gene_of_interest}` HPA microglia context: microglia nCPM=`{_format_optional_float(row.get('hpa_microglia_nCPM'))}`, CNS macrophage nCPM=`{_format_optional_float(row.get('hpa_cns_macrophage_nCPM'))}`"
        )
        rhbdf2_lines.append(
            f"- `{gene_of_interest}` {brain_context_result.context_label} mean log2 fold change: `{_format_optional_float(row.get('proteomics_mean_log2_fc_case_vs_control'))}`"
        )
    return [
        "- Top HPA brain or microglia-context genes: "
        + (", ".join(f"`{item}`" for item in brain_context_result.top_hpa_microglia_genes) if brain_context_result.top_hpa_microglia_genes else "none detected"),
        f"- Top {brain_context_result.context_label} genes from the configured proteomics context: "
        + (", ".join(f"`{item}`" for item in brain_context_result.top_case_proteomics_genes) if brain_context_result.top_case_proteomics_genes else "none detected"),
        *rhbdf2_lines,
    ]


def _build_activity_proxy_lines(
    pathway_activity_proxy_result: PathwayActivityProxyResult | None,
) -> list[str]:
    if pathway_activity_proxy_result is None:
        return ["- Transcript-based pathway activity proxies were not run."]
    summary = pathway_activity_proxy_result.summary
    sortable = summary.sort_values(
        ["donor_paired_padj", "cell_level_padj", "module_name"],
        ascending=[True, True, True],
        na_position="last",
    ) if not summary.empty else pd.DataFrame()
    top_rows = [
        f"`{row.module_name}` (delta={_format_optional_float(row.cell_level_delta_high_minus_low)}, donor_padj={_format_optional_pvalue(row.donor_paired_padj)})"
        for row in sortable.head(5).itertuples(index=False)
    ]
    return [
        "- Top transcript-based proxy modules: "
        + (", ".join(top_rows) if top_rows else "none detected"),
        "- These module scores are indirect transcript-based proxies and not direct protein activity measurements.",
    ]


def _build_status_lines(status_summary: pd.DataFrame, name_column: str) -> list[str]:
    if status_summary is None or status_summary.empty:
        return ["- No status summary was recorded."]
    lines = []
    for row in status_summary.itertuples(index=False):
        name = getattr(row, name_column)
        message = getattr(row, "message", "")
        status = getattr(row, "status", "unknown")
        lines.append(f"- `{name}`: `{status}`" + (f" ({message})" if message else ""))
    return lines


def _format_optional_pvalue(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2e}"


def _format_optional_float(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"
