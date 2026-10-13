#!/usr/bin/env python3
"""Command-line pipeline for exploratory RHBDF2 microglia transcriptomics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.brain_context_utils import run_brain_context_evidence
from src.config_utils import load_config
from src.continuous_evidence_utils import run_continuous_evidence
from src.de_utils import run_differential_expression
from src.enrich_utils import (
    export_enrichment_gene_lists,
    export_ranked_gene_lists,
    run_enrichr_enrichment,
    run_gprofiler_enrichment,
)
from src.evidence_utils import build_candidate_evidence_table, summarize_state_metadata_associations
from src.gene_utils import (
    compute_candidate_gene_correlations,
    ensure_unique_var_names,
    gene_presence_table,
    require_gene_present,
    summarize_candidate_genes_by_group,
)
from src.genetics_evidence_utils import run_genetics_evidence
from src.activity_proxy_utils import run_pathway_activity_proxies
from src.io_utils import (
    create_run_paths,
    list_relative_outputs,
    load_h5ad,
    save_config_snapshot,
    save_dataset_summary,
    save_status_table,
    summarize_anndata,
)
from src.metadata_utils import export_value_counts, suggest_metadata_columns, summarize_metadata, validate_metadata_columns
from src.plot_utils import (
    save_correlation_heatmap,
    save_dotplot,
    save_matrixplot,
    save_umap_plots_if_available,
    save_violin_plot,
)
from src.preprocess_utils import prepare_analysis_anndata
from src.protein_context_utils import build_protein_context_summary
from src.protein_evidence_utils import run_protein_evidence
from src.public_proteomics_overlap_utils import summarize_public_proteomics_overlap
from src.pseudobulk_utils import run_pseudobulk_differential_expression, run_within_subpopulation_pseudobulk
from src.report_utils import generate_run_report
from src.state_utils import assign_expression_state, build_state_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the exploratory RHBDF2 microglia transcriptomics pipeline."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    run_paths = create_run_paths(config)
    module_status_rows: list[dict[str, object]] = []

    print("[1/14] Creating run folders and saving the resolved config")
    config_snapshot_path = save_config_snapshot(config, run_paths.summaries_dir / "config_used.yaml")

    print(f"[2/14] Loading AnnData file from {config['input_path']}")
    adata = load_h5ad(config["input_path"])
    ensure_unique_var_names(adata)

    print("[3/14] Summarizing the loaded dataset")
    dataset_summary = summarize_anndata(adata, config["gene_of_interest"])
    save_dataset_summary(adata, dataset_summary, run_paths.summaries_dir)

    print("[4/14] Validating configured metadata columns")
    metadata_validation = validate_metadata_columns(adata, config["metadata"])
    metadata_validation.to_csv(run_paths.summaries_dir / "metadata_validation.csv", index=False)
    metadata_summary = summarize_metadata(adata)
    metadata_summary.to_csv(run_paths.summaries_dir / "metadata_summary.csv", index=False)
    metadata_suggestions = suggest_metadata_columns(adata)
    metadata_suggestions.to_csv(run_paths.summaries_dir / "metadata_suggestions.csv", index=False)
    export_value_counts(
        adata,
        list(config["metadata"].values()),
        run_paths.summaries_dir,
    )

    print("[5/14] Selecting the configured analysis matrix and preparing the analysis copy")
    preprocessing_result = prepare_analysis_anndata(
        adata,
        config["preprocessing"],
        run_paths.summaries_dir,
    )
    source_adata = preprocessing_result.source_adata

    print("[6/14] Checking the gene of interest and candidate panel")
    genes_to_check = [config["gene_of_interest"], *config["candidate_genes"]]
    gene_presence = gene_presence_table(source_adata, genes_to_check)
    gene_presence.to_csv(run_paths.summaries_dir / "gene_presence.csv", index=False)
    require_gene_present(source_adata, config["gene_of_interest"])

    print("[7/14] Assigning the exploratory RHBDF2 high/low state")
    source_adata, state_result = assign_expression_state(
        source_adata,
        config["gene_of_interest"],
        config["state_definition"],
    )
    adata.obs[state_result.column_name] = source_adata.obs[state_result.column_name].copy()
    preprocessing_result.adata.obs[state_result.column_name] = source_adata.obs[state_result.column_name].copy()
    state_summary = build_state_summary(source_adata, state_result)
    state_summary.to_csv(run_paths.summaries_dir / "state_summary.csv", index=False)

    print("[8/14] Exporting candidate summaries and exploratory plots")
    candidate_summary = summarize_candidate_genes_by_group(
        preprocessing_result.adata,
        config["candidate_genes"],
        state_result.column_name,
        run_paths.tables_dir / "candidate_gene_summary_by_state.csv",
    )
    plot_status = maybe_save_plots(preprocessing_result.adata, config, run_paths.plots_dir, state_result.column_name)
    save_status_table(plot_status, run_paths.plot_status_path)

    print("[9/14] Running exploratory cell-level differential expression")
    de_config = dict(config["differential_expression"])
    de_config["group_column"] = state_result.column_name
    de_result = run_differential_expression(preprocessing_result.adata, de_config, run_paths.tables_dir)
    export_enrichment_gene_lists(
        de_result.positive_results,
        de_result.negative_results,
        run_paths.enrichment_dir,
    )
    export_ranked_gene_lists(
        results=de_result.full_results,
        output_dir=run_paths.enrichment_dir,
        effect_column="logfoldchanges",
        padj_column="pvals_adj",
        prefix="cell_level_",
    )
    background_genes = (
        [str(gene) for gene in source_adata.var_names]
        if config.get("enrichment", {}).get("use_dataset_background", True)
        else None
    )
    gprofiler_result = _run_optional_step(
        module_name="cell_level_gprofiler",
        enabled=bool(config.get("enrichment", {}).get("enabled", True)),
        action=lambda: run_gprofiler_enrichment(
            de_result.positive_results,
            de_result.negative_results,
            config["enrichment"],
            run_paths.enrichment_dir,
            background_genes=background_genes,
        ),
        status_rows=module_status_rows,
    )
    enrichr_background = (
        [str(gene) for gene in source_adata.var_names]
        if config.get("enrichr", {}).get("use_dataset_background", True)
        else None
    )
    enrichr_result = _run_optional_step(
        module_name="cell_level_enrichr",
        enabled=bool(config.get("enrichr", {}).get("enabled", True)),
        action=lambda: run_enrichr_enrichment(
            de_result.positive_results,
            de_result.negative_results,
            config["enrichr"],
            run_paths.enrichment_dir,
            background_genes=enrichr_background,
        ),
        status_rows=module_status_rows,
    )

    print("[10/14] Running donor-aware pseudobulk comparison")
    pseudobulk_result = None
    pseudobulk_gprofiler_result = None
    pseudobulk_enrichr_result = None
    subpopulation_pseudobulk_result = None
    if config.get("pseudobulk", {}).get("enabled", True):
        pseudobulk_config = dict(config["pseudobulk"])
        pseudobulk_config["group_column"] = pseudobulk_config.get("group_column") or state_result.column_name
        pseudobulk_config["group_a"] = pseudobulk_config.get("group_a") or de_config["group_a"]
        pseudobulk_config["group_b"] = pseudobulk_config.get("group_b") or de_config["group_b"]
        pseudobulk_result = _run_optional_step(
            module_name="pseudobulk",
            enabled=True,
            action=lambda: run_pseudobulk_differential_expression(
                adata=source_adata,
                pseudobulk_config=pseudobulk_config,
                output_dir=run_paths.pseudobulk_dir,
                candidate_genes=config["candidate_genes"],
                state_column=state_result.column_name,
                fallback_donor_column=config["metadata"].get("donor_column"),
            ),
            status_rows=module_status_rows,
            eligibility_exceptions=(ValueError,),
        )
        if pseudobulk_result is not None:
            export_enrichment_gene_lists(
                pseudobulk_result.positive_results,
                pseudobulk_result.negative_results,
                run_paths.enrichment_dir,
                prefix="pseudobulk_",
            )
            export_ranked_gene_lists(
                results=pseudobulk_result.full_results,
                output_dir=run_paths.enrichment_dir,
                effect_column="effect_size_log1p_cpm",
                padj_column="pvals_adj",
                prefix="pseudobulk_",
            )
            pseudobulk_gprofiler_result = _run_optional_step(
                module_name="pseudobulk_gprofiler",
                enabled=bool(config.get("enrichment", {}).get("enabled", True)),
                action=lambda: run_gprofiler_enrichment(
                    pseudobulk_result.positive_results,
                    pseudobulk_result.negative_results,
                    config["enrichment"],
                    run_paths.enrichment_dir,
                    background_genes=background_genes,
                    prefix="pseudobulk_",
                ),
                status_rows=module_status_rows,
            )
            pseudobulk_enrichr_result = _run_optional_step(
                module_name="pseudobulk_enrichr",
                enabled=bool(config.get("enrichr", {}).get("enabled", True)),
                action=lambda: run_enrichr_enrichment(
                    pseudobulk_result.positive_results,
                    pseudobulk_result.negative_results,
                    config["enrichr"],
                    run_paths.enrichment_dir,
                    background_genes=enrichr_background,
                    prefix="pseudobulk_",
                ),
                status_rows=module_status_rows,
            )

            subgroup_column = pseudobulk_config.get("subgroup_column") or config["metadata"].get("cluster_column")
            subpopulation_pseudobulk_result = _run_optional_step(
                module_name="subpopulation_pseudobulk",
                enabled=bool(pseudobulk_config.get("within_subpopulation", True)),
                action=lambda: run_within_subpopulation_pseudobulk(
                    adata=source_adata,
                    pseudobulk_config=pseudobulk_config,
                    output_dir=run_paths.pseudobulk_dir,
                    candidate_genes=config["candidate_genes"],
                    state_column=state_result.column_name,
                    fallback_donor_column=config["metadata"].get("donor_column"),
                    subgroup_column=str(subgroup_column) if subgroup_column else None,
                ),
                status_rows=module_status_rows,
            )
    else:
        _record_status(module_status_rows, "pseudobulk", False, "skipped", "Disabled in config.")
        _record_status(module_status_rows, "subpopulation_pseudobulk", False, "skipped", "Disabled in config.")

    print("[11/14] Running continuous and hurdle-style association modeling")
    continuous_result = _run_optional_step(
        module_name="continuous_evidence",
        enabled=bool(config.get("continuous_evidence", {}).get("enabled", True)),
        action=lambda: run_continuous_evidence(
            adata=preprocessing_result.adata,
            gene_of_interest=config["gene_of_interest"],
            candidate_genes=config["candidate_genes"],
            continuous_config=config["continuous_evidence"],
            output_dir=run_paths.continuous_dir,
            fallback_donor_column=config["metadata"].get("donor_column"),
        ),
        status_rows=module_status_rows,
    )
    continuous_gprofiler_result = None
    continuous_enrichr_result = None
    pathway_activity_proxy_result = None
    if continuous_result is not None:
        export_enrichment_gene_lists(
            continuous_result.positive_results,
            continuous_result.negative_results,
            run_paths.enrichment_dir,
            prefix="continuous_",
        )
        export_ranked_gene_lists(
            results=continuous_result.full_results,
            output_dir=run_paths.enrichment_dir,
            effect_column="slope_per_log1p_rhbdf2",
            padj_column="pvals_adj",
            prefix="continuous_",
        )
        continuous_gprofiler_result = _run_optional_step(
            module_name="continuous_gprofiler",
            enabled=bool(config.get("enrichment", {}).get("enabled", True)),
            action=lambda: run_gprofiler_enrichment(
                continuous_result.positive_results,
                continuous_result.negative_results,
                config["enrichment"],
                run_paths.enrichment_dir,
                background_genes=background_genes,
                prefix="continuous_",
            ),
            status_rows=module_status_rows,
        )
        continuous_enrichr_result = _run_optional_step(
            module_name="continuous_enrichr",
            enabled=bool(config.get("enrichr", {}).get("enabled", True)),
            action=lambda: run_enrichr_enrichment(
                continuous_result.positive_results,
                continuous_result.negative_results,
                config["enrichr"],
                run_paths.enrichment_dir,
                background_genes=enrichr_background,
                prefix="continuous_",
            ),
            status_rows=module_status_rows,
        )
    pathway_activity_proxy_result = _run_optional_step(
        module_name="pathway_activity_proxy",
        enabled=bool(config.get("pathway_activity_proxy", {}).get("enabled", True)),
        action=lambda: run_pathway_activity_proxies(
            adata=preprocessing_result.adata,
            gene_of_interest=config["gene_of_interest"],
            state_column=state_result.column_name,
            high_label=state_result.high_label,
            low_label=state_result.low_label,
            proxy_config=config.get("pathway_activity_proxy", {}),
            output_dir=run_paths.activity_proxies_dir,
            fallback_donor_column=config["metadata"].get("donor_column"),
        ),
        status_rows=module_status_rows,
    )

    print("[12/14] Computing candidate-gene correlations")
    correlations = compute_candidate_gene_correlations(
        preprocessing_result.adata,
        config["gene_of_interest"],
        config["candidate_genes"],
        run_paths.tables_dir / "candidate_gene_correlations.csv",
    )
    save_correlation_heatmap(
        correlations,
        run_paths.plots_dir / "candidate_gene_correlation_heatmap.png",
    )
    metadata_association_summary, _ = summarize_state_metadata_associations(
        adata=adata,
        state_column=state_result.column_name,
        metadata_columns=config.get("evidence", {}).get("metadata_association_columns", []),
        output_dir=run_paths.tables_dir,
    )

    print("[13/14] Collecting external evidence layers and compiling the candidate summary")
    protein_evidence_result = _run_optional_step(
        module_name="protein_evidence",
        enabled=bool(config.get("protein_evidence", {}).get("enabled", True)),
        action=lambda: run_protein_evidence(
            candidate_genes=[config["gene_of_interest"], *config["candidate_genes"]],
            protein_config=config.get("protein_evidence", {}),
            output_dir=run_paths.external_evidence_dir,
        ),
        status_rows=module_status_rows,
    )
    genetics_evidence_result = _run_optional_step(
        module_name="genetics_evidence",
        enabled=bool(config.get("genetics_evidence", {}).get("enabled", True)),
        action=lambda: run_genetics_evidence(
            candidate_genes=[config["gene_of_interest"], *config["candidate_genes"]],
            genetics_config=config.get("genetics_evidence", {}),
            output_dir=run_paths.external_evidence_dir,
        ),
        status_rows=module_status_rows,
    )
    brain_context_result = _run_optional_step(
        module_name="brain_context_evidence",
        enabled=bool(config.get("brain_context_evidence", {}).get("enabled", True)),
        action=lambda: run_brain_context_evidence(
            candidate_genes=[config["gene_of_interest"], *config["candidate_genes"]],
            brain_context_config=config.get("brain_context_evidence", {}),
            output_dir=run_paths.external_evidence_dir,
        ),
        status_rows=module_status_rows,
    )
    public_proteomics_overlap_result = summarize_public_proteomics_overlap(
        candidate_genes=config["candidate_genes"],
        cell_de_results=de_result.full_results,
        pseudobulk_de_results=pseudobulk_result.full_results if pseudobulk_result is not None else None,
        continuous_results=continuous_result.full_results if continuous_result is not None else None,
        brain_context_summary=brain_context_result.combined_summary if brain_context_result is not None else None,
        output_dir=run_paths.external_evidence_dir,
    )
    protein_context_result = build_protein_context_summary(
        candidate_genes=config["candidate_genes"],
        protein_summary=protein_evidence_result.summary if protein_evidence_result is not None else None,
        brain_context_summary=brain_context_result.combined_summary if brain_context_result is not None else None,
        proteomics_overlap_summary=public_proteomics_overlap_result.summary if public_proteomics_overlap_result is not None else None,
        output_dir=run_paths.external_evidence_dir,
    )
    build_candidate_evidence_table(
        candidate_genes=config["candidate_genes"],
        candidate_summary=candidate_summary,
        correlations=correlations,
        cell_de_results=de_result.full_results,
        continuous_results=continuous_result.full_results if continuous_result is not None else None,
        pseudobulk_de_results=pseudobulk_result.full_results if pseudobulk_result is not None else None,
        pseudobulk_candidate_expression=pseudobulk_result.candidate_expression if pseudobulk_result is not None else None,
        pseudobulk_paired_candidate_deltas=pseudobulk_result.paired_candidate_deltas if pseudobulk_result is not None else None,
        candidate_hurdle_summary=continuous_result.candidate_hurdle_summary if continuous_result is not None else None,
        subpopulation_pseudobulk_gene_summary=(
            subpopulation_pseudobulk_result.gene_summary if subpopulation_pseudobulk_result is not None else None
        ),
        protein_summary=protein_evidence_result.summary if protein_evidence_result is not None else None,
        genetics_summary=genetics_evidence_result.summary if genetics_evidence_result is not None else None,
        brain_context_summary=brain_context_result.combined_summary if brain_context_result is not None else None,
        proteomics_overlap_summary=public_proteomics_overlap_result.summary if public_proteomics_overlap_result is not None else None,
        gene_of_interest=config["gene_of_interest"],
        state_column=state_result.column_name,
        high_label=state_result.high_label,
        low_label=state_result.low_label,
        evidence_config=config.get("evidence", {}),
        output_path=run_paths.tables_dir / "candidate_evidence_summary.csv",
    )
    save_status_table(module_status_rows, run_paths.module_status_path)

    print("[14/14] Writing the markdown run report")
    output_files = sorted(set(list_relative_outputs(run_paths.run_dir) + ["run_summary.md"]))
    generate_run_report(
        config=config,
        run_dir=run_paths.run_dir,
        report_copy_path=run_paths.report_copy_path,
        config_snapshot_path=config_snapshot_path,
        dataset_summary=dataset_summary,
        metadata_validation=metadata_validation,
        metadata_suggestions=metadata_suggestions,
        gene_presence=gene_presence,
        state_result=state_result,
        state_summary=state_summary,
        preprocessing_summary=preprocessing_result.summary,
        de_results=de_result.full_results,
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
        module_status_summary=pd.DataFrame(module_status_rows),
        plot_status_summary=pd.DataFrame(plot_status),
        output_files=output_files,
    )

    print(f"Pipeline completed successfully. Outputs are in: {run_paths.run_dir}")
    return 0


def maybe_save_plots(adata, config: dict, plots_dir: Path, state_column: str) -> list[dict[str, object]]:
    """Generate plots requested in the config while handling optional metadata cleanly."""
    plot_config = config.get("plots", {})
    metadata_config = config.get("metadata", {})
    gene_of_interest = config["gene_of_interest"]
    candidate_genes = config["candidate_genes"]
    rows: list[dict[str, object]] = []

    if plot_config.get("save_violin", True):
        rows.append(
            _attempt_plot(
                plot_name="violin_by_state",
                output_path=plots_dir / f"{gene_of_interest.lower()}_violin_by_state.png",
                action=lambda: save_violin_plot(
                    adata,
                    gene_of_interest,
                    state_column,
                    plots_dir / f"{gene_of_interest.lower()}_violin_by_state.png",
                    plot_config=plot_config,
                ),
            )
        )
        for config_key in ["diagnosis_column", "cluster_column"]:
            column = metadata_config.get(config_key)
            if column:
                rows.append(
                    _attempt_plot(
                        plot_name=f"violin_by_{column}",
                        output_path=plots_dir / f"{gene_of_interest.lower()}_violin_by_{column}.png",
                        action=lambda column=column: save_violin_plot(
                            adata,
                            gene_of_interest,
                            column,
                            plots_dir / f"{gene_of_interest.lower()}_violin_by_{column}.png",
                            plot_config=plot_config,
                        ),
                    )
                )

    if plot_config.get("save_dotplot", True):
        rows.append(
            _attempt_plot(
                plot_name="dotplot_by_state",
                output_path=plots_dir / "candidate_dotplot_by_state.png",
                action=lambda: save_dotplot(
                    adata,
                    candidate_genes,
                    state_column,
                    plots_dir / "candidate_dotplot_by_state.png",
                    plot_config=plot_config,
                ),
            )
        )
        for config_key in ["diagnosis_column", "cluster_column"]:
            column = metadata_config.get(config_key)
            if column:
                rows.append(
                    _attempt_plot(
                        plot_name=f"dotplot_by_{column}",
                        output_path=plots_dir / f"candidate_dotplot_by_{column}.png",
                        action=lambda column=column: save_dotplot(
                            adata,
                            candidate_genes,
                            column,
                            plots_dir / f"candidate_dotplot_by_{column}.png",
                            plot_config=plot_config,
                        ),
                    )
                )

    if plot_config.get("save_matrixplot", True):
        rows.append(
            _attempt_plot(
                plot_name="matrixplot_by_state",
                output_path=plots_dir / "candidate_matrixplot_by_state.png",
                action=lambda: save_matrixplot(
                    adata,
                    candidate_genes,
                    state_column,
                    plots_dir / "candidate_matrixplot_by_state.png",
                    plot_config=plot_config,
                ),
            )
        )

    if plot_config.get("save_umap_if_available", True):
        try:
            saved = save_umap_plots_if_available(
                adata,
                [gene_of_interest, *candidate_genes],
                plots_dir,
                plot_config=plot_config,
            )
            rows.append(
                {
                    "plot_name": "umap_gene_panels",
                    "status": "completed" if saved else "skipped",
                    "output_path": str(plots_dir),
                    "message": "Saved UMAP gene overlays." if saved else "UMAP embedding was not available.",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "plot_name": "umap_gene_panels",
                    "status": "failed",
                    "output_path": str(plots_dir),
                    "message": str(exc),
                }
            )
    return rows


def _attempt_plot(plot_name: str, output_path: Path, action) -> dict[str, object]:
    try:
        saved_path = action()
        if saved_path is None:
            return {
                "plot_name": plot_name,
                "status": "skipped",
                "output_path": str(output_path),
                "message": "Plot was skipped because the required grouping column, embedding, or genes were unavailable.",
            }
        return {
            "plot_name": plot_name,
            "status": "completed",
            "output_path": str(saved_path),
            "message": "Plot generated successfully.",
        }
    except Exception as exc:
        return {
            "plot_name": plot_name,
            "status": "failed",
            "output_path": str(output_path),
            "message": str(exc),
        }


def _run_optional_step(
    module_name: str,
    enabled: bool,
    action,
    status_rows: list[dict[str, object]],
    eligibility_exceptions: tuple[type[BaseException], ...] = (),
):
    if not enabled:
        _record_status(status_rows, module_name, False, "skipped", "Disabled in config.")
        return None
    try:
        result = action()
    except ImportError as exc:
        _record_status(status_rows, module_name, True, "skipped", str(exc))
        return None
    except eligibility_exceptions as exc:
        _record_status(status_rows, module_name, True, "not_eligible", str(exc))
        return None
    except Exception as exc:
        _record_status(status_rows, module_name, True, "failed", str(exc))
        return None
    if result is None:
        _record_status(status_rows, module_name, True, "not_eligible", "No eligible inputs were available for this optional step.")
        return None
    _record_status(status_rows, module_name, True, "completed", "Completed successfully.")
    return result


def _record_status(
    status_rows: list[dict[str, object]],
    module_name: str,
    enabled: bool,
    status: str,
    message: str,
) -> None:
    status_rows.append(
        {
            "module_name": module_name,
            "enabled": enabled,
            "status": status,
            "message": message,
        }
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
