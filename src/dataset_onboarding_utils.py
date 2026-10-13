"""Helpers for onboarding additional processed H5AD datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .metadata_utils import suggest_metadata_columns

DEFAULT_CANDIDATE_GENES = [
    "RHBDF2",
    "ADAM17",
    "TREM2",
    "TYROBP",
    "SYK",
    "TMEM173",
    "TBK1",
    "IRF3",
    "RELA",
    "TNF",
    "IL6R",
    "ITGAX",
    "CLEC7A",
    "CST7",
    "LPL",
]


@dataclass
class DatasetConfigScaffoldResult:
    """Summary of a scaffolded dataset config."""

    dataset_name: str
    input_path: Path
    output_config_path: Path
    matrix_source: str
    diagnosis_column: str | None
    donor_column: str | None
    cluster_column: str | None
    metadata_suggestions_path: Path


def scaffold_dataset_config(
    adata,
    dataset_name: str,
    input_path: str | Path,
    output_config_path: str | Path,
    gene_of_interest: str = "RHBDF2",
    candidate_genes: list[str] | None = None,
) -> DatasetConfigScaffoldResult:
    """Create a dataset-specific YAML config scaffold from a processed H5AD file."""
    input_path = Path(input_path).expanduser().resolve()
    output_config_path = Path(output_config_path).expanduser().resolve()
    output_config_path.parent.mkdir(parents=True, exist_ok=True)

    suggestions = suggest_metadata_columns(adata)
    suggestions_path = output_config_path.with_name(f"{output_config_path.stem}_metadata_suggestions.csv")
    suggestions.to_csv(suggestions_path, index=False)

    diagnosis_column = _best_suggestion(suggestions, "diagnosis")
    donor_column = _best_suggestion(suggestions, "donor")
    cluster_column = _best_suggestion(suggestions, "cluster")
    sample_column = _best_suggestion(suggestions, "sample")

    matrix_source = _infer_matrix_source(adata, gene_of_interest)
    scaffold = {
        "dataset_name": dataset_name,
        "input_path": _display_path(input_path, output_config_path),
        "output_root": "results",
        "report_root": "reports",
        "run_name": f"{dataset_name}_run",
        "overwrite_existing_run": True,
        "gene_of_interest": gene_of_interest,
        "candidate_genes": candidate_genes or DEFAULT_CANDIDATE_GENES,
        "metadata": {
            "diagnosis_column": diagnosis_column,
            "cluster_column": cluster_column,
            "donor_column": donor_column or sample_column,
        },
        "state_definition": {
            "method": "median",
            "quantile": 0.75,
            "new_column_name": f"{gene_of_interest}_state",
            "high_label": f"{gene_of_interest}_high",
            "low_label": f"{gene_of_interest}_low",
        },
        "preprocessing": {
            "matrix_source": matrix_source,
            "layer_name": None,
            "normalize_total": True,
            "target_sum": 10000,
            "log1p": True,
        },
        "differential_expression": {
            "method": "wilcoxon",
            "group_column": f"{gene_of_interest}_state",
            "group_a": f"{gene_of_interest}_high",
            "group_b": f"{gene_of_interest}_low",
            "top_n_export": 100,
        },
        "pseudobulk": {
            "enabled": True,
            "donor_column": donor_column or sample_column,
            "group_column": f"{gene_of_interest}_state",
            "group_a": f"{gene_of_interest}_high",
            "group_b": f"{gene_of_interest}_low",
            "paired_only": True,
            "min_cells_per_sample": 25,
            "min_donors_per_group": 3,
            "normalization_target_sum": 1000000,
            "top_n_export": 100,
            "within_subpopulation": bool(cluster_column),
            "subgroup_column": cluster_column,
            "min_cells_per_subpopulation": 100,
            "within_subpopulation_min_cells_per_sample": None,
            "within_subpopulation_min_donors_per_group": None,
        },
        "enrichment": {
            "enabled": True,
            "organism": "hsapiens",
            "sources": ["GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG", "WP"],
            "user_threshold": 0.05,
            "significance_threshold_method": "g_SCS",
            "domain_scope": "custom",
            "use_dataset_background": True,
            "no_iea": False,
            "max_terms_report": 5,
        },
        "enrichr": {
            "enabled": True,
            "organism": "human",
            "libraries": [
                "MSigDB_Hallmark_2020",
                "Azimuth_Cell_Types_2021",
                "CellMarker_2024",
                "PanglaoDB_Augmented_2021",
            ],
            "cutoff": 0.05,
            "use_dataset_background": True,
            "max_terms_report": 5,
        },
        "evidence": {
            "enabled": True,
            "metadata_association_columns": [
                column
                for column in [diagnosis_column, cluster_column, donor_column or sample_column]
                if column
            ],
            "module_weights": {
                "cell_level": 1.0,
                "pseudobulk": 3.0,
                "continuous": 2.0,
                "detection": 1.0,
                "positive_expression": 1.0,
                "subpopulation_pseudobulk": 2.0,
                "protein_context": 1.0,
                "genetics": 1.5,
                "brain_context": 1.0,
            },
        },
        "continuous_evidence": {
            "enabled": True,
            "donor_column": donor_column or sample_column,
            "top_n_export": 100,
            "min_positive_cells": 25,
        },
        "protein_evidence": {
            "enabled": True,
            "organism_id": 9606,
            "reviewed_only": True,
            "timeout_seconds": 30,
        },
        "genetics_evidence": {
            "enabled": True,
            "disease_focus_label": "Alzheimer's disease",
            "disease_keywords": ["alzheimer", "dementia", "cognitive", "neurodegenerative"],
            "gwas_trait_page_size": 500,
            "gwas_association_page_size": 200,
            "gwas_max_association_pages": 5,
            "opentargets_page_size": 500,
            "timeout_seconds": 30,
        },
        "brain_context_evidence": {
            "enabled": True,
            "context_label": "Alzheimer's disease brain proteomics",
            "expression_atlas_experiment": "E-PROT-61",
            "case_labels": ["Alzheimer's disease"],
            "control_labels": ["normal"],
            "timeout_seconds": 30,
        },
        "plots": {
            "save_umap_if_available": "X_umap" in adata.obsm,
            "save_violin": True,
            "save_dotplot": True,
            "save_matrixplot": True,
            "dpi": 300,
            "violin_rotation": 35,
            "category_label_rotation": 45,
            "max_umap_genes": 4,
        },
    }
    output_config_path.write_text(yaml.safe_dump(scaffold, sort_keys=False), encoding="utf-8")

    return DatasetConfigScaffoldResult(
        dataset_name=dataset_name,
        input_path=input_path,
        output_config_path=output_config_path,
        matrix_source=matrix_source,
        diagnosis_column=diagnosis_column,
        donor_column=donor_column or sample_column,
        cluster_column=cluster_column,
        metadata_suggestions_path=suggestions_path,
    )


def _best_suggestion(suggestions, role: str) -> str | None:
    subset = suggestions[suggestions["suggested_role"] == role]
    if subset.empty:
        return None
    return str(subset.iloc[0]["column_name"])


def _infer_matrix_source(adata, gene_of_interest: str) -> str:
    if gene_of_interest in adata.var_names:
        return "X"
    raw_var_names = getattr(getattr(adata, "raw", None), "var_names", None)
    if raw_var_names is not None and gene_of_interest in raw_var_names:
        return "raw"
    return "X"


def _display_path(input_path: Path, output_config_path: Path) -> str:
    project_root = output_config_path.parent.parent if output_config_path.parent.name == "configs" else output_config_path.parent
    try:
        return str(input_path.relative_to(project_root))
    except ValueError:
        return str(input_path)
