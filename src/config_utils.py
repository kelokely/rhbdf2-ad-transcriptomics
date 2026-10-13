"""Config helpers for the RHBDF2 microglia pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL_FIELDS = [
    "dataset_name",
    "input_path",
    "output_root",
    "report_root",
    "gene_of_interest",
    "candidate_genes",
    "metadata",
    "state_definition",
    "preprocessing",
    "differential_expression",
    "pseudobulk",
    "enrichment",
    "enrichr",
    "evidence",
    "continuous_evidence",
    "protein_evidence",
    "genetics_evidence",
    "brain_context_evidence",
    "pathway_activity_proxy",
    "plots",
]

REQUIRED_STATE_FIELDS = [
    "method",
    "new_column_name",
    "high_label",
    "low_label",
]

REQUIRED_DE_FIELDS = [
    "method",
    "group_column",
    "group_a",
    "group_b",
    "top_n_export",
]

DEFAULTS: dict[str, Any] = {
    "run_name": None,
    "overwrite_existing_run": False,
    "metadata": {
        "diagnosis_column": None,
        "cluster_column": None,
        "donor_column": None,
    },
    "plots": {
        "save_umap_if_available": True,
        "save_violin": True,
        "save_dotplot": True,
        "save_matrixplot": True,
        "dpi": 300,
        "violin_rotation": 35,
        "category_label_rotation": 45,
        "max_umap_genes": 4,
    },
    "state_definition": {
        "method": "median",
        "quantile": 0.75,
        "new_column_name": "RHBDF2_state",
        "high_label": "RHBDF2_high",
        "low_label": "RHBDF2_low",
    },
    "preprocessing": {
        "matrix_source": "X",
        "layer_name": None,
        "normalize_total": True,
        "target_sum": 10000,
        "log1p": True,
    },
    "pseudobulk": {
        "enabled": True,
        "donor_column": None,
        "group_column": None,
        "group_a": None,
        "group_b": None,
        "paired_only": True,
        "min_cells_per_sample": 25,
        "min_donors_per_group": 3,
        "normalization_target_sum": 1000000,
        "top_n_export": 100,
        "within_subpopulation": True,
        "subgroup_column": None,
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
        "metadata_association_columns": [],
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
            "public_proteomics_overlap": 1.5,
        },
    },
    "continuous_evidence": {
        "enabled": True,
        "donor_column": None,
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
        "source_accession": "E-PROT-61",
        "local_matrix_path": None,
        "local_matrix_sheet_name": None,
        "local_matrix_region_column": "Brain Region",
        "local_matrix_group_column": "Group",
        "local_matrix_sample_id_column": "Sample ID",
        "local_matrix_values_are_log2": False,
        "case_labels": ["Alzheimer's disease"],
        "control_labels": ["normal"],
        "timeout_seconds": 30,
    },
    "pathway_activity_proxy": {
        "enabled": True,
        "donor_column": None,
        "min_genes_present": 2,
        "gene_sets": {
            "trem2_tyrobp_syk_response": ["TREM2", "TYROBP", "SYK", "ITGAX", "CLEC7A", "CST7", "LPL", "FCER1G"],
            "inflammatory_nfkb": ["TNF", "IL6R", "RELA", "NFKBIA", "IL1B", "JUN", "FOS", "CCL2"],
            "phagocytosis_myeloid": ["TREM2", "TYROBP", "ITGAX", "CLEC7A", "LPL", "CST7", "CTSB", "AXL"],
            "lysosomal_activation": ["CST7", "LPL", "CTSB", "CTSD", "LAMP1", "LAMP2", "PSAP", "GPNMB"],
            "sting_tbk1_irf3_response": ["TMEM173", "TBK1", "IRF3", "ISG15", "IFIT1", "OAS1", "IFI27", "MX1"],
        },
    },
}

ALLOWED_STATE_METHODS = {"median", "nonzero", "quantile"}
ALLOWED_MATRIX_SOURCES = {"x", "raw", "layer"}
NESTED_SECTION_FIELDS = [
    "metadata",
    "state_definition",
    "preprocessing",
    "differential_expression",
    "pseudobulk",
    "enrichment",
    "enrichr",
    "evidence",
    "continuous_evidence",
    "protein_evidence",
    "genetics_evidence",
    "brain_context_evidence",
    "pathway_activity_proxy",
    "plots",
]


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load, validate, and normalize a YAML configuration file."""
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("Config file must define a YAML mapping at the top level.")

    merged = _deep_merge(DEFAULTS, config)
    validate_config(merged)

    project_root = infer_project_root(config_path)
    merged["input_path"] = str(resolve_project_path(project_root, merged["input_path"]))
    merged["output_root"] = str(resolve_project_path(project_root, merged["output_root"]))
    merged["report_root"] = str(resolve_project_path(project_root, merged["report_root"]))
    local_matrix_path = merged.get("brain_context_evidence", {}).get("local_matrix_path")
    if str(local_matrix_path or "").strip():
        merged["brain_context_evidence"]["local_matrix_path"] = str(
            resolve_project_path(project_root, local_matrix_path)
        )
    merged["_config_path"] = str(config_path)
    merged["_project_root"] = str(project_root)

    if not merged.get("run_name"):
        merged["run_name"] = f"{merged['dataset_name']}_run"

    return merged


def validate_config(config: dict[str, Any]) -> None:
    """Validate required config fields and a few basic value constraints."""
    missing = [field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")

    for field in ["dataset_name", "input_path", "output_root", "report_root", "gene_of_interest"]:
        if not str(config.get(field, "")).strip():
            raise ValueError(f"'{field}' must be a non-empty string.")

    _validate_bool(config.get("overwrite_existing_run", False), "overwrite_existing_run")

    for section_name in NESTED_SECTION_FIELDS:
        _require_mapping(config, section_name)

    if not isinstance(config["candidate_genes"], list) or not config["candidate_genes"]:
        raise ValueError("'candidate_genes' must be a non-empty list of gene symbols.")
    if not all(str(gene).strip() for gene in config["candidate_genes"]):
        raise ValueError("'candidate_genes' must not contain empty values.")

    if not isinstance(config["metadata"], dict):
        raise ValueError("'metadata' must be a mapping of metadata keys to obs column names.")

    state_definition = config["state_definition"]
    missing_state = [field for field in REQUIRED_STATE_FIELDS if field not in state_definition]
    if missing_state:
        raise ValueError(
            "Missing required state_definition fields: "
            + ", ".join(missing_state)
        )

    method = state_definition["method"]
    if method not in ALLOWED_STATE_METHODS:
        raise ValueError(
            f"Unsupported state_definition.method '{method}'. "
            f"Choose from: {sorted(ALLOWED_STATE_METHODS)}"
        )
    for field in ["new_column_name", "high_label", "low_label"]:
        if not str(state_definition.get(field, "")).strip():
            raise ValueError(f"'state_definition.{field}' must be a non-empty string.")

    if method == "quantile":
        quantile = float(state_definition.get("quantile", 0.75))
        if not 0.0 < quantile < 1.0:
            raise ValueError("'state_definition.quantile' must be between 0 and 1.")

    de_config = config["differential_expression"]
    missing_de = [field for field in REQUIRED_DE_FIELDS if field not in de_config]
    if missing_de:
        raise ValueError(
            "Missing required differential_expression fields: "
            + ", ".join(missing_de)
        )
    for field in ["method", "group_column", "group_a", "group_b"]:
        if not str(de_config.get(field, "")).strip():
            raise ValueError(f"'differential_expression.{field}' must be a non-empty string.")

    top_n_export = int(de_config["top_n_export"])
    if top_n_export < 1:
        raise ValueError("'differential_expression.top_n_export' must be at least 1.")
    if str(de_config["group_a"]) == str(de_config["group_b"]):
        raise ValueError("'differential_expression.group_a' and 'group_b' must differ.")

    preprocessing = config["preprocessing"]
    matrix_source = str(preprocessing.get("matrix_source", "X")).strip().lower()
    if matrix_source not in ALLOWED_MATRIX_SOURCES:
        raise ValueError(
            f"Unsupported preprocessing.matrix_source '{preprocessing.get('matrix_source')}'. "
            f"Choose from: {sorted(ALLOWED_MATRIX_SOURCES)}"
        )
    layer_name = preprocessing.get("layer_name")
    if matrix_source == "layer" and not str(layer_name or "").strip():
        raise ValueError(
            "'preprocessing.layer_name' must be set when preprocessing.matrix_source is 'layer'."
        )
    _validate_bool(preprocessing.get("normalize_total", True), "preprocessing.normalize_total")
    _validate_bool(preprocessing.get("log1p", True), "preprocessing.log1p")
    if preprocessing.get("normalize_total", True):
        target_sum = float(preprocessing.get("target_sum", 10000))
        if target_sum <= 0:
            raise ValueError("'preprocessing.target_sum' must be greater than 0.")

    pseudobulk = config["pseudobulk"]
    for field in ["enabled", "paired_only", "within_subpopulation"]:
        _validate_bool(pseudobulk.get(field, DEFAULTS["pseudobulk"][field]), f"pseudobulk.{field}")
    if pseudobulk.get("enabled", True):
        min_cells = int(pseudobulk.get("min_cells_per_sample", 20))
        min_donors = int(pseudobulk.get("min_donors_per_group", 2))
        normalization_target_sum = float(pseudobulk.get("normalization_target_sum", 1000000))
        top_n_pseudobulk = int(pseudobulk.get("top_n_export", 100))
        min_cells_per_subpopulation = int(pseudobulk.get("min_cells_per_subpopulation", 100))
        within_subpopulation_min_cells = pseudobulk.get("within_subpopulation_min_cells_per_sample")
        within_subpopulation_min_donors = pseudobulk.get("within_subpopulation_min_donors_per_group")
        if min_cells < 1:
            raise ValueError("'pseudobulk.min_cells_per_sample' must be at least 1.")
        if min_donors < 1:
            raise ValueError("'pseudobulk.min_donors_per_group' must be at least 1.")
        if normalization_target_sum <= 0:
            raise ValueError("'pseudobulk.normalization_target_sum' must be greater than 0.")
        if top_n_pseudobulk < 1:
            raise ValueError("'pseudobulk.top_n_export' must be at least 1.")
        if min_cells_per_subpopulation < 1:
            raise ValueError("'pseudobulk.min_cells_per_subpopulation' must be at least 1.")
        if within_subpopulation_min_cells is not None and int(within_subpopulation_min_cells) < 1:
            raise ValueError("'pseudobulk.within_subpopulation_min_cells_per_sample' must be at least 1.")
        if within_subpopulation_min_donors is not None and int(within_subpopulation_min_donors) < 1:
            raise ValueError("'pseudobulk.within_subpopulation_min_donors_per_group' must be at least 1.")

    enrichment = config["enrichment"]
    for field in ["enabled", "use_dataset_background", "no_iea"]:
        _validate_bool(enrichment.get(field, DEFAULTS["enrichment"][field]), f"enrichment.{field}")
    if enrichment.get("enabled", True):
        if not isinstance(enrichment.get("sources"), list) or not enrichment["sources"]:
            raise ValueError("'enrichment.sources' must be a non-empty list when enrichment is enabled.")
        user_threshold = float(enrichment.get("user_threshold", 0.05))
        max_terms_report = int(enrichment.get("max_terms_report", 5))
        if not 0.0 < user_threshold <= 1.0:
            raise ValueError("'enrichment.user_threshold' must be between 0 and 1.")
        if max_terms_report < 1:
            raise ValueError("'enrichment.max_terms_report' must be at least 1.")

    enrichr = config["enrichr"]
    for field in ["enabled", "use_dataset_background"]:
        _validate_bool(enrichr.get(field, DEFAULTS["enrichr"][field]), f"enrichr.{field}")
    if enrichr.get("enabled", True):
        if not isinstance(enrichr.get("libraries"), list) or not enrichr["libraries"]:
            raise ValueError("'enrichr.libraries' must be a non-empty list when Enrichr is enabled.")
        cutoff = float(enrichr.get("cutoff", 0.05))
        max_terms_report = int(enrichr.get("max_terms_report", 5))
        if not 0.0 < cutoff <= 1.0:
            raise ValueError("'enrichr.cutoff' must be between 0 and 1.")
        if max_terms_report < 1:
            raise ValueError("'enrichr.max_terms_report' must be at least 1.")

    evidence = config["evidence"]
    _validate_bool(evidence.get("enabled", True), "evidence.enabled")
    if evidence.get("enabled", True):
        if not isinstance(evidence.get("metadata_association_columns"), list):
            raise ValueError("'evidence.metadata_association_columns' must be a list.")
        module_weights = evidence.get("module_weights", DEFAULTS["evidence"]["module_weights"])
        if not isinstance(module_weights, dict) or not module_weights:
            raise ValueError("'evidence.module_weights' must be a non-empty mapping when evidence is enabled.")
        for module_name, weight in module_weights.items():
            try:
                numeric_weight = float(weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"'evidence.module_weights.{module_name}' must be a numeric value."
                ) from exc
            if numeric_weight < 0:
                raise ValueError(
                    f"'evidence.module_weights.{module_name}' must be greater than or equal to 0."
                )

    continuous_evidence = config["continuous_evidence"]
    _validate_bool(continuous_evidence.get("enabled", True), "continuous_evidence.enabled")
    if continuous_evidence.get("enabled", True):
        top_n_export = int(continuous_evidence.get("top_n_export", 100))
        min_positive_cells = int(continuous_evidence.get("min_positive_cells", 25))
        if top_n_export < 1:
            raise ValueError("'continuous_evidence.top_n_export' must be at least 1.")
        if min_positive_cells < 1:
            raise ValueError("'continuous_evidence.min_positive_cells' must be at least 1.")

    protein_evidence = config["protein_evidence"]
    _validate_bool(protein_evidence.get("enabled", True), "protein_evidence.enabled")
    _validate_bool(protein_evidence.get("reviewed_only", True), "protein_evidence.reviewed_only")
    if protein_evidence.get("enabled", True):
        organism_id = int(protein_evidence.get("organism_id", 9606))
        timeout_seconds = float(protein_evidence.get("timeout_seconds", 30))
        if organism_id < 1:
            raise ValueError("'protein_evidence.organism_id' must be a positive integer.")
        if timeout_seconds <= 0:
            raise ValueError("'protein_evidence.timeout_seconds' must be greater than 0.")

    genetics_evidence = config["genetics_evidence"]
    _validate_bool(genetics_evidence.get("enabled", True), "genetics_evidence.enabled")
    if genetics_evidence.get("enabled", True):
        if not str(genetics_evidence.get("disease_focus_label", "")).strip():
            raise ValueError("'genetics_evidence.disease_focus_label' must be a non-empty string.")
        if not isinstance(genetics_evidence.get("disease_keywords"), list) or not genetics_evidence["disease_keywords"]:
            raise ValueError("'genetics_evidence.disease_keywords' must be a non-empty list.")
        timeout_seconds = float(genetics_evidence.get("timeout_seconds", 30))
        if timeout_seconds <= 0:
            raise ValueError("'genetics_evidence.timeout_seconds' must be greater than 0.")

    brain_context_evidence = config["brain_context_evidence"]
    _validate_bool(brain_context_evidence.get("enabled", True), "brain_context_evidence.enabled")
    if brain_context_evidence.get("enabled", True):
        if not str(brain_context_evidence.get("context_label", "")).strip():
            raise ValueError("'brain_context_evidence.context_label' must be a non-empty string.")
        timeout_seconds = float(brain_context_evidence.get("timeout_seconds", 30))
        if timeout_seconds <= 0:
            raise ValueError("'brain_context_evidence.timeout_seconds' must be greater than 0.")
        _validate_bool(
            brain_context_evidence.get("local_matrix_values_are_log2", False),
            "brain_context_evidence.local_matrix_values_are_log2",
        )
        local_matrix_path = str(brain_context_evidence.get("local_matrix_path") or "").strip()
        if local_matrix_path:
            for field_name in [
                "local_matrix_region_column",
                "local_matrix_group_column",
                "local_matrix_sample_id_column",
            ]:
                if not str(brain_context_evidence.get(field_name, "")).strip():
                    raise ValueError(
                        f"'brain_context_evidence.{field_name}' must be a non-empty string when local_matrix_path is set."
                    )
        else:
            experiment_accession = str(brain_context_evidence.get("expression_atlas_experiment", "E-PROT-61")).strip()
            if not experiment_accession:
                raise ValueError(
                    "'brain_context_evidence.expression_atlas_experiment' must be set when enabled and local_matrix_path is not provided."
                )
        for field_name in ["case_labels", "control_labels"]:
            values = brain_context_evidence.get(field_name)
            if not isinstance(values, list) or not values or not all(str(value).strip() for value in values):
                raise ValueError(
                    f"'brain_context_evidence.{field_name}' must be a non-empty list of strings when enabled."
                )

    pathway_activity_proxy = config["pathway_activity_proxy"]
    _validate_bool(pathway_activity_proxy.get("enabled", True), "pathway_activity_proxy.enabled")
    if pathway_activity_proxy.get("enabled", True):
        min_genes_present = int(pathway_activity_proxy.get("min_genes_present", 2))
        if min_genes_present < 1:
            raise ValueError("'pathway_activity_proxy.min_genes_present' must be at least 1.")
        gene_sets = pathway_activity_proxy.get("gene_sets")
        if not isinstance(gene_sets, dict) or not gene_sets:
            raise ValueError("'pathway_activity_proxy.gene_sets' must be a non-empty mapping.")
        for module_name, genes in gene_sets.items():
            if not isinstance(genes, list) or not genes:
                raise ValueError(
                    f"'pathway_activity_proxy.gene_sets.{module_name}' must be a non-empty list."
                )

    plots = config["plots"]
    for field in ["save_umap_if_available", "save_violin", "save_dotplot", "save_matrixplot"]:
        _validate_bool(plots.get(field, DEFAULTS["plots"][field]), f"plots.{field}")
    dpi = int(plots.get("dpi", DEFAULTS["plots"]["dpi"]))
    max_umap_genes = int(plots.get("max_umap_genes", DEFAULTS["plots"]["max_umap_genes"]))
    violin_rotation = float(plots.get("violin_rotation", DEFAULTS["plots"]["violin_rotation"]))
    category_label_rotation = float(
        plots.get("category_label_rotation", DEFAULTS["plots"]["category_label_rotation"])
    )
    if dpi < 72:
        raise ValueError("'plots.dpi' must be at least 72.")
    if max_umap_genes < 1:
        raise ValueError("'plots.max_umap_genes' must be at least 1.")
    if not 0.0 <= violin_rotation <= 90.0:
        raise ValueError("'plots.violin_rotation' must be between 0 and 90.")
    if not 0.0 <= category_label_rotation <= 90.0:
        raise ValueError("'plots.category_label_rotation' must be between 0 and 90.")


def infer_project_root(config_path: Path) -> Path:
    """Infer the project root from a config path."""
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    """Resolve a path value relative to the inferred project root."""
    path_value = Path(value).expanduser()
    if path_value.is_absolute():
        return path_value
    return (project_root / path_value).resolve()


def get_config_path(config: dict[str, Any]) -> Path:
    """Return the normalized config file path."""
    return Path(config["_config_path"])


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the config without internal helper keys."""
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _require_mapping(config: dict[str, Any], section_name: str) -> None:
    if not isinstance(config.get(section_name), dict):
        raise ValueError(f"'{section_name}' must be a mapping.")


def _validate_bool(value: Any, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"'{field_name}' must be a boolean.")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, dict):
            merged[key] = _deep_merge(value, {})
        else:
            merged[key] = value
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
