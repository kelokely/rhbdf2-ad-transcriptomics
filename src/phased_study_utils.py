"""Helpers for multi-run phased study integration across AD and PD contexts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import shutil

import pandas as pd
import yaml

from .config_utils import infer_project_root, resolve_project_path


DEFAULT_PHASED_STUDY_CONFIG: dict[str, Any] = {
    "overwrite_existing_output": True,
    "module_delta_support_threshold": 0.02,
    "contexts": {},
}

TRANSCRIPT_LAYER_WEIGHTS = {
    "pseudobulk_significant_fdr_0_05": 4.0,
    "continuous_significant_fdr_0_05": 3.0,
    "cell_level_significant_fdr_0_05": 1.0,
    "subpopulation_supported": 2.0,
    "detection_significant_fdr_0_05": 1.0,
    "positive_expression_significant_fdr_0_05": 1.0,
}

NETWORK_SCORE_WEIGHTS = {
    "protein_context_supported_fraction": 1.0,
    "public_proteomics_overlap_supported_fraction": 1.0,
    "genetics_supported_fraction": 1.25,
    "brain_context_supported_fraction": 0.75,
}


@dataclass
class StudyRunSpec:
    """One finished pipeline run included in the phased study."""

    context_name: str
    run_label: str
    run_dir: Path
    min_replicated_continuous_runs: int


@dataclass
class StudyPaths:
    """Filesystem outputs for the phased integration layer."""

    study_name: str
    output_dir: Path
    phase0_dir: Path
    phase1_dir: Path
    phase2_dir: Path
    phase3_dir: Path
    report_path: Path
    config_snapshot_path: Path


@dataclass
class RunArtifacts:
    """Finished-run inputs required for phased integration."""

    spec: StudyRunSpec
    dataset_name: str
    anchor_gene: str
    candidate_summary: pd.DataFrame
    activity_proxy_summary: pd.DataFrame | None
    pathway_gene_sets: dict[str, list[str]]
    network_edges: pd.DataFrame | None
    state_summary: pd.DataFrame


def load_phased_study_config(config_path: str | Path) -> dict[str, Any]:
    """Load and normalize a phased-study config."""
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Phased-study config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Phased-study config must define a YAML mapping at the top level.")

    config = _deep_merge(DEFAULT_PHASED_STUDY_CONFIG, raw)
    _validate_phased_study_config(config)

    project_root = infer_project_root(config_path)
    config["output_dir"] = str(resolve_project_path(project_root, config["output_dir"]))
    report_path = config.get("report_path")
    if str(report_path or "").strip():
        config["report_path"] = str(resolve_project_path(project_root, report_path))
    else:
        config["report_path"] = str((Path(config["output_dir"]) / "phased_study_summary.md").resolve())

    for context_name, context in config["contexts"].items():
        normalized_runs: list[dict[str, Any]] = []
        for index, run in enumerate(context["runs"], start=1):
            if isinstance(run, str):
                run = {"path": run}
            run_mapping = dict(run)
            run_mapping["path"] = str(resolve_project_path(project_root, run_mapping["path"]))
            if not str(run_mapping.get("label", "")).strip():
                run_mapping["label"] = f"{context_name}_run_{index}"
            normalized_runs.append(run_mapping)
        context["runs"] = normalized_runs

    config["_config_path"] = str(config_path)
    config["_project_root"] = str(project_root)
    return config


def create_study_paths(config: dict[str, Any]) -> StudyPaths:
    """Prepare deterministic output folders for a phased study."""
    output_dir = Path(config["output_dir"])
    overwrite_existing = bool(config.get("overwrite_existing_output", True))

    if output_dir.exists() and overwrite_existing:
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    phase0_dir = output_dir / "phase0_run_inventory"
    phase1_dir = output_dir / "phase1_core_vs_periphery"
    phase2_dir = output_dir / "phase2_module_integration"
    phase3_dir = output_dir / "phase3_network_prioritization"
    for path in [output_dir, phase0_dir, phase1_dir, phase2_dir, phase3_dir]:
        path.mkdir(parents=True, exist_ok=True)

    report_path = Path(config["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    config_snapshot_path = output_dir / "phased_study_config_used.yaml"

    return StudyPaths(
        study_name=str(config["study_name"]),
        output_dir=output_dir,
        phase0_dir=phase0_dir,
        phase1_dir=phase1_dir,
        phase2_dir=phase2_dir,
        phase3_dir=phase3_dir,
        report_path=report_path,
        config_snapshot_path=config_snapshot_path,
    )


def save_phased_study_config_snapshot(config: dict[str, Any], destination: Path) -> Path:
    """Persist the effective phased-study config."""
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    destination.write_text(yaml.safe_dump(public, sort_keys=False), encoding="utf-8")
    return destination


def build_run_specs(config: dict[str, Any]) -> list[StudyRunSpec]:
    """Expand the config into run specifications."""
    specs: list[StudyRunSpec] = []
    for context_name, context in config["contexts"].items():
        min_continuous_runs = int(context.get("min_replicated_continuous_runs", 1))
        for run in context["runs"]:
            specs.append(
                StudyRunSpec(
                    context_name=context_name,
                    run_label=str(run["label"]),
                    run_dir=Path(run["path"]),
                    min_replicated_continuous_runs=min_continuous_runs,
                )
            )
    return specs


def load_run_artifacts(specs: list[StudyRunSpec]) -> list[RunArtifacts]:
    """Load the finished-run outputs required for phased integration."""
    artifacts: list[RunArtifacts] = []
    for spec in specs:
        run_dir = spec.run_dir
        config = _load_yaml(run_dir / "summaries" / "config_used.yaml")
        candidate_summary = pd.read_csv(run_dir / "tables" / "candidate_evidence_summary.csv")
        state_summary = pd.read_csv(run_dir / "summaries" / "state_summary.csv")
        activity_proxy_summary = _load_optional_csv(run_dir / "activity_proxies" / "pathway_activity_proxy_summary.csv")
        network_edges = _load_optional_csv(run_dir / "external_evidence" / "protein_candidate_network_edges.csv")
        pathway_gene_sets = _load_gene_sets(run_dir / "activity_proxies" / "pathway_activity_proxy_gene_sets.yaml")
        artifacts.append(
            RunArtifacts(
                spec=spec,
                dataset_name=str(config.get("dataset_name", spec.run_label)),
                anchor_gene=str(config.get("gene_of_interest", "RHBDF2")),
                candidate_summary=candidate_summary,
                activity_proxy_summary=activity_proxy_summary,
                pathway_gene_sets=pathway_gene_sets,
                network_edges=network_edges,
                state_summary=state_summary,
            )
        )
    return artifacts


def build_run_manifest(artifacts: list[RunArtifacts], output_path: Path) -> pd.DataFrame:
    """Summarize which runs contribute to the phased study."""
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        total_cells = int(artifact.state_summary["count"].sum()) if "count" in artifact.state_summary.columns else 0
        high_fraction = derive_high_fraction(artifact.state_summary)
        rows.append(
            {
                "context_name": artifact.spec.context_name,
                "run_label": artifact.spec.run_label,
                "dataset_name": artifact.dataset_name,
                "run_dir": str(artifact.spec.run_dir),
                "cells_in_state_summary": total_cells,
                "rhbdf2_high_fraction": high_fraction,
                "n_candidate_genes": int(artifact.candidate_summary["gene"].nunique()),
                "has_activity_proxy_summary": artifact.activity_proxy_summary is not None,
                "has_network_edges": artifact.network_edges is not None,
            }
        )
    manifest = pd.DataFrame(rows).sort_values(["context_name", "run_label"]).reset_index(drop=True)
    manifest.to_csv(output_path, index=False)
    return manifest


def build_candidate_phase_outputs(
    artifacts: list[RunArtifacts],
    backbone_context: str,
    extension_context: str,
    anchor_gene: str,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Build the cross-context candidate core-vs-periphery tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[pd.DataFrame] = []
    context_min_continuous_runs = {
        artifact.spec.context_name: artifact.spec.min_replicated_continuous_runs
        for artifact in artifacts
    }

    for artifact in artifacts:
        frame = artifact.candidate_summary.copy()
        frame["context_name"] = artifact.spec.context_name
        frame["run_label"] = artifact.spec.run_label
        frame["dataset_name"] = artifact.dataset_name
        run_rows.append(frame)

    run_level = pd.concat(run_rows, ignore_index=True, sort=False)
    run_level.to_csv(output_dir / "candidate_run_level_support_matrix.csv", index=False)

    context_rows: list[dict[str, Any]] = []
    for context_name, frame in run_level.groupby("context_name", observed=True):
        for gene, gene_frame in frame.groupby("gene", observed=True):
            context_rows.append(
                _summarize_candidate_context(
                    gene=gene,
                    context_name=context_name,
                    frame=gene_frame,
                    min_replicated_continuous_runs=context_min_continuous_runs.get(context_name, 1),
                )
            )

    context_summary = pd.DataFrame(context_rows).sort_values(["context_name", "transcript_support_score", "gene"], ascending=[True, False, True])
    context_summary.to_csv(output_dir / "candidate_context_summary.csv", index=False)

    merged = _merge_context_candidate_views(
        context_summary=context_summary,
        backbone_context=backbone_context,
        extension_context=extension_context,
        anchor_gene=anchor_gene,
    )
    merged.to_csv(output_dir / "candidate_core_vs_context_summary.csv", index=False)

    top_candidates = merged.sort_values(
        ["classification_rank", "shared_core_score", f"{backbone_context}_transcript_support_score", f"{extension_context}_transcript_support_score", "gene"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    top_candidates.to_csv(output_dir / "candidate_core_vs_context_ranked.csv", index=False)
    classification_counts = (
        top_candidates["phase1_classification"]
        .fillna("unclassified")
        .value_counts(dropna=False)
        .rename_axis("phase1_classification")
        .reset_index(name="n_genes")
    )
    classification_counts.to_csv(output_dir / "candidate_phase1_classification_counts.csv", index=False)
    phase1_summary_path = _write_phase1_summary(
        ranked_candidates=top_candidates,
        output_dir=output_dir,
        backbone_context=backbone_context,
        extension_context=extension_context,
        anchor_gene=anchor_gene,
    )
    figure_paths = _write_phase1_figures(
        ranked_candidates=top_candidates,
        output_dir=output_dir,
        backbone_context=backbone_context,
        extension_context=extension_context,
        anchor_gene=anchor_gene,
    )
    return {
        "run_level": run_level,
        "context_summary": context_summary,
        "merged": merged,
        "ranked": top_candidates,
        "classification_counts": classification_counts,
        "summary_path": phase1_summary_path,
        "figure_paths": figure_paths,
    }


def build_module_phase_outputs(
    artifacts: list[RunArtifacts],
    backbone_context: str,
    extension_context: str,
    output_dir: Path,
    delta_support_threshold: float,
) -> dict[str, pd.DataFrame]:
    """Build the cross-context pathway-module comparison tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[pd.DataFrame] = []
    module_gene_sets: dict[str, list[str]] = {}
    for artifact in artifacts:
        if artifact.activity_proxy_summary is None:
            continue
        frame = artifact.activity_proxy_summary.copy()
        frame["context_name"] = artifact.spec.context_name
        frame["run_label"] = artifact.spec.run_label
        frame["dataset_name"] = artifact.dataset_name
        run_rows.append(frame)
        for module_name, genes in artifact.pathway_gene_sets.items():
            module_gene_sets.setdefault(module_name, genes)

    if not run_rows:
        empty = pd.DataFrame()
        empty.to_csv(output_dir / "module_run_level_support_matrix.csv", index=False)
        empty.to_csv(output_dir / "module_context_summary.csv", index=False)
        empty.to_csv(output_dir / "module_core_vs_context_summary.csv", index=False)
        return {"run_level": empty, "context_summary": empty, "merged": empty, "gene_sets": pd.DataFrame()}

    run_level = pd.concat(run_rows, ignore_index=True, sort=False)
    run_level.to_csv(output_dir / "module_run_level_support_matrix.csv", index=False)

    context_rows: list[dict[str, Any]] = []
    for context_name, frame in run_level.groupby("context_name", observed=True):
        for module_name, module_frame in frame.groupby("module_name", observed=True):
            context_rows.append(
                _summarize_module_context(
                    module_name=module_name,
                    context_name=context_name,
                    frame=module_frame,
                    delta_support_threshold=delta_support_threshold,
                )
            )

    context_summary = pd.DataFrame(context_rows).sort_values(["context_name", "module_support_score", "module_name"], ascending=[True, False, True])
    context_summary.to_csv(output_dir / "module_context_summary.csv", index=False)

    merged = _merge_context_module_views(
        context_summary=context_summary,
        backbone_context=backbone_context,
        extension_context=extension_context,
    )
    merged.to_csv(output_dir / "module_core_vs_context_summary.csv", index=False)

    gene_set_rows = [
        {"module_name": module_name, "genes": ";".join(genes)}
        for module_name, genes in sorted(module_gene_sets.items())
    ]
    gene_sets_frame = pd.DataFrame(gene_set_rows)
    gene_sets_frame.to_csv(output_dir / "module_gene_sets.csv", index=False)
    classification_counts = (
        merged["phase2_classification"]
        .fillna("unclassified")
        .value_counts(dropna=False)
        .rename_axis("phase2_classification")
        .reset_index(name="n_modules")
    )
    classification_counts.to_csv(output_dir / "module_phase2_classification_counts.csv", index=False)
    phase2_summary_path = _write_phase2_summary(
        merged=merged,
        run_level=run_level,
        classification_counts=classification_counts,
        output_dir=output_dir,
        backbone_context=backbone_context,
        extension_context=extension_context,
    )
    figure_paths = _write_phase2_figures(
        merged=merged,
        run_level=run_level,
        output_dir=output_dir,
        backbone_context=backbone_context,
        extension_context=extension_context,
    )
    return {
        "run_level": run_level,
        "context_summary": context_summary,
        "merged": merged,
        "gene_sets": gene_sets_frame,
        "classification_counts": classification_counts,
        "summary_path": phase2_summary_path,
        "figure_paths": figure_paths,
    }


def build_network_phase_outputs(
    artifacts: list[RunArtifacts],
    candidate_phase: dict[str, pd.DataFrame],
    module_phase: dict[str, pd.DataFrame],
    backbone_context: str,
    extension_context: str,
    anchor_gene: str,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Build the heuristic network-informed prioritization tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_summary = candidate_phase["ranked"].copy()
    module_summary = module_phase.get("merged", pd.DataFrame()).copy()

    gene_to_modules = _build_gene_to_modules(module_phase.get("gene_sets", pd.DataFrame()))
    module_strength_lookup = _build_module_strength_lookup(module_summary, backbone_context, extension_context)
    candidate_summary["module_memberships"] = candidate_summary["gene"].map(lambda gene: ";".join(gene_to_modules.get(gene, [])))
    candidate_summary[f"{backbone_context}_module_support_score"] = candidate_summary["gene"].map(
        lambda gene: sum(module_strength_lookup.get((module_name, backbone_context), 0.0) for module_name in gene_to_modules.get(gene, []))
    )
    candidate_summary[f"{extension_context}_module_support_score"] = candidate_summary["gene"].map(
        lambda gene: sum(module_strength_lookup.get((module_name, extension_context), 0.0) for module_name in gene_to_modules.get(gene, []))
    )

    candidate_summary["network_context_score"] = candidate_summary.apply(_compute_network_context_score, axis=1)
    candidate_summary["external_context_score"] = candidate_summary.apply(_compute_external_context_score, axis=1)
    candidate_summary[f"{backbone_context}_priority_score"] = (
        candidate_summary[f"{backbone_context}_transcript_support_score"].fillna(0.0)
        + 0.5 * candidate_summary[f"{backbone_context}_module_support_score"].fillna(0.0)
        + candidate_summary["network_context_score"]
        + candidate_summary["external_context_score"]
    )
    candidate_summary[f"{extension_context}_priority_score"] = (
        candidate_summary[f"{extension_context}_transcript_support_score"].fillna(0.0)
        + 0.5 * candidate_summary[f"{extension_context}_module_support_score"].fillna(0.0)
        + candidate_summary["network_context_score"]
        + candidate_summary["external_context_score"]
    )
    candidate_summary["shared_priority_score"] = (
        candidate_summary[[
            f"{backbone_context}_transcript_support_score",
            f"{extension_context}_transcript_support_score",
        ]].min(axis=1, skipna=True).fillna(0.0)
        + 0.5
        * candidate_summary[[
            f"{backbone_context}_module_support_score",
            f"{extension_context}_module_support_score",
        ]].min(axis=1, skipna=True).fillna(0.0)
        + candidate_summary["network_context_score"]
        + candidate_summary["external_context_score"]
    )
    candidate_summary["network_role"] = candidate_summary.apply(
        lambda row: _classify_network_role(
            row=row,
            backbone_context=backbone_context,
            extension_context=extension_context,
            anchor_gene=anchor_gene,
        ),
        axis=1,
    )

    prioritized = candidate_summary.sort_values(
        ["shared_priority_score", f"{backbone_context}_priority_score", f"{extension_context}_priority_score", "gene"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    prioritized.to_csv(output_dir / "network_priority_summary.csv", index=False)
    role_counts = (
        prioritized["network_role"]
        .fillna("unclassified")
        .value_counts(dropna=False)
        .rename_axis("network_role")
        .reset_index(name="n_genes")
    )
    role_counts.to_csv(output_dir / "network_role_counts.csv", index=False)

    combined_edges = _combine_network_edges(artifacts)
    prioritized_genes = prioritized.loc[
        prioritized["network_role"] != "lower_priority",
        "gene",
    ].astype(str).tolist()
    if anchor_gene not in prioritized_genes:
        prioritized_genes.append(anchor_gene)
    prioritized_edges = _filter_prioritized_edges(combined_edges, prioritized_genes)
    prioritized_edges.to_csv(output_dir / "prioritized_network_edges.csv", index=False)
    phase3_summary_path = _write_phase3_summary(
        prioritized=prioritized,
        prioritized_edges=prioritized_edges,
        role_counts=role_counts,
        output_dir=output_dir,
        backbone_context=backbone_context,
        extension_context=extension_context,
        anchor_gene=anchor_gene,
    )
    figure_paths = _write_phase3_figures(
        prioritized=prioritized,
        output_dir=output_dir,
        backbone_context=backbone_context,
        extension_context=extension_context,
        anchor_gene=anchor_gene,
    )
    return {
        "priority": prioritized,
        "edges": prioritized_edges,
        "role_counts": role_counts,
        "summary_path": phase3_summary_path,
        "figure_paths": figure_paths,
    }


def write_phased_study_report(
    config: dict[str, Any],
    paths: StudyPaths,
    manifest: pd.DataFrame,
    candidate_phase: dict[str, pd.DataFrame],
    module_phase: dict[str, pd.DataFrame],
    network_phase: dict[str, pd.DataFrame],
) -> Path:
    """Write a compact markdown summary for the phased study."""
    backbone_context = str(config["backbone_context"])
    extension_context = str(config["extension_context"])
    ranked_candidates = candidate_phase["ranked"]
    classification_counts = candidate_phase.get("classification_counts", pd.DataFrame())
    module_summary = module_phase.get("merged", pd.DataFrame())
    module_classification_counts = module_phase.get("classification_counts", pd.DataFrame())
    network_summary = network_phase["priority"]
    network_role_counts = network_phase.get("role_counts", pd.DataFrame())

    top_anchor = ranked_candidates[ranked_candidates["phase1_classification"] == "shared_core_anchor"].head(1)
    top_shared = ranked_candidates[ranked_candidates["phase1_classification"] == "shared_core"].head(5)
    top_backbone = ranked_candidates[
        ranked_candidates["phase1_classification"].isin(["backbone_specific_periphery", "backbone_supported_only"])
    ].head(5)
    top_extension = ranked_candidates[
        ranked_candidates["phase1_classification"].isin(["extension_specific_periphery", "extension_inverse_periphery", "extension_supported_only"])
    ].head(5)
    top_network = network_summary.head(10)

    lines = [
        f"# Phased Study Summary: {config['study_name']}",
        "",
        "## Study Scope",
        f"- Backbone context: `{backbone_context}`",
        f"- Extension context: `{extension_context}`",
        f"- Anchor gene: `{config['anchor_gene']}`",
        f"- Config snapshot: `{paths.config_snapshot_path}`",
        "",
        "## Included Runs",
        _markdown_table(manifest),
        "",
        "## Phase 1: Candidate Core vs Context-Specific Support",
        "### Shared Core Anchor",
        _markdown_table(_subset_columns(top_anchor, [
            "gene",
            "phase1_classification",
            "shared_core_score",
            f"{backbone_context}_transcript_support_score",
            f"{extension_context}_transcript_support_score",
            f"{backbone_context}_dominant_direction",
            f"{extension_context}_dominant_direction",
        ])),
        "",
        "### Shared Core Candidates",
        _markdown_table(_subset_columns(top_shared, [
            "gene",
            "phase1_classification",
            "shared_core_score",
            f"{backbone_context}_transcript_support_score",
            f"{extension_context}_transcript_support_score",
            f"{backbone_context}_dominant_direction",
            f"{extension_context}_dominant_direction",
        ])),
        "",
        f"### {backbone_context}-Enriched Candidates",
        _markdown_table(_subset_columns(top_backbone, [
            "gene",
            "phase1_classification",
            f"{backbone_context}_transcript_support_score",
            f"{extension_context}_transcript_support_score",
            f"{backbone_context}_dominant_direction",
            f"{extension_context}_dominant_direction",
        ])),
        "",
        f"### {extension_context}-Enriched Candidates",
        _markdown_table(_subset_columns(top_extension, [
            "gene",
            "phase1_classification",
            f"{backbone_context}_transcript_support_score",
            f"{extension_context}_transcript_support_score",
            f"{backbone_context}_dominant_direction",
            f"{extension_context}_dominant_direction",
        ])),
        "",
        "### Phase 1 Classification Counts",
        _markdown_table(classification_counts),
        "",
        "### Phase 1 Files",
        f"- `{candidate_phase.get('summary_path', paths.phase1_dir / 'phase1_summary.md')}`",
        *[
            f"- `{path}`"
            for path in candidate_phase.get("figure_paths", [])
        ],
        "",
        "## Phase 2: Module-Level Integration",
    ]
    if module_summary.empty:
        lines.append("- No module summaries were available.")
    else:
        lines.append(
            _markdown_table(
                _subset_columns(
                    module_summary.head(10),
                    [
                        "module_name",
                        "phase2_classification",
                        f"{backbone_context}_module_support_score",
                        f"{extension_context}_module_support_score",
                        f"{backbone_context}_dominant_direction",
                        f"{extension_context}_dominant_direction",
                    ],
                )
            )
        )
        lines.extend(
            [
                "",
                "### Phase 2 Classification Counts",
                _markdown_table(module_classification_counts),
                "",
                "### Phase 2 Files",
                f"- `{module_phase.get('summary_path', paths.phase2_dir / 'phase2_summary.md')}`",
                *[
                    f"- `{path}`"
                    for path in module_phase.get("figure_paths", [])
                ],
            ]
        )

    lines.extend(
        [
            "",
            "## Phase 3: Network-Informed Prioritization",
            _markdown_table(
                _subset_columns(
                    top_network,
                    [
                        "gene",
                        "network_role",
                        "phase1_classification",
                        "shared_priority_score",
                        f"{backbone_context}_priority_score",
                        f"{extension_context}_priority_score",
                    ],
                )
            ),
            "",
            "### Phase 3 Role Counts",
            _markdown_table(network_role_counts),
            "",
            "### Phase 3 Files",
            f"- `{network_phase.get('summary_path', paths.phase3_dir / 'phase3_summary.md')}`",
            *[
                f"- `{path}`"
                for path in network_phase.get("figure_paths", [])
            ],
            "",
            "## Output Files",
            f"- `{paths.output_dir / 'phase0_run_inventory' / 'run_manifest.csv'}`",
            f"- `{paths.output_dir / 'phase1_core_vs_periphery' / 'candidate_core_vs_context_ranked.csv'}`",
            f"- `{paths.output_dir / 'phase2_module_integration' / 'module_core_vs_context_summary.csv'}`",
            f"- `{paths.output_dir / 'phase3_network_prioritization' / 'network_priority_summary.csv'}`",
            f"- `{paths.output_dir / 'phase3_network_prioritization' / 'prioritized_network_edges.csv'}`",
            "",
            "## Caveat",
            "- Phase 3 is a prioritization layer built from transcript support, module behavior, and external context. It is intended to rank follow-up targets, not to prove mechanism.",
            "",
        ]
    )

    report_text = "\n".join(lines)
    (paths.output_dir / "phased_study_summary.md").write_text(report_text, encoding="utf-8")
    paths.report_path.write_text(report_text, encoding="utf-8")
    return paths.report_path


def run_phased_study(config_path: str | Path) -> dict[str, Any]:
    """Execute the AD/PD phased integration and write outputs."""
    config = load_phased_study_config(config_path)
    paths = create_study_paths(config)
    save_phased_study_config_snapshot(config, paths.config_snapshot_path)

    specs = build_run_specs(config)
    artifacts = load_run_artifacts(specs)
    manifest = build_run_manifest(artifacts, paths.phase0_dir / "run_manifest.csv")

    anchor_gene = str(config["anchor_gene"])
    candidate_phase = build_candidate_phase_outputs(
        artifacts=artifacts,
        backbone_context=str(config["backbone_context"]),
        extension_context=str(config["extension_context"]),
        anchor_gene=anchor_gene,
        output_dir=paths.phase1_dir,
    )
    module_phase = build_module_phase_outputs(
        artifacts=artifacts,
        backbone_context=str(config["backbone_context"]),
        extension_context=str(config["extension_context"]),
        output_dir=paths.phase2_dir,
        delta_support_threshold=float(config.get("module_delta_support_threshold", 0.02)),
    )
    network_phase = build_network_phase_outputs(
        artifacts=artifacts,
        candidate_phase=candidate_phase,
        module_phase=module_phase,
        backbone_context=str(config["backbone_context"]),
        extension_context=str(config["extension_context"]),
        anchor_gene=anchor_gene,
        output_dir=paths.phase3_dir,
    )
    report_path = write_phased_study_report(
        config=config,
        paths=paths,
        manifest=manifest,
        candidate_phase=candidate_phase,
        module_phase=module_phase,
        network_phase=network_phase,
    )
    return {
        "config": config,
        "paths": paths,
        "manifest": manifest,
        "candidate_phase": candidate_phase,
        "module_phase": module_phase,
        "network_phase": network_phase,
        "report_path": report_path,
    }


def derive_high_fraction(state_summary: pd.DataFrame) -> float:
    """Derive the fraction of cells assigned to the high state."""
    if state_summary.empty or "count" not in state_summary.columns:
        return float("nan")
    state_column = next((column for column in state_summary.columns if column != "count"), None)
    if state_column is None:
        return float("nan")
    total = float(state_summary["count"].sum())
    high_count = float(
        state_summary.loc[
            state_summary[state_column].astype(str).str.contains("high", case=False, na=False),
            "count",
        ].sum()
    )
    return high_count / total if total else float("nan")


def _summarize_candidate_context(
    gene: str,
    context_name: str,
    frame: pd.DataFrame,
    min_replicated_continuous_runs: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "gene": gene,
        "context_name": context_name,
        "n_runs_present": int(len(frame)),
    }

    bool_columns = [
        "cell_level_significant_fdr_0_05",
        "continuous_significant_fdr_0_05",
        "pseudobulk_significant_fdr_0_05",
        "subpopulation_supported",
        "detection_significant_fdr_0_05",
        "positive_expression_significant_fdr_0_05",
        "protein_context_supported",
        "genetics_supported",
        "brain_context_supported",
        "public_proteomics_overlap_supported",
    ]
    for column in bool_columns:
        series = _bool_series(frame.get(column))
        summary[f"{column}_count"] = int(series.sum())
        summary[f"{column}_fraction"] = float(series.mean()) if len(series) else 0.0

    metric_columns = [
        "cell_level_logfoldchange",
        "continuous_slope_per_log1p_rhbdf2",
        "pseudobulk_effect_log1p_cpm",
        "subpopulation_pseudobulk_mean_effect_log1p_cpm",
        "weighted_transcript_evidence_score",
        "weighted_context_evidence_score",
        "weighted_total_evidence_score",
        "reactome_pathway_count",
        "annotated_ptm_feature_count",
        "candidate_panel_interaction_count",
        "candidate_panel_ptm_out_count",
        "candidate_panel_ptm_in_count",
        "gwas_association_count",
        "opentargets_matching_disease_count",
    ]
    for column in metric_columns:
        if column in frame.columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            summary[f"mean_{column}"] = float(numeric.mean()) if numeric.notna().any() else float("nan")

    transcript_score = 0.0
    for column, weight in TRANSCRIPT_LAYER_WEIGHTS.items():
        transcript_score += weight * float(summary.get(f"{column}_count", 0))
    summary["transcript_support_score"] = transcript_score
    summary["strong_transcript_supported"] = bool(
        summary.get("pseudobulk_significant_fdr_0_05_count", 0) >= 1
        or summary.get("continuous_significant_fdr_0_05_count", 0) >= int(min_replicated_continuous_runs)
    )
    summary["any_transcript_supported"] = bool(
        summary["strong_transcript_supported"]
        or summary.get("cell_level_significant_fdr_0_05_count", 0) >= 1
        or summary.get("subpopulation_supported_count", 0) >= 1
        or summary.get("detection_significant_fdr_0_05_count", 0) >= 1
        or summary.get("positive_expression_significant_fdr_0_05_count", 0) >= 1
    )
    direction_score = _direction_score(
        frame=frame,
        metric_to_support={
            "pseudobulk_effect_log1p_cpm": ("pseudobulk_significant_fdr_0_05", 4.0),
            "continuous_slope_per_log1p_rhbdf2": ("continuous_significant_fdr_0_05", 3.0),
            "cell_level_logfoldchange": ("cell_level_significant_fdr_0_05", 1.0),
            "subpopulation_pseudobulk_mean_effect_log1p_cpm": ("subpopulation_supported", 2.0),
        },
    )
    summary["direction_score"] = direction_score
    summary["dominant_direction"] = _direction_label(direction_score)
    return summary


def _merge_context_candidate_views(
    context_summary: pd.DataFrame,
    backbone_context: str,
    extension_context: str,
    anchor_gene: str,
) -> pd.DataFrame:
    backbone = _rename_context_columns(
        context_summary[context_summary["context_name"] == backbone_context],
        context_name=backbone_context,
    )
    extension = _rename_context_columns(
        context_summary[context_summary["context_name"] == extension_context],
        context_name=extension_context,
    )
    merged = backbone.merge(extension, on="gene", how="outer")

    merged["shared_core_score"] = merged[[
        f"{backbone_context}_transcript_support_score",
        f"{extension_context}_transcript_support_score",
    ]].min(axis=1, skipna=True).fillna(0.0)
    merged["phase1_classification"] = merged.apply(
        lambda row: _classify_candidate_core_vs_context(
            row=row,
            backbone_context=backbone_context,
            extension_context=extension_context,
            anchor_gene=anchor_gene,
        ),
        axis=1,
    )
    merged["classification_rank"] = merged["phase1_classification"].map(
        {
            "shared_core_anchor": 0,
            "shared_core": 1,
            "backbone_specific_periphery": 2,
            "extension_specific_periphery": 3,
            "extension_inverse_periphery": 4,
            "shared_periphery": 5,
            "context_unstable": 6,
            "backbone_supported_only": 7,
            "extension_supported_only": 8,
            "unsupported": 9,
        }
    ).fillna(99)
    return merged


def _classify_candidate_core_vs_context(
    row: pd.Series,
    backbone_context: str,
    extension_context: str,
    anchor_gene: str,
) -> str:
    if str(row.get("gene")) == anchor_gene:
        return "shared_core_anchor"

    backbone_strong = _coerce_bool(row.get(f"{backbone_context}_strong_transcript_supported", False))
    extension_strong = _coerce_bool(row.get(f"{extension_context}_strong_transcript_supported", False))
    backbone_any = _coerce_bool(row.get(f"{backbone_context}_any_transcript_supported", False))
    extension_any = _coerce_bool(row.get(f"{extension_context}_any_transcript_supported", False))
    backbone_direction = str(row.get(f"{backbone_context}_dominant_direction", "unavailable"))
    extension_direction = str(row.get(f"{extension_context}_dominant_direction", "unavailable"))
    same_direction = (
        backbone_direction in {"positive", "negative"}
        and extension_direction in {"positive", "negative"}
        and backbone_direction == extension_direction
    )

    if backbone_strong and extension_strong and same_direction:
        return "shared_core"
    if backbone_strong and not extension_any:
        return "backbone_specific_periphery"
    if extension_strong and not backbone_any:
        return "extension_inverse_periphery" if extension_direction == "negative" else "extension_specific_periphery"
    if backbone_any and extension_any and same_direction:
        return "shared_periphery"
    if backbone_any and extension_any and not same_direction:
        return "context_unstable"
    if backbone_any and not extension_any:
        return "backbone_supported_only"
    if extension_any and not backbone_any:
        return "extension_supported_only"
    return "unsupported"


def _summarize_module_context(
    module_name: str,
    context_name: str,
    frame: pd.DataFrame,
    delta_support_threshold: float,
) -> dict[str, Any]:
    cell_sig = _bool_series(frame.get("cell_level_padj", pd.Series(dtype=float)).fillna(1.0) < 0.05)
    donor_sig = _bool_series(frame.get("donor_paired_padj", pd.Series(dtype=float)).fillna(1.0) < 0.05)
    mean_cell_delta = _safe_mean(frame.get("cell_level_delta_high_minus_low"))
    mean_donor_delta = _safe_mean(frame.get("donor_paired_mean_delta"))
    direction_score = 2.0 * _sign_or_zero(mean_donor_delta) * int(donor_sig.sum() > 0) + _sign_or_zero(mean_cell_delta) * int(cell_sig.sum() > 0)
    supported = bool(
        donor_sig.sum() > 0
        or cell_sig.sum() > 0
        or abs(mean_cell_delta) >= delta_support_threshold
        or abs(mean_donor_delta) >= delta_support_threshold
    )
    summary = {
        "module_name": module_name,
        "context_name": context_name,
        "n_runs_present": int(len(frame)),
        "cell_level_significant_count": int(cell_sig.sum()),
        "donor_paired_significant_count": int(donor_sig.sum()),
        "mean_cell_level_delta_high_minus_low": mean_cell_delta,
        "mean_donor_paired_delta_high_minus_low": mean_donor_delta,
        "module_support_score": float(2 * int(donor_sig.sum()) + int(cell_sig.sum()) + abs(mean_cell_delta) * 5.0),
        "strong_module_supported": bool(donor_sig.sum() > 0 or cell_sig.sum() >= 1),
        "any_module_supported": supported,
        "direction_score": direction_score,
        "dominant_direction": _direction_label(direction_score),
    }
    return summary


def _merge_context_module_views(
    context_summary: pd.DataFrame,
    backbone_context: str,
    extension_context: str,
) -> pd.DataFrame:
    backbone = _rename_context_columns(
        context_summary[context_summary["context_name"] == backbone_context],
        context_name=backbone_context,
        name_column="module_name",
    )
    extension = _rename_context_columns(
        context_summary[context_summary["context_name"] == extension_context],
        context_name=extension_context,
        name_column="module_name",
    )
    merged = backbone.merge(extension, on="module_name", how="outer")
    merged["phase2_classification"] = merged.apply(
        lambda row: _classify_module_core_vs_context(
            row=row,
            backbone_context=backbone_context,
            extension_context=extension_context,
        ),
        axis=1,
    )
    return merged.sort_values(
        [f"{backbone_context}_module_support_score", f"{extension_context}_module_support_score", "module_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _classify_module_core_vs_context(
    row: pd.Series,
    backbone_context: str,
    extension_context: str,
) -> str:
    backbone_strong = _coerce_bool(row.get(f"{backbone_context}_strong_module_supported", False))
    extension_strong = _coerce_bool(row.get(f"{extension_context}_strong_module_supported", False))
    backbone_any = _coerce_bool(row.get(f"{backbone_context}_any_module_supported", False))
    extension_any = _coerce_bool(row.get(f"{extension_context}_any_module_supported", False))
    backbone_direction = str(row.get(f"{backbone_context}_dominant_direction", "unavailable"))
    extension_direction = str(row.get(f"{extension_context}_dominant_direction", "unavailable"))
    same_direction = (
        backbone_direction in {"positive", "negative"}
        and extension_direction in {"positive", "negative"}
        and backbone_direction == extension_direction
    )
    if backbone_strong and extension_strong and same_direction:
        return "shared_core_module"
    if backbone_strong and not extension_any:
        return "backbone_enriched_module"
    if extension_strong and not backbone_any:
        return "extension_enriched_module"
    if backbone_any and extension_any and same_direction:
        return "shared_peripheral_module"
    if backbone_any and extension_any and not same_direction:
        return "context_unstable_module"
    if backbone_any and not extension_any:
        return "backbone_supported_only_module"
    if extension_any and not backbone_any:
        return "extension_supported_only_module"
    return "unsupported_module"


def _compute_network_context_score(row: pd.Series) -> float:
    reactome = _mean_across_context_prefixed_columns(row, "mean_reactome_pathway_count")
    interactions = _mean_across_context_prefixed_columns(row, "mean_candidate_panel_interaction_count")
    ptm = (
        _mean_across_context_prefixed_columns(row, "mean_annotated_ptm_feature_count")
        + _mean_across_context_prefixed_columns(row, "mean_candidate_panel_ptm_out_count")
        + _mean_across_context_prefixed_columns(row, "mean_candidate_panel_ptm_in_count")
    )
    return 0.4 * math.log1p(max(0.0, reactome)) + 0.6 * math.log1p(max(0.0, interactions)) + 0.6 * math.log1p(max(0.0, ptm))


def _compute_external_context_score(row: pd.Series) -> float:
    total = 0.0
    for column, weight in NETWORK_SCORE_WEIGHTS.items():
        values = [
            row.get(f"{context}_{column}")
            for context in _extract_context_prefixes(row.index)
            if f"{context}_{column}" in row.index
        ]
        numeric = [float(value) for value in values if pd.notna(value)]
        mean_value = sum(numeric) / len(numeric) if numeric else 0.0
        total += weight * mean_value
    return total


def _classify_network_role(
    row: pd.Series,
    backbone_context: str,
    extension_context: str,
    anchor_gene: str,
) -> str:
    gene = str(row.get("gene"))
    if gene == anchor_gene:
        return "core_anchor"

    classification = str(row.get("phase1_classification", "unsupported"))
    backbone_priority = float(row.get(f"{backbone_context}_priority_score", 0.0) or 0.0)
    extension_priority = float(row.get(f"{extension_context}_priority_score", 0.0) or 0.0)
    shared_priority = float(row.get("shared_priority_score", 0.0) or 0.0)
    extension_direction = str(row.get(f"{extension_context}_dominant_direction", "unavailable"))

    if classification == "shared_core" and shared_priority >= 4.0:
        return "shared_core_node"
    if classification == "extension_inverse_periphery" and extension_priority >= 3.0 and extension_direction == "negative":
        return "extension_inverse_marker"
    if backbone_priority >= extension_priority + 2.0 and backbone_priority >= 4.0:
        return "backbone_branch_node"
    if extension_priority >= backbone_priority + 2.0 and extension_priority >= 4.0:
        return "extension_branch_node"
    if classification == "shared_periphery" and shared_priority >= 3.0:
        return "shared_peripheral_node"
    return "lower_priority"


def _combine_network_edges(artifacts: list[RunArtifacts]) -> pd.DataFrame:
    frames = [artifact.network_edges for artifact in artifacts if artifact.network_edges is not None and not artifact.network_edges.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "source_genesymbol",
                "target_genesymbol",
                "is_directed",
                "is_stimulation",
                "is_inhibition",
                "consensus_direction",
                "consensus_stimulation",
                "consensus_inhibition",
                "sources",
                "references",
                "curation_effort",
            ]
        )
    combined = pd.concat(frames, ignore_index=True, sort=False)
    subset = ["source", "target", "source_genesymbol", "target_genesymbol"]
    subset = [column for column in subset if column in combined.columns]
    return combined.drop_duplicates(subset=subset).reset_index(drop=True)


def _filter_prioritized_edges(edges: pd.DataFrame, prioritized_genes: list[str]) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()
    gene_set = {str(gene) for gene in prioritized_genes}
    filtered = edges[
        edges["source_genesymbol"].astype(str).isin(gene_set)
        & edges["target_genesymbol"].astype(str).isin(gene_set)
    ].copy()
    return filtered.sort_values(["source_genesymbol", "target_genesymbol"]).reset_index(drop=True)


def _rename_context_columns(
    frame: pd.DataFrame,
    context_name: str,
    name_column: str = "gene",
) -> pd.DataFrame:
    rename_map = {column: f"{context_name}_{column}" for column in frame.columns if column not in {name_column, "context_name"}}
    return frame.drop(columns=["context_name"], errors="ignore").rename(columns=rename_map)


def _direction_score(
    frame: pd.DataFrame,
    metric_to_support: dict[str, tuple[str, float]],
) -> float:
    total = 0.0
    for metric_column, (support_column, weight) in metric_to_support.items():
        if metric_column not in frame.columns:
            continue
        metric = pd.to_numeric(frame[metric_column], errors="coerce")
        if support_column in frame.columns:
            supported = _bool_series(frame[support_column])
            metric = metric.where(supported)
        valid = metric.dropna()
        if valid.empty:
            continue
        signs = valid.apply(_sign_or_zero)
        total += weight * float(signs.mean())
    return total


def _direction_label(score: float) -> str:
    if score > 0.25:
        return "positive"
    if score < -0.25:
        return "negative"
    if abs(score) > 0:
        return "mixed"
    return "unavailable"


def _build_gene_to_modules(gene_sets: pd.DataFrame) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    if gene_sets.empty:
        return mapping
    for row in gene_sets.itertuples(index=False):
        genes = [gene for gene in str(getattr(row, "genes", "")).split(";") if gene]
        for gene in genes:
            mapping.setdefault(gene, []).append(str(row.module_name))
    return mapping


def _build_module_strength_lookup(
    module_summary: pd.DataFrame,
    backbone_context: str,
    extension_context: str,
) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    if module_summary.empty:
        return lookup
    for row in module_summary.itertuples(index=False):
        lookup[(str(row.module_name), backbone_context)] = float(
            getattr(row, f"{backbone_context}_module_support_score", 0.0) or 0.0
        )
        lookup[(str(row.module_name), extension_context)] = float(
            getattr(row, f"{extension_context}_module_support_score", 0.0) or 0.0
        )
    return lookup


def _subset_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    present = [column for column in columns if column in frame.columns]
    return frame[present].copy()


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| status |\n|---|\n| no rows available |"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
        divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
        body = [
            "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
        return "\n".join([header, divider, *body])


def _write_phase1_summary(
    ranked_candidates: pd.DataFrame,
    output_dir: Path,
    backbone_context: str,
    extension_context: str,
    anchor_gene: str,
) -> Path:
    anchor_row = ranked_candidates[ranked_candidates["gene"] == anchor_gene].head(1)
    shared_rows = ranked_candidates[ranked_candidates["phase1_classification"] == "shared_core"].head(10)
    backbone_rows = ranked_candidates[
        ranked_candidates["phase1_classification"].isin(["backbone_specific_periphery", "backbone_supported_only"])
    ].head(10)
    extension_rows = ranked_candidates[
        ranked_candidates["phase1_classification"].isin(["extension_specific_periphery", "extension_inverse_periphery", "extension_supported_only"])
    ].head(10)
    classification_counts = (
        ranked_candidates["phase1_classification"]
        .fillna("unclassified")
        .value_counts(dropna=False)
        .rename_axis("phase1_classification")
        .reset_index(name="n_genes")
    )

    lines = [
        "# Phase 1 Summary",
        "",
        f"- Backbone context: `{backbone_context}`",
        f"- Extension context: `{extension_context}`",
        f"- Anchor gene: `{anchor_gene}`",
        "",
        "## Shared Core Anchor",
        _markdown_table(_subset_columns(anchor_row, [
            "gene",
            "phase1_classification",
            "shared_core_score",
            f"{backbone_context}_transcript_support_score",
            f"{extension_context}_transcript_support_score",
            f"{backbone_context}_dominant_direction",
            f"{extension_context}_dominant_direction",
        ])),
        "",
        "## Shared Core Candidates",
        _markdown_table(_subset_columns(shared_rows, [
            "gene",
            "phase1_classification",
            "shared_core_score",
            f"{backbone_context}_transcript_support_score",
            f"{extension_context}_transcript_support_score",
            f"{backbone_context}_dominant_direction",
            f"{extension_context}_dominant_direction",
        ])),
        "",
        f"## {backbone_context}-Weighted Candidates",
        _markdown_table(_subset_columns(backbone_rows, [
            "gene",
            "phase1_classification",
            f"{backbone_context}_transcript_support_score",
            f"{extension_context}_transcript_support_score",
            f"{backbone_context}_dominant_direction",
            f"{extension_context}_dominant_direction",
        ])),
        "",
        f"## {extension_context}-Weighted Candidates",
        _markdown_table(_subset_columns(extension_rows, [
            "gene",
            "phase1_classification",
            f"{backbone_context}_transcript_support_score",
            f"{extension_context}_transcript_support_score",
            f"{backbone_context}_dominant_direction",
            f"{extension_context}_dominant_direction",
        ])),
        "",
        "## Classification Counts",
        _markdown_table(classification_counts),
        "",
    ]
    path = output_dir / "phase1_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_phase1_figures(
    ranked_candidates: pd.DataFrame,
    output_dir: Path,
    backbone_context: str,
    extension_context: str,
    anchor_gene: str,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    figure_paths: list[Path] = []

    support_scatter_path = output_dir / "phase1_candidate_support_scatter.png"
    _plot_phase1_candidate_support_scatter(
        ranked_candidates=ranked_candidates,
        backbone_context=backbone_context,
        extension_context=extension_context,
        anchor_gene=anchor_gene,
        output_path=support_scatter_path,
        plt=plt,
    )
    figure_paths.append(support_scatter_path)

    heatmap_path = output_dir / "phase1_top_candidate_heatmap.png"
    _plot_phase1_top_candidate_heatmap(
        ranked_candidates=ranked_candidates,
        backbone_context=backbone_context,
        extension_context=extension_context,
        anchor_gene=anchor_gene,
        output_path=heatmap_path,
        plt=plt,
    )
    figure_paths.append(heatmap_path)
    return figure_paths


def _write_phase2_summary(
    merged: pd.DataFrame,
    run_level: pd.DataFrame,
    classification_counts: pd.DataFrame,
    output_dir: Path,
    backbone_context: str,
    extension_context: str,
) -> Path:
    top_modules = merged.head(10)
    run_level_preview = run_level[[
        column
        for column in [
            "module_name",
            "context_name",
            "run_label",
            "cell_level_delta_high_minus_low",
            "cell_level_padj",
            "donor_paired_mean_delta",
            "donor_paired_padj",
        ]
        if column in run_level.columns
    ]].copy()

    lines = [
        "# Phase 2 Summary",
        "",
        f"- Backbone context: `{backbone_context}`",
        f"- Extension context: `{extension_context}`",
        "",
        "## Context-Level Module Summary",
        _markdown_table(_subset_columns(top_modules, [
            "module_name",
            "phase2_classification",
            f"{backbone_context}_module_support_score",
            f"{extension_context}_module_support_score",
            f"{backbone_context}_dominant_direction",
            f"{extension_context}_dominant_direction",
        ])),
        "",
        "## Classification Counts",
        _markdown_table(classification_counts),
        "",
        "## Run-Level Module Preview",
        _markdown_table(run_level_preview.head(15)),
        "",
    ]
    path = output_dir / "phase2_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_phase2_figures(
    merged: pd.DataFrame,
    run_level: pd.DataFrame,
    output_dir: Path,
    backbone_context: str,
    extension_context: str,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    figure_paths: list[Path] = []

    support_path = output_dir / "phase2_module_support_comparison.png"
    _plot_phase2_module_support_comparison(
        merged=merged,
        backbone_context=backbone_context,
        extension_context=extension_context,
        output_path=support_path,
        plt=plt,
    )
    figure_paths.append(support_path)

    heatmap_path = output_dir / "phase2_module_runlevel_heatmap.png"
    _plot_phase2_module_runlevel_heatmap(
        run_level=run_level,
        output_path=heatmap_path,
        plt=plt,
    )
    figure_paths.append(heatmap_path)
    return figure_paths


def _write_phase3_summary(
    prioritized: pd.DataFrame,
    prioritized_edges: pd.DataFrame,
    role_counts: pd.DataFrame,
    output_dir: Path,
    backbone_context: str,
    extension_context: str,
    anchor_gene: str,
) -> Path:
    top_nodes = prioritized.head(15)
    top_backbone = prioritized[prioritized["network_role"] == "backbone_branch_node"].head(10)
    top_extension = prioritized[
        prioritized["network_role"].isin(["extension_branch_node", "extension_inverse_marker"])
    ].head(10)
    anchor_row = prioritized[prioritized["gene"] == anchor_gene].head(1)

    edge_preview = prioritized_edges[[
        column
        for column in [
            "source_genesymbol",
            "target_genesymbol",
            "consensus_direction",
            "consensus_stimulation",
            "consensus_inhibition",
            "curation_effort",
        ]
        if column in prioritized_edges.columns
    ]].copy()

    lines = [
        "# Phase 3 Summary",
        "",
        f"- Backbone context: `{backbone_context}`",
        f"- Extension context: `{extension_context}`",
        f"- Anchor gene: `{anchor_gene}`",
        "",
        "## Core Anchor",
        _markdown_table(_subset_columns(anchor_row, [
            "gene",
            "network_role",
            "phase1_classification",
            "shared_priority_score",
            f"{backbone_context}_priority_score",
            f"{extension_context}_priority_score",
        ])),
        "",
        "## Top Prioritized Nodes",
        _markdown_table(_subset_columns(top_nodes, [
            "gene",
            "network_role",
            "phase1_classification",
            "shared_priority_score",
            f"{backbone_context}_priority_score",
            f"{extension_context}_priority_score",
        ])),
        "",
        f"## {backbone_context}-Branch Nodes",
        _markdown_table(_subset_columns(top_backbone, [
            "gene",
            "network_role",
            f"{backbone_context}_priority_score",
            f"{extension_context}_priority_score",
        ])),
        "",
        f"## {extension_context}-Branch Nodes",
        _markdown_table(_subset_columns(top_extension, [
            "gene",
            "network_role",
            f"{backbone_context}_priority_score",
            f"{extension_context}_priority_score",
        ])),
        "",
        "## Network Role Counts",
        _markdown_table(role_counts),
        "",
        "## Prioritized Edge Preview",
        _markdown_table(edge_preview.head(15)),
        "",
    ]
    path = output_dir / "phase3_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_phase3_figures(
    prioritized: pd.DataFrame,
    output_dir: Path,
    backbone_context: str,
    extension_context: str,
    anchor_gene: str,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    figure_paths: list[Path] = []

    scatter_path = output_dir / "phase3_priority_score_scatter.png"
    _plot_phase3_priority_score_scatter(
        prioritized=prioritized,
        output_path=scatter_path,
        backbone_context=backbone_context,
        extension_context=extension_context,
        anchor_gene=anchor_gene,
        plt=plt,
    )
    figure_paths.append(scatter_path)

    barplot_path = output_dir / "phase3_top_priority_barplot.png"
    _plot_phase3_top_priority_barplot(
        prioritized=prioritized,
        output_path=barplot_path,
        plt=plt,
    )
    figure_paths.append(barplot_path)
    return figure_paths


def _plot_phase1_candidate_support_scatter(
    ranked_candidates: pd.DataFrame,
    backbone_context: str,
    extension_context: str,
    anchor_gene: str,
    output_path: Path,
    plt,
) -> None:
    frame = ranked_candidates.copy()
    x_col = f"{backbone_context}_transcript_support_score"
    y_col = f"{extension_context}_transcript_support_score"
    frame[x_col] = pd.to_numeric(frame.get(x_col), errors="coerce").fillna(0.0)
    frame[y_col] = pd.to_numeric(frame.get(y_col), errors="coerce").fillna(0.0)
    frame["phase1_classification"] = frame["phase1_classification"].fillna("unclassified")

    color_map = {
        "shared_core_anchor": "#1b9e77",
        "shared_core": "#66a61e",
        "shared_periphery": "#4daf4a",
        "backbone_specific_periphery": "#377eb8",
        "backbone_supported_only": "#80b1d3",
        "extension_specific_periphery": "#e41a1c",
        "extension_inverse_periphery": "#984ea3",
        "extension_supported_only": "#fb8072",
        "context_unstable": "#ff7f00",
        "unsupported": "#b3b3b3",
    }

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for classification, subset in frame.groupby("phase1_classification", observed=True):
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=80 if classification == "shared_core_anchor" else 50,
            alpha=0.9,
            color=color_map.get(classification, "#999999"),
            edgecolor="white",
            linewidth=0.7,
            label=classification.replace("_", " "),
        )

    ax.axhline(0, color="#cccccc", linewidth=1)
    ax.axvline(0, color="#cccccc", linewidth=1)
    ax.set_xlabel(f"{backbone_context} transcript support score")
    ax.set_ylabel(f"{extension_context} transcript support score")
    ax.set_title("Phase 1: Core vs Context-Specific Candidate Support")

    annotations = frame.nlargest(8, [x_col, y_col], keep="all").copy()
    anchor = frame[frame["gene"] == anchor_gene]
    annotations = pd.concat([annotations, anchor], ignore_index=True).drop_duplicates(subset=["gene"])
    for row in annotations.itertuples(index=False):
        x = float(getattr(row, x_col))
        y = float(getattr(row, y_col))
        ax.annotate(
            str(row.gene),
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
        )

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_phase1_top_candidate_heatmap(
    ranked_candidates: pd.DataFrame,
    backbone_context: str,
    extension_context: str,
    anchor_gene: str,
    output_path: Path,
    plt,
) -> None:
    frame = ranked_candidates.copy()
    x_col = f"{backbone_context}_transcript_support_score"
    y_col = f"{extension_context}_transcript_support_score"
    gene_order = []
    gene_order.extend(frame.loc[frame["gene"] == anchor_gene, "gene"].astype(str).tolist())
    gene_order.extend(
        frame.loc[
            frame["phase1_classification"].isin([
                "backbone_specific_periphery",
                "extension_inverse_periphery",
                "shared_periphery",
                "backbone_supported_only",
                "extension_supported_only",
            ]),
            "gene",
        ].astype(str).head(11).tolist()
    )
    gene_order = list(dict.fromkeys(gene_order))
    subset = frame[frame["gene"].isin(gene_order)].copy()
    subset = subset.set_index("gene").reindex(gene_order)

    value_columns = [
        x_col,
        y_col,
        "shared_core_score",
    ]
    plot_matrix = subset[value_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.45 * len(subset) + 1.5)))
    image = ax.imshow(plot_matrix, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(value_columns)))
    ax.set_xticklabels([
        f"{backbone_context}\nsupport",
        f"{extension_context}\nsupport",
        "shared\nscore",
    ])
    ax.set_yticks(range(len(subset.index)))
    ax.set_yticklabels(subset.index.tolist())
    ax.set_title("Phase 1: Top Candidate Support Heatmap")

    for i in range(plot_matrix.shape[0]):
        for j in range(plot_matrix.shape[1]):
            ax.text(j, i, f"{plot_matrix[i, j]:.1f}", ha="center", va="center", fontsize=7, color="#1a1a1a")

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("support score", rotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_phase2_module_support_comparison(
    merged: pd.DataFrame,
    backbone_context: str,
    extension_context: str,
    output_path: Path,
    plt,
) -> None:
    frame = merged.copy()
    x_col = f"{backbone_context}_module_support_score"
    y_col = f"{extension_context}_module_support_score"
    frame[x_col] = pd.to_numeric(frame.get(x_col), errors="coerce").fillna(0.0)
    frame[y_col] = pd.to_numeric(frame.get(y_col), errors="coerce").fillna(0.0)

    color_map = {
        "shared_core_module": "#1b9e77",
        "shared_peripheral_module": "#66a61e",
        "backbone_enriched_module": "#377eb8",
        "backbone_supported_only_module": "#80b1d3",
        "extension_enriched_module": "#e41a1c",
        "extension_supported_only_module": "#fb8072",
        "context_unstable_module": "#ff7f00",
        "unsupported_module": "#b3b3b3",
    }

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    for classification, subset in frame.groupby("phase2_classification", observed=True):
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=80,
            alpha=0.95,
            color=color_map.get(classification, "#999999"),
            edgecolor="white",
            linewidth=0.8,
            label=str(classification).replace("_", " "),
        )

    for row in frame.itertuples(index=False):
        ax.annotate(
            str(row.module_name),
            (float(getattr(row, x_col)), float(getattr(row, y_col))),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
        )

    ax.axhline(0, color="#cccccc", linewidth=1)
    ax.axvline(0, color="#cccccc", linewidth=1)
    ax.set_xlabel(f"{backbone_context} module support score")
    ax.set_ylabel(f"{extension_context} module support score")
    ax.set_title("Phase 2: Module Support by Context")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_phase2_module_runlevel_heatmap(
    run_level: pd.DataFrame,
    output_path: Path,
    plt,
) -> None:
    if run_level.empty:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "No run-level module data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    run_level = run_level.copy()
    run_level["column_label"] = (
        run_level["context_name"].astype(str)
        + "\n"
        + run_level["run_label"].astype(str)
        + "\ncell"
    )
    cell_matrix = run_level.pivot(index="module_name", columns="column_label", values="cell_level_delta_high_minus_low")

    donor_frame = run_level.copy()
    donor_frame["column_label"] = (
        donor_frame["context_name"].astype(str)
        + "\n"
        + donor_frame["run_label"].astype(str)
        + "\ndonor"
    )
    donor_matrix = donor_frame.pivot(index="module_name", columns="column_label", values="donor_paired_mean_delta")

    module_order = sorted(set(cell_matrix.index.tolist()) | set(donor_matrix.index.tolist()))
    combined = pd.concat(
        [
            cell_matrix.reindex(module_order),
            donor_matrix.reindex(module_order),
        ],
        axis=1,
    )
    combined = combined.loc[module_order]
    values = combined.fillna(0.0).to_numpy(dtype=float)
    vmax = max(abs(values.min()), abs(values.max()), 1e-6)

    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * combined.shape[1]), max(4.5, 0.55 * combined.shape[0] + 1.5)))
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(combined.shape[1]))
    ax.set_xticklabels(combined.columns.tolist(), rotation=35, ha="right")
    ax.set_yticks(range(combined.shape[0]))
    ax.set_yticklabels(combined.index.tolist())
    ax.set_title("Phase 2: Run-Level Module Delta Heatmap")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=7, color="#111111")

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("delta high minus low", rotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_phase3_priority_score_scatter(
    prioritized: pd.DataFrame,
    output_path: Path,
    backbone_context: str,
    extension_context: str,
    anchor_gene: str,
    plt,
) -> None:
    frame = prioritized.copy()
    x_col = f"{backbone_context}_priority_score"
    y_col = f"{extension_context}_priority_score"
    frame[x_col] = pd.to_numeric(frame.get(x_col), errors="coerce").fillna(0.0)
    frame[y_col] = pd.to_numeric(frame.get(y_col), errors="coerce").fillna(0.0)

    color_map = {
        "core_anchor": "#1b9e77",
        "shared_core_node": "#66a61e",
        "shared_peripheral_node": "#4daf4a",
        "backbone_branch_node": "#377eb8",
        "extension_branch_node": "#e41a1c",
        "extension_inverse_marker": "#984ea3",
        "lower_priority": "#b3b3b3",
    }

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for role, subset in frame.groupby("network_role", observed=True):
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=90 if role == "core_anchor" else 55,
            alpha=0.92,
            color=color_map.get(role, "#999999"),
            edgecolor="white",
            linewidth=0.8,
            label=str(role).replace("_", " "),
        )

    annotations = pd.concat(
        [
            frame.head(10),
            frame[frame["gene"] == anchor_gene],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["gene"])
    for row in annotations.itertuples(index=False):
        ax.annotate(
            str(row.gene),
            (float(getattr(row, x_col)), float(getattr(row, y_col))),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
        )

    ax.axhline(0, color="#cccccc", linewidth=1)
    ax.axvline(0, color="#cccccc", linewidth=1)
    ax.set_xlabel(f"{backbone_context} priority score")
    ax.set_ylabel(f"{extension_context} priority score")
    ax.set_title("Phase 3: Network-Informed Priority Scores by Context")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_phase3_top_priority_barplot(
    prioritized: pd.DataFrame,
    output_path: Path,
    plt,
) -> None:
    frame = prioritized.head(12).copy()
    color_map = {
        "core_anchor": "#1b9e77",
        "shared_core_node": "#66a61e",
        "shared_peripheral_node": "#4daf4a",
        "backbone_branch_node": "#377eb8",
        "extension_branch_node": "#e41a1c",
        "extension_inverse_marker": "#984ea3",
        "lower_priority": "#b3b3b3",
    }
    frame = frame.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.0, max(4.5, 0.45 * len(frame) + 1.5)))
    colors = [color_map.get(role, "#999999") for role in frame["network_role"].astype(str)]
    ax.barh(frame["gene"], frame["shared_priority_score"], color=colors, edgecolor="white")
    ax.set_xlabel("shared priority score")
    ax.set_title("Phase 3: Top Prioritized Nodes")

    for row in frame.itertuples(index=False):
        ax.text(
            float(row.shared_priority_score) + 0.05,
            row.gene,
            str(row.network_role).replace("_", " "),
            va="center",
            fontsize=8,
            color="#222222",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _safe_mean(series: pd.Series | Any) -> float:
    if series is None:
        return float("nan")
    numeric = pd.to_numeric(pd.Series(series), errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else float("nan")


def _average_available(row: pd.Series, columns: list[str]) -> float:
    values = [float(row[column]) for column in columns if column in row.index and pd.notna(row[column])]
    return sum(values) / len(values) if values else 0.0


def _mean_across_context_prefixed_columns(row: pd.Series, suffix: str) -> float:
    values = [
        float(row[column])
        for column in row.index
        if str(column).endswith(f"_{suffix}") and pd.notna(row[column])
    ]
    return sum(values) / len(values) if values else 0.0


def _extract_context_prefixes(columns: pd.Index) -> list[str]:
    prefixes = set()
    for column in columns:
        column_text = str(column)
        if "_" not in column_text:
            continue
        prefixes.add(column_text.split("_", 1)[0])
    return sorted(prefixes)


def _sign_or_zero(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    numeric = float(value)
    if numeric > 0:
        return 1.0
    if numeric < 0:
        return -1.0
    return 0.0


def _bool_series(series: pd.Series | Any) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    values = pd.Series(series)
    return values.apply(_coerce_bool).astype(bool)


def _coerce_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "", "nan", "none"}:
        return False
    return bool(text)


def _markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_gene_sets(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    data = _load_yaml(path)
    gene_sets: dict[str, list[str]] = {}
    for module_name, payload in data.items():
        if isinstance(payload, dict):
            genes = payload.get("requested_genes") or payload.get("present_genes") or []
        else:
            genes = payload or []
        gene_sets[str(module_name)] = [str(gene) for gene in genes if str(gene).strip()]
    return gene_sets


def _load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, value in base.items():
        merged[key] = _deep_merge(value, {}) if isinstance(value, dict) else value
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_phased_study_config(config: dict[str, Any]) -> None:
    required_fields = ["study_name", "output_dir", "backbone_context", "extension_context", "anchor_gene", "contexts"]
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(f"Missing required phased-study config fields: {', '.join(missing)}")

    for field in ["study_name", "output_dir", "backbone_context", "extension_context", "anchor_gene"]:
        if not str(config.get(field, "")).strip():
            raise ValueError(f"'{field}' must be a non-empty string.")

    if not isinstance(config.get("overwrite_existing_output", True), bool):
        raise ValueError("'overwrite_existing_output' must be a boolean.")

    contexts = config.get("contexts")
    if not isinstance(contexts, dict) or len(contexts) < 2:
        raise ValueError("'contexts' must be a mapping with at least two named contexts.")
    for required_context in [config["backbone_context"], config["extension_context"]]:
        if required_context not in contexts:
            raise ValueError(
                f"Configured context '{required_context}' was not found in 'contexts'."
            )

    for context_name, context in contexts.items():
        if not isinstance(context, dict):
            raise ValueError(f"'contexts.{context_name}' must be a mapping.")
        if "runs" not in context:
            raise ValueError(f"'contexts.{context_name}.runs' is required.")
        runs = context["runs"]
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"'contexts.{context_name}.runs' must be a non-empty list.")
        if int(context.get("min_replicated_continuous_runs", 1)) < 1:
            raise ValueError(f"'contexts.{context_name}.min_replicated_continuous_runs' must be at least 1.")
        for run in runs:
            if isinstance(run, str):
                run = {"path": run}
            if not isinstance(run, dict):
                raise ValueError(f"Each item in 'contexts.{context_name}.runs' must be a mapping or string path.")
            if not str(run.get("path", "")).strip():
                raise ValueError(f"Each item in 'contexts.{context_name}.runs' must define a non-empty 'path'.")
