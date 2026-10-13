#!/usr/bin/env python3
"""Synthesize donor-aware RHBDF2 associations across independent AD cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from src.ad_cohort_utils import build_registry_status, load_cohort_registry
from src.independent_ad_synthesis_utils import (
    build_ligand_receptor_gate,
    load_donor_candidate_results,
    load_donor_program_results,
    plot_replication_heatmap,
    summarize_replication,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/ad_external_cohorts.yaml")
    parser.add_argument("--output", default="results/independent_ad_cohorts/synthesis")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_cohort_registry(args.registry)
    root = Path(registry["_project_root"])
    output = _resolve(root, args.output)
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}")
        shutil.rmtree(output)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    cohort_dirs = {
        "GSE243292": root / "results/GSE243292_multicell_AD_extension",
        **{
            cohort: root / "results/independent_ad_cohorts" / cohort
            for cohort, spec in registry["cohorts"].items()
            if cohort != "ROSMAP_MULTI_REGION"
            and spec.get("analysis_mode") != "spatial_support_only"
        },
    }
    candidates, candidate_status = load_donor_candidate_results(cohort_dirs)
    programs, program_status = load_donor_program_results(cohort_dirs)
    statuses = pd.concat([candidate_status, program_status], ignore_index=True, sort=False)
    statuses.to_csv(tables / "result_input_status.csv", index=False)
    candidates.to_csv(tables / "donor_candidate_results_all_cohorts.csv", index=False)
    programs.to_csv(tables / "donor_program_results_all_cohorts.csv", index=False)

    primary_candidates = candidates[
        candidates["analysis_subset"].astype(str).eq("primary")
    ] if not candidates.empty else candidates
    primary_programs = programs[
        programs["analysis_subset"].astype(str).eq("primary")
    ] if not programs.empty else programs
    candidate_summary = summarize_replication(
        primary_candidates,
        "names",
        "effect_per_log1p_rhbdf2_detection_per_1000_cells",
        "pvals",
        "pvals_adj",
    )
    program_summary = summarize_replication(
        primary_programs[primary_programs["status"].eq("completed")] if not primary_programs.empty else primary_programs,
        "program",
        "effect",
        "pval",
        "pvals_adj_global",
    )
    candidate_summary.to_csv(tables / "candidate_cross_cohort_replication.csv", index=False)
    program_summary.to_csv(tables / "program_cross_cohort_replication.csv", index=False)
    if not candidates.empty and candidates["analysis_subset"].astype(str).ne("primary").any():
        candidates[candidates["analysis_subset"].astype(str).ne("primary")].to_csv(
            tables / "candidate_sensitivity_results.csv", index=False
        )
    if not programs.empty and programs["analysis_subset"].astype(str).ne("primary").any():
        programs[programs["analysis_subset"].astype(str).ne("primary")].to_csv(
            tables / "program_sensitivity_results.csv", index=False
        )

    registry_status = build_registry_status(registry)
    registry_status.to_csv(tables / "cohort_registry_status.csv", index=False)
    included = set(candidate_status.loc[candidate_status["status"].eq("included"), "cohort"])
    donor_valid = registry_status[
        registry_status["cohort"].isin(included)
        & registry_status["donor_inference_eligible"].fillna(False)
    ]["cohort"].nunique()
    # The discovery cohort is donor-valid but is not part of the external registry.
    if "GSE243292" in included:
        donor_valid += 1
    gate_candidates = candidate_summary[
        ~candidate_summary["names"].astype(str).str.upper().eq("RHBDF2")
    ] if not candidate_summary.empty else candidate_summary
    gate = build_ligand_receptor_gate(gate_candidates, program_summary, int(donor_valid))
    ligand_receptor_replication = (
        root
        / "results/independent_ad_cohorts/ligand_receptor/tables/donor_fdr_supported_lr_replication.csv"
    )
    if gate.iloc[0]["status"] == "eligible_not_run" and ligand_receptor_replication.exists():
        lr_results = pd.read_csv(ligand_receptor_replication)
        replicated_edges = int(
            lr_results["replication_status"].eq("replicated_two_or_more_cohorts").sum()
        ) if "replication_status" in lr_results else 0
        gate.loc[0, "status"] = "completed"
        gate.loc[0, "replicated_ligand_receptor_edges"] = replicated_edges
        gate.loc[0, "reason"] = (
            "The replication gate passed and the donor-FDR OmniPath node overlay was completed; "
            f"{replicated_edges} directed cell-pair edge(s) were supported in at least two cohorts."
        )
    gate.to_csv(tables / "ligand_receptor_gate.csv", index=False)

    external_path = root / "results/independent_ad_cohorts/external_evidence/expanded_candidate_evidence_summary.csv"
    if external_path.exists():
        external = pd.read_csv(external_path)
        external.to_csv(tables / "external_database_coverage_snapshot.csv", index=False)
    orthogonal_path = root / "results/independent_ad_cohorts/orthogonal_evidence/orthogonal_resource_coverage_by_gene.csv"
    if orthogonal_path.exists():
        orthogonal = pd.read_csv(orthogonal_path)
        orthogonal.to_csv(tables / "orthogonal_resource_coverage_snapshot.csv", index=False)

    plot_replication_heatmap(
        primary_candidates[~primary_candidates["self_association"].fillna(False)] if not primary_candidates.empty else primary_candidates,
        "names",
        "effect_per_log1p_rhbdf2_detection_per_1000_cells",
        "pvals",
        figures / "candidate_donor_association_heatmap.png",
    )
    plot_replication_heatmap(
        primary_programs[primary_programs["status"].eq("completed")] if not primary_programs.empty else primary_programs,
        "program",
        "effect",
        "pval",
        figures / "program_donor_association_heatmap.png",
    )
    _write_report(output, registry_status, statuses, candidate_summary, program_summary, gate)
    print(f"Independent AD synthesis completed: {output}")
    return 0


def _write_report(
    output: Path,
    registry_status: pd.DataFrame,
    input_status: pd.DataFrame,
    candidates: pd.DataFrame,
    programs: pd.DataFrame,
    gate: pd.DataFrame,
) -> None:
    included = sorted(input_status.loc[input_status["status"].eq("included"), "cohort"].unique())
    unavailable = sorted(input_status.loc[input_status["status"].eq("unavailable"), "cohort"].unique())
    spatial_support = sorted(
        registry_status.loc[
            registry_status["role"].astype(str).eq("spatial_localization_support"), "cohort"
        ].astype(str).tolist()
    )
    candidate_counts = candidates["replication_tier"].value_counts().to_dict() if not candidates.empty else {}
    self_anchor_fdr = int(
        candidates[
            candidates["names"].astype(str).str.upper().eq("RHBDF2")
            & candidates["replication_tier"].eq("replicated_fdr")
        ].shape[0]
    ) if not candidates.empty else 0
    nonself_candidate_fdr = int(
        candidates[
            ~candidates["names"].astype(str).str.upper().eq("RHBDF2")
            & candidates["replication_tier"].eq("replicated_fdr")
        ].shape[0]
    ) if not candidates.empty else 0
    program_counts = programs["replication_tier"].value_counts().to_dict() if not programs.empty else {}
    lines = [
        "# Independent Alzheimer cohort synthesis",
        "",
        "## Scope",
        "",
        "This synthesis compares donor-level associations produced independently within each cohort. Cells and effect estimates are not pooled across cohorts, and external databases are not counted as replications.",
        "",
        "## Data availability",
        "",
        f"- Cohorts with donor-result tables: {', '.join(included) if included else 'none'}",
        f"- Registered cohorts not yet available to this synthesis: {', '.join(unavailable) if unavailable else 'none'}",
        f"- Spatial-support cohorts excluded from snRNA-seq replication accounting: {', '.join(spatial_support) if spatial_support else 'none'}",
        "- Controlled ROSMAP data remain access-dependent and are not represented as analyzed data.",
        "",
        "## Replication accounting",
        "",
        f"- Candidate tiers: {candidate_counts}",
        f"- FDR-replicated RHBDF2 self-association anchors: {self_anchor_fdr}",
        f"- FDR-replicated non-self candidate associations: {nonself_candidate_fdr}",
        f"- Program tiers: {program_counts}",
        "- `replicated_fdr` requires same-cell-type, same-direction FDR support in at least two independent cohorts.",
        "- `replicated_nominal` requires analogous nominal-p support and is weaker evidence.",
        "- RHBDF2 self-association is an internal anchor/QC relationship and is excluded from downstream replication and communication gates.",
        "",
        "## Communication-analysis gate",
        "",
        f"- Status: {gate.iloc[0]['status']}",
        f"- Reason: {gate.iloc[0]['reason']}",
        "",
        "## Interpretation boundary",
        "",
        "These tables quantify reproducibility of transcript-level donor associations. They do not establish RHBDF2 causality, iRhom2 protein activity, ADAM17 substrate shedding, or therapeutic efficacy. Database records provide dated mechanistic or translational context only.",
    ]
    (output / "independent_ad_synthesis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
