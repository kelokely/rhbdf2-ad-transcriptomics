#!/usr/bin/env python3
"""Run the GSE243292 astrocyte-first and multi-cell AD extension."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import scanpy as sc
import yaml

from src.multicell_ad_utils import (
    CELL_TYPE_LABELS,
    build_cell_type_eligibility,
    build_donor_cross_cell_table,
    build_subpopulation_eligibility,
    evaluate_ligand_receptor_gate,
    filter_myeloid_like_cells,
    load_backed_population,
    load_backed_subset,
    run_cell_level_continuous_layer,
    run_continuous_donor_pseudobulk,
    run_program_associations,
    save_extension_plots,
    summarize_program_specificity,
    test_donor_rhbdf2_pathology_association,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GSE243292 multi-cell Alzheimer extension.")
    parser.add_argument("--config", required=True, help="Path to the multi-cell YAML config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    full_path = _resolve(project_root, config["full_brain_path"])
    astro_path = _resolve(project_root, config["astrocyte_path"])
    output_dir = _resolve(project_root, config["output_dir"])
    _prepare_output(output_dir, bool(config.get("overwrite", False)))
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    eligibility_dir = output_dir / "eligibility"
    astro_dir = output_dir / "astrocytes"
    cell_type_dir = output_dir / "cell_types"
    cross_dir = output_dir / "cross_cell_type"
    program_dir = output_dir / "programs"
    figure_dir = output_dir / "figures"
    for directory in [eligibility_dir, astro_dir, cell_type_dir, cross_dir, program_dir, figure_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    thresholds = config["eligibility"]
    print("[1/7] Building the full-brain cell-type eligibility table")
    eligibility = build_cell_type_eligibility(
        full_path=full_path,
        output_path=eligibility_dir / "cell_type_eligibility.csv",
        gene="RHBDF2",
        cell_type_column=config["metadata"]["cell_type_column"],
        donor_column=config["metadata"]["donor_column"],
        pathology_column=config["metadata"]["pathology_column"],
        minimum_cells=int(thresholds["minimum_cells"]),
        minimum_detected_cells=int(thresholds["minimum_detected_cells"]),
        minimum_donors_with_detection=int(thresholds["minimum_donors_with_detection"]),
    )

    print("[2/7] Loading clean dedicated astrocytes and recording doublet exclusions")
    astro_backed = sc.read_h5ad(astro_path, backed="r")
    try:
        excluded = set(config["astrocytes"]["excluded_subpopulations"])
        mask = ~astro_backed.obs[config["astrocytes"]["subpopulation_column"]].astype(str).isin(excluded).to_numpy()
    finally:
        astro_backed.file.close()
    all_astro = load_backed_subset(astro_path, np.ones(mask.shape[0], dtype=bool))
    build_subpopulation_eligibility(
        adata=all_astro,
        output_path=eligibility_dir / "astrocyte_subpopulation_eligibility.csv",
        gene="RHBDF2",
        subgroup_column=config["astrocytes"]["subpopulation_column"],
        donor_column=config["metadata"]["donor_column"],
        pathology_column=config["metadata"]["pathology_column"],
        excluded_labels=list(excluded),
    )
    del all_astro
    astro = load_backed_subset(astro_path, mask)
    pd.DataFrame(
        [{"cells_before": int(mask.shape[0]), "cells_after": int(mask.sum()), "excluded_cells": int((~mask).sum()), "excluded_labels": ";".join(sorted(excluded))}]
    ).to_csv(astro_dir / "astrocyte_filter_summary.csv", index=False)
    astro = filter_myeloid_like_cells(
        adata=astro,
        marker_genes=config["purity_qc"]["myeloid_markers"],
        output_path=astro_dir / "myeloid_like_filter_summary.csv",
        minimum_markers=int(config["purity_qc"]["minimum_markers"]),
    )

    print("[3/7] Running astrocyte continuous, hurdle, and donor-continuous pseudobulk analyses")
    astro_continuous = run_cell_level_continuous_layer(
        raw_adata=astro,
        output_dir=astro_dir / "continuous_hurdle",
        candidate_genes=config["candidate_genes"],
        donor_column=config["metadata"]["donor_column"],
    )
    donor_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    donor_candidate_frames: list[pd.DataFrame] = []
    hurdle_frames: list[pd.DataFrame] = []
    astro_hurdle = astro_continuous.candidate_hurdle_summary.copy()
    astro_hurdle["cell_type"] = "Astrocytes"
    hurdle_frames.append(astro_hurdle)
    astro_validation_rows: list[pd.DataFrame] = []
    overall_astro = run_continuous_donor_pseudobulk(
        adata=astro,
        output_dir=astro_dir / "donor_pseudobulk" / "overall_clean_astrocytes",
        candidate_genes=config["candidate_genes"],
        donor_column=config["metadata"]["donor_column"],
        pathology_column=config["metadata"]["pathology_column"],
        minimum_donors=int(config["donor_pseudobulk"]["minimum_donors"]),
        adjust_for_pathology=True,
    )
    if overall_astro.status == "completed":
        donor_tables["Astrocytes"] = (overall_astro.sample_metadata, overall_astro.log_cpm)
        frame = overall_astro.candidate_results.copy()
        frame["cell_type"] = "Astrocytes"
        donor_candidate_frames.append(frame)

    subgroup_column = config["astrocytes"]["subpopulation_column"]
    for subgroup in config["astrocytes"]["primary_subpopulations"]:
        subset = astro[astro.obs[subgroup_column].astype(str) == subgroup].copy()
        result = run_continuous_donor_pseudobulk(
            adata=subset,
            output_dir=astro_dir / "donor_pseudobulk" / "subpopulations" / _safe_name(subgroup),
            candidate_genes=config["candidate_genes"],
            donor_column=config["metadata"]["donor_column"],
            pathology_column=config["metadata"]["pathology_column"],
            minimum_donors=int(config["donor_pseudobulk"]["minimum_donors"]),
            adjust_for_pathology=True,
        )
        if result.status == "completed":
            frame = result.candidate_results.copy()
            frame["validation_stratum"] = f"subpopulation:{subgroup}"
            astro_validation_rows.append(frame)
        del subset

    pathology_column = config["metadata"]["pathology_column"]
    for pathology, donor_count in astro.obs[[config["metadata"]["donor_column"], pathology_column]].drop_duplicates().groupby(pathology_column, observed=True)[config["metadata"]["donor_column"]].nunique().items():
        subset = astro[astro.obs[pathology_column].astype(str) == str(pathology)].copy()
        result = run_continuous_donor_pseudobulk(
            adata=subset,
            output_dir=astro_dir / "donor_pseudobulk" / "pathology" / _safe_name(str(pathology)),
            candidate_genes=config["candidate_genes"],
            donor_column=config["metadata"]["donor_column"],
            pathology_column=pathology_column,
            minimum_donors=int(config["donor_pseudobulk"]["minimum_donors"]),
            adjust_for_pathology=False,
        )
        if result.status == "completed":
            frame = result.candidate_results.copy()
            frame["validation_stratum"] = f"pathology:{pathology}"
            astro_validation_rows.append(frame)
        del subset
    astro_validation = pd.concat(astro_validation_rows, ignore_index=True) if astro_validation_rows else pd.DataFrame()
    astro_validation.to_csv(astro_dir / "astrocyte_stratified_candidate_validation.csv", index=False)
    del astro

    print("[4/7] Running eligible full-brain cell types separately")
    for row in eligibility.itertuples(index=False):
        code = str(row.cell_type_code)
        if code not in CELL_TYPE_LABELS or code == "Ast":
            continue
        if not bool(row.donor_continuous_pseudobulk_eligible):
            population_output = cell_type_dir / _safe_name(str(row.cell_type))
            population_output.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"status": "not_eligible", "reason": row.eligibility_note}]).to_csv(
                population_output / "status.csv", index=False
            )
            continue
        population = load_backed_population(full_path, config["metadata"]["cell_type_column"], code)
        population_output = cell_type_dir / _safe_name(str(row.cell_type))
        population_output.mkdir(parents=True, exist_ok=True)
        if code != "Mic":
            population = filter_myeloid_like_cells(
                adata=population,
                marker_genes=config["purity_qc"]["myeloid_markers"],
                output_path=population_output / "myeloid_like_filter_summary.csv",
                minimum_markers=int(config["purity_qc"]["minimum_markers"]),
            )
        if bool(row.cell_level_continuous_eligible):
            continuous_result = run_cell_level_continuous_layer(
                raw_adata=population,
                output_dir=population_output / "continuous_hurdle",
                candidate_genes=config["candidate_genes"],
                donor_column=config["metadata"]["donor_column"],
            )
            frame = continuous_result.candidate_hurdle_summary.copy()
            frame["cell_type"] = str(row.cell_type)
            hurdle_frames.append(frame)
        donor_result = run_continuous_donor_pseudobulk(
            adata=population,
            output_dir=population_output / "donor_pseudobulk",
            candidate_genes=config["candidate_genes"],
            donor_column=config["metadata"]["donor_column"],
            pathology_column=pathology_column,
            minimum_donors=int(config["donor_pseudobulk"]["minimum_donors"]),
            adjust_for_pathology=True,
        )
        if donor_result.status == "completed":
            donor_tables[str(row.cell_type)] = (donor_result.sample_metadata, donor_result.log_cpm)
            frame = donor_result.candidate_results.copy()
            frame["cell_type"] = str(row.cell_type)
            donor_candidate_frames.append(frame)
        del population

    print("[5/7] Comparing donor-level RHBDF2 across cell types")
    donor_combined, donor_correlation = build_donor_cross_cell_table(donor_tables, cross_dir)
    donor_candidates = pd.concat(donor_candidate_frames, ignore_index=True)
    donor_candidates.to_csv(cross_dir / "donor_candidate_associations_all_cell_types.csv", index=False)
    pd.concat(hurdle_frames, ignore_index=True).to_csv(
        cross_dir / "cell_level_candidate_hurdle_all_cell_types.csv", index=False
    )
    pathology_summary = (
        donor_combined.groupby(["cell_type", pathology_column], observed=True)
        .agg(
            n_donors=("donor_id", "nunique"),
            mean_detection_fraction=("rhbdf2_detection_fraction", "mean"),
            median_detection_fraction=("rhbdf2_detection_fraction", "median"),
        )
        .reset_index()
    )
    pathology_summary.to_csv(cross_dir / "rhbdf2_detection_by_cell_type_and_pathology.csv", index=False)
    pathology_tests = test_donor_rhbdf2_pathology_association(
        donor_table=donor_combined,
        pathology_column=pathology_column,
        output_path=cross_dir / "donor_rhbdf2_pathology_association.csv",
    )

    print("[6/7] Testing shared versus cell-type-specific programs")
    program_results = run_program_associations(
        donor_tables=donor_tables,
        program_genes=config["programs"],
        output_path=program_dir / "donor_program_associations.csv",
        pathology_column=pathology_column,
    )
    summarize_program_specificity(program_results, program_dir / "program_specificity_summary.csv")
    gate = evaluate_ligand_receptor_gate(
        program_results=program_results,
        astro_validation=astro_validation,
        output_path=output_dir / "ligand_receptor" / "gate_status.csv",
    )

    print("[7/7] Saving figures and data-driven report")
    save_extension_plots(eligibility, donor_correlation, program_results, figure_dir, dpi=int(config.get("figure_dpi", 300)))
    _write_report(
        output_dir,
        eligibility,
        overall_astro,
        astro_validation,
        donor_candidates,
        program_results,
        pathology_tests,
        gate,
    )
    print(f"Multi-cell AD extension completed: {output_dir}")
    return 0


def _write_report(
    output_dir,
    eligibility,
    overall_astro,
    astro_validation,
    donor_candidates,
    program_results,
    pathology_tests,
    gate,
) -> None:
    eligible = eligibility.loc[eligibility["analysis_status"] == "eligible", "cell_type"].astype(str).tolist()
    astro_hits = overall_astro.candidate_results[
        (~overall_astro.candidate_results.get("self_association", False).astype(bool))
        & (overall_astro.candidate_results["pvals_adj"] < 0.05)
    ] if not overall_astro.candidate_results.empty else pd.DataFrame()
    program_hits = program_results[program_results["pvals_adj_global"].fillna(1.0) < 0.05]
    stratified_nonself = astro_validation[
        (astro_validation.get("pvals_adj", pd.Series(dtype=float)).fillna(1.0) < 0.05)
        & (~astro_validation.get("self_association", pd.Series(False, index=astro_validation.index)).fillna(False).astype(bool))
        & (astro_validation.get("names", pd.Series("", index=astro_validation.index)).astype(str) != "RHBDF2")
    ] if not astro_validation.empty else astro_validation
    donor_candidate_hits = donor_candidates[
        (donor_candidates["pvals_adj"].fillna(1.0) < 0.05)
        & (~donor_candidates["self_association"].fillna(False).astype(bool))
        & (donor_candidates["names"] != "RHBDF2")
    ]
    pathology_hits = pathology_tests[pathology_tests["pvals_adj"].fillna(1.0) < 0.05]
    lines = [
        "# GSE243292 Multi-Cell AD Extension",
        "",
        "## Dataset and design",
        "",
        "The analysis used the official full-brain and dedicated astrocyte GSE243292 processed H5AD objects. Cell types were analyzed separately; no pooled pan-cell high/low contrast was performed.",
        "",
        "## Eligibility results",
        "",
        f"Eligible cell types: {', '.join(eligible) if eligible else 'none'}.",
        "Pericytes were not separately annotated. Endothelial cells were retained for descriptive reporting when they did not meet inferential thresholds.",
        "",
        "## Astrocyte-first results",
        "",
        f"Clean astrocyte donor pseudobulk status: {overall_astro.status}. Non-self candidate FDR hits: {', '.join(astro_hits['names'].astype(str)) if not astro_hits.empty else 'none'}.",
        f"Stratified astrocyte non-self candidate FDR hits: {int(stratified_nonself.shape[0])}.",
        "",
        "## Cross-cell donor results",
        "",
        f"Non-self donor-level candidate FDR hits across all cell types: {int(donor_candidate_hits.shape[0])}.",
        f"Cell types with donor-level RHBDF2 pathology association after FDR correction: {', '.join(pathology_hits['cell_type'].astype(str)) if not pathology_hits.empty else 'none'}.",
        "",
        "## Cross-cell program results",
        "",
        f"Donor-level within-cell-type program FDR hits: {int(program_hits.shape[0])}.",
        "",
        "## Ligand-receptor gate",
        "",
        f"Status: {gate.iloc[0]['status']}. {gate.iloc[0]['reason']}",
        "",
        "## Interpretation guardrails",
        "",
        "RHBDF2 is sparse outside microglia, so non-microglial analyses use continuous detection-rate models rather than a forced binary state. Cell-level continuous and hurdle results are exploratory. Donor-level pseudobulk is the primary inferential layer. The dedicated astrocyte and full-brain objects come from the same cohort and therefore provide cross-cell-type comparison, not independent cohort replication.",
    ]
    (output_dir / "multicell_extension_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _safe_name(value: str) -> str:
    return "_".join(value.replace("/", " ").split())


if __name__ == "__main__":
    raise SystemExit(main())
