#!/usr/bin/env python3
"""Run threshold-aware, donor-aware analysis for one independent AD cohort."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from src.ad_cohort_utils import (
    build_canonical_cell_type_eligibility,
    load_cohort_registry,
    prepare_registered_cohort,
)
from src.multicell_ad_utils import (
    build_donor_cross_cell_table,
    load_backed_population,
    run_cell_level_continuous_layer,
    run_continuous_donor_pseudobulk,
    run_program_associations,
    summarize_program_specificity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/ad_external_cohorts.yaml")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-root", default="results/independent_ad_cohorts")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_cohort_registry(args.registry)
    project_root = Path(registry["_project_root"])
    spec = registry["cohorts"][args.cohort]
    input_path = prepare_registered_cohort(registry, args.cohort, overwrite=False)
    output_dir = _resolve(project_root, args.output_root) / args.cohort
    _prepare_output(output_dir, args.overwrite)
    eligibility_dir = output_dir / "eligibility"
    cell_type_dir = output_dir / "cell_types"
    cross_dir = output_dir / "cross_cell_type"
    program_dir = output_dir / "programs"
    for directory in (eligibility_dir, cell_type_dir, cross_dir, program_dir):
        directory.mkdir(parents=True, exist_ok=True)

    thresholds = registry["eligibility"]
    eligibility = build_canonical_cell_type_eligibility(
        input_path,
        eligibility_dir / "cell_type_eligibility.csv",
        minimum_cells=int(thresholds["minimum_cells"]),
        minimum_detected_cells=int(thresholds["minimum_detected_cells"]),
        minimum_donors=int(thresholds["minimum_donors"]),
        minimum_donors_with_detection=int(thresholds["minimum_donors_with_detection"]),
    )

    donor_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    donor_table_metadata: dict[str, dict[str, str]] = {}
    donor_candidate_frames: list[pd.DataFrame] = []
    hurdle_frames: list[pd.DataFrame] = []
    for row in eligibility.itertuples(index=False):
        population_output = cell_type_dir / _safe_name(row.cell_type)
        population_output.mkdir(parents=True, exist_ok=True)
        if not bool(row.cell_level_continuous_eligible):
            pd.DataFrame([{"status": "not_eligible", "reason": row.eligibility_note}]).to_csv(
                population_output / "status.csv", index=False
            )
            continue
        population = load_backed_population(input_path, "cell_type", row.cell_type)
        views = _analysis_views(
            population,
            spec.get("analysis_strata"),
            spec.get("sensitivity_subsets", []),
        )
        for stratum_value, subset_label, analysis_data in views:
            stratum_output = population_output
            if stratum_value is not None:
                stratum_output = stratum_output / f"{_safe_name(str(spec['analysis_strata']))}_{_safe_name(str(stratum_value))}"
            if subset_label is not None:
                stratum_output = stratum_output / f"sensitivity_{_safe_name(subset_label)}"
            continuous = run_cell_level_continuous_layer(
                raw_adata=analysis_data,
                output_dir=stratum_output / "continuous_hurdle",
                candidate_genes=registry["candidate_genes"],
                donor_column="donor_id" if bool(spec.get("donor_inference_eligible", False)) else "__unavailable__",
            )
            hurdle = continuous.candidate_hurdle_summary.copy()
            hurdle["cohort"] = args.cohort
            hurdle["cell_type"] = row.cell_type
            hurdle["analysis_stratum_column"] = spec.get("analysis_strata", "")
            hurdle["analysis_stratum"] = stratum_value if stratum_value is not None else "all"
            hurdle["analysis_subset"] = subset_label if subset_label is not None else "primary"
            hurdle["donor_adjusted"] = bool(spec.get("donor_inference_eligible", False))
            hurdle_frames.append(hurdle)

            if bool(row.donor_continuous_pseudobulk_eligible):
                donor_result = run_continuous_donor_pseudobulk(
                    adata=analysis_data,
                    output_dir=stratum_output / "donor_pseudobulk",
                    candidate_genes=registry["candidate_genes"],
                    donor_column="donor_id",
                    pathology_column="pathology",
                    minimum_donors=int(thresholds["minimum_donors"]),
                    adjust_for_pathology=True,
                )
                if donor_result.status == "completed":
                    table_key = (
                        f"{row.cell_type}||stratum={stratum_value if stratum_value is not None else 'all'}"
                        f"||subset={subset_label if subset_label is not None else 'primary'}"
                    )
                    donor_tables[table_key] = (donor_result.sample_metadata, donor_result.log_cpm)
                    donor_table_metadata[table_key] = {
                        "cell_type": str(row.cell_type),
                        "analysis_stratum": str(stratum_value) if stratum_value is not None else "all",
                        "analysis_subset": subset_label if subset_label is not None else "primary",
                    }
                    candidates = donor_result.candidate_results.copy()
                    candidates["cohort"] = args.cohort
                    candidates["cell_type"] = row.cell_type
                    candidates["analysis_stratum_column"] = spec.get("analysis_strata", "")
                    candidates["analysis_stratum"] = stratum_value if stratum_value is not None else "all"
                    candidates["analysis_subset"] = subset_label if subset_label is not None else "primary"
                    donor_candidate_frames.append(candidates)
            else:
                pd.DataFrame([{"status": "not_eligible", "reason": row.eligibility_note}]).to_csv(
                    stratum_output / "donor_pseudobulk_status.csv", index=False
                )
            if analysis_data is not population:
                del analysis_data
        del population

    hurdle_all = pd.concat(hurdle_frames, ignore_index=True) if hurdle_frames else pd.DataFrame()
    hurdle_all.to_csv(cross_dir / "candidate_hurdle_all_cell_types.csv", index=False)
    donor_candidates = (
        pd.concat(donor_candidate_frames, ignore_index=True) if donor_candidate_frames else pd.DataFrame()
    )
    donor_candidates.to_csv(cross_dir / "donor_candidate_associations_all_cell_types.csv", index=False)

    program_results = pd.DataFrame()
    if donor_tables:
        combined, correlations = build_donor_cross_cell_table(donor_tables, cross_dir)
        program_results = run_program_associations(
            donor_tables=donor_tables,
            program_genes=registry["programs"],
            output_path=program_dir / "donor_program_associations.csv",
            pathology_column="pathology",
        )
        if not program_results.empty:
            program_metadata = pd.DataFrame(
                [donor_table_metadata[str(key)] for key in program_results["cell_type"]],
                index=program_results.index,
            )
            program_results["analysis_stratum_column"] = spec.get("analysis_strata", "")
            program_results["analysis_stratum"] = program_metadata["analysis_stratum"]
            program_results["analysis_subset"] = program_metadata["analysis_subset"]
            program_results["cell_type"] = program_metadata["cell_type"]
            program_results.to_csv(program_dir / "donor_program_associations.csv", index=False)
        summarize_program_specificity(program_results, program_dir / "program_specificity_summary.csv")
        del combined, correlations
    else:
        pd.DataFrame(
            [{"status": "not_run", "reason": spec.get("donor_ineligibility_reason", "No eligible donor-level populations.")}]
        ).to_csv(program_dir / "status.csv", index=False)

    _write_report(output_dir, args.cohort, spec, eligibility, donor_candidates, program_results)
    print(f"Independent cohort analysis completed: {output_dir}")
    return 0


def _write_report(
    output_dir: Path,
    cohort: str,
    spec: dict[str, object],
    eligibility: pd.DataFrame,
    donor_candidates: pd.DataFrame,
    program_results: pd.DataFrame,
) -> None:
    cell_eligible = eligibility.loc[eligibility["cell_level_continuous_eligible"], "cell_type"].tolist()
    donor_eligible = eligibility.loc[eligibility["donor_continuous_pseudobulk_eligible"], "cell_type"].tolist()
    primary_candidates = _primary_rows(donor_candidates)
    sensitivity_candidates = _sensitivity_rows(donor_candidates)
    primary_programs = _primary_rows(program_results)
    sensitivity_programs = _sensitivity_rows(program_results)
    candidate_hits = _candidate_fdr_hits(primary_candidates)
    candidate_sensitivity_hits = _candidate_fdr_hits(sensitivity_candidates)
    program_hits = _program_fdr_hits(primary_programs)
    program_sensitivity_hits = _program_fdr_hits(sensitivity_programs)
    lines = [
        f"# {cohort} independent AD cohort analysis",
        "",
        "## Dataset",
        "",
        f"- Title: {spec['title']}",
        f"- Tissue: {spec.get('tissue', 'not specified')}",
        f"- Role: {spec['role']}",
        f"- Donor-level inference eligible: {bool(spec.get('donor_inference_eligible', False))}",
        f"- Analysis stratum: {spec.get('analysis_strata', 'none')}",
        f"- Matrix scope: {spec.get('matrix_scope', 'full source feature matrix')}",
        "",
        "## Eligibility",
        "",
        f"- Cell-level continuous/hurdle populations: {', '.join(cell_eligible) if cell_eligible else 'none'}",
        f"- Donor-pseudobulk populations: {', '.join(donor_eligible) if donor_eligible else 'none'}",
        "",
        "## High-level output summary",
        "",
        f"- Primary non-self donor-level candidate FDR hits: {int(candidate_hits.shape[0])}",
        f"- Primary donor-level program global-FDR hits: {int(program_hits.shape[0])}",
        f"- Sensitivity-view non-self candidate FDR hits: {int(candidate_sensitivity_hits.shape[0])}",
        f"- Sensitivity-view program global-FDR hits: {int(program_sensitivity_hits.shape[0])}",
        "",
        "## Interpretation boundary",
        "",
        "Cell-level continuous and hurdle results are exploratory. Donor pseudobulk is primary where valid cell-to-donor assignments exist. This cohort is analyzed separately and is never pooled at the nucleus level with another cohort.",
    ]
    if not bool(spec.get("donor_inference_eligible", False)):
        lines.extend(["", f"Donor-level analyses were prohibited: {spec.get('donor_ineligibility_reason', '')}"])
    (output_dir / "independent_cohort_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _primary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "analysis_subset" not in frame.columns:
        return frame
    return frame[frame["analysis_subset"].fillna("primary").eq("primary")]


def _sensitivity_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "analysis_subset" not in frame.columns:
        return frame.iloc[0:0]
    return frame[~frame["analysis_subset"].fillna("primary").eq("primary")]


def _candidate_fdr_hits(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[
        frame["pvals_adj"].fillna(1.0).lt(0.05)
        & ~frame["self_association"].fillna(False).astype(bool)
    ]


def _program_fdr_hits(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[frame["pvals_adj_global"].fillna(1.0).lt(0.05)]


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _safe_name(value: str) -> str:
    return "_".join(str(value).replace("/", " ").split())


def _analysis_strata(adata, column: object) -> list[tuple[str | None, object]]:
    if not column:
        return [(None, adata)]
    column = str(column)
    if column not in adata.obs.columns:
        raise ValueError(f"Configured analysis stratum column '{column}' is missing from adata.obs.")
    strata = []
    for value in sorted(adata.obs[column].dropna().astype(str).unique()):
        subset = adata[adata.obs[column].astype(str).eq(value)].copy()
        if subset.n_obs:
            strata.append((value, subset))
    return strata


def _analysis_views(adata, stratum_column: object, sensitivity_subsets: object):
    """Yield primary and prespecified sensitivity views without duplicating cohorts."""
    for stratum_value, base in _analysis_strata(adata, stratum_column):
        yield stratum_value, None, base
        for subset in sensitivity_subsets or []:
            column = str(subset["column"])
            value = str(subset["value"])
            label = str(subset["label"])
            if column not in base.obs.columns:
                raise ValueError(f"Sensitivity subset column '{column}' is missing from adata.obs.")
            selected = base[base.obs[column].astype(str).eq(value)].copy()
            if selected.n_obs:
                yield stratum_value, label, selected


if __name__ == "__main__":
    raise SystemExit(main())
