#!/usr/bin/env python3
"""Run donor-only sensitivity models within metadata strata for one AD cohort."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from src.ad_cohort_utils import build_canonical_cell_type_eligibility, load_cohort_registry, prepare_registered_cohort
from src.multicell_ad_utils import load_backed_population, run_continuous_donor_pseudobulk, run_program_associations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/ad_external_cohorts.yaml")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--stratum-column", required=True)
    parser.add_argument("--output-root", default="results/independent_ad_cohorts")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_cohort_registry(args.registry)
    root = Path(registry["_project_root"])
    input_path = prepare_registered_cohort(registry, args.cohort, overwrite=False)
    output = _resolve(root, args.output_root) / args.cohort / f"{_safe_name(args.stratum_column)}_stratified"
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    thresholds = registry["eligibility"]
    eligibility = build_canonical_cell_type_eligibility(
        input_path,
        output / "cell_type_eligibility.csv",
        minimum_cells=int(thresholds["minimum_cells"]),
        minimum_detected_cells=int(thresholds["minimum_detected_cells"]),
        minimum_donors=int(thresholds["minimum_donors"]),
        minimum_donors_with_detection=int(thresholds["minimum_donors_with_detection"]),
    )
    candidate_frames = []
    donor_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    metadata_rows = []
    for row in eligibility.itertuples(index=False):
        if not bool(row.donor_continuous_pseudobulk_eligible):
            continue
        population = load_backed_population(input_path, "cell_type", row.cell_type)
        if args.stratum_column not in population.obs:
            raise ValueError(f"Stratum column '{args.stratum_column}' is missing from adata.obs.")
        for stratum in sorted(population.obs[args.stratum_column].dropna().astype(str).unique()):
            selected = population[population.obs[args.stratum_column].astype(str).eq(stratum)].copy()
            donors = int(selected.obs["donor_id"].nunique())
            detected_donors = int(selected.obs.loc[selected[:, "RHBDF2"].X.toarray().reshape(-1) > 0, "donor_id"].nunique())
            if donors < int(thresholds["minimum_donors"]) or detected_donors < int(thresholds["minimum_donors_with_detection"]):
                metadata_rows.append({"cell_type": row.cell_type, "stratum": stratum, "status": "not_eligible", "n_donors": donors, "detected_donors": detected_donors})
                continue
            stratum_output = output / "cell_types" / _safe_name(row.cell_type) / _safe_name(stratum)
            result = run_continuous_donor_pseudobulk(
                adata=selected,
                output_dir=stratum_output,
                candidate_genes=registry["candidate_genes"],
                donor_column="donor_id",
                pathology_column="pathology",
                minimum_donors=int(thresholds["minimum_donors"]),
                adjust_for_pathology=True,
            )
            metadata_rows.append({"cell_type": row.cell_type, "stratum": stratum, "status": result.status, "n_donors": donors, "detected_donors": detected_donors})
            if result.status != "completed":
                continue
            candidates = result.candidate_results.copy()
            candidates["cohort"] = args.cohort
            candidates["cell_type"] = row.cell_type
            candidates["analysis_stratum_column"] = args.stratum_column
            candidates["analysis_stratum"] = stratum
            candidates["analysis_subset"] = f"sensitivity_{args.stratum_column}"
            candidate_frames.append(candidates)
            key = f"{row.cell_type}||{args.stratum_column}={stratum}"
            donor_tables[key] = (result.sample_metadata, result.log_cpm)
        del population
    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    candidates.to_csv(output / "donor_candidate_associations_by_stratum.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(output / "stratum_eligibility_status.csv", index=False)
    programs = run_program_associations(
        donor_tables=donor_tables,
        program_genes=registry["programs"],
        output_path=output / "donor_program_associations_by_stratum.csv",
        pathology_column="pathology",
    ) if donor_tables else pd.DataFrame()
    if not programs.empty:
        split = programs["cell_type"].str.extract(r"^(.*?)\|\|" + args.stratum_column + r"=(.*)$")
        programs["source_table_key"] = programs["cell_type"]
        programs["cell_type"] = split[0]
        programs["analysis_stratum_column"] = args.stratum_column
        programs["analysis_stratum"] = split[1]
        programs["analysis_subset"] = f"sensitivity_{args.stratum_column}"
        programs.to_csv(output / "donor_program_associations_by_stratum.csv", index=False)
    report = [
        f"# {args.cohort} donor-only {args.stratum_column} sensitivity",
        "",
        "The full-donor pathology-adjusted cohort model remains primary. These strata are sensitivity views of the same donors and are never counted as independent cohort replications.",
        "",
        f"- Completed candidate rows: {candidates.shape[0]}",
        f"- Completed program rows: {programs.shape[0]}",
        f"- Eligible cell-type/stratum models: {sum(row['status'] == 'completed' for row in metadata_rows)}",
    ]
    (output / "stratified_donor_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Stratified donor analysis completed: {output}")
    return 0


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _safe_name(value: str) -> str:
    return "_".join(str(value).replace("/", " ").split())


if __name__ == "__main__":
    raise SystemExit(main())
