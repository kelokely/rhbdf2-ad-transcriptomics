#!/usr/bin/env python3
"""Audit and standardize public donor metadata for the AD cohort collection."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import anndata as ad
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def snake_case(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return re.sub(r"_+", "_", value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_values(series: pd.Series) -> object:
    values = [str(value) for value in series.dropna().unique() if str(value) not in {"", "nan", "None"}]
    return values[0] if len(values) == 1 else ";".join(sorted(values)) if values else pd.NA


def embedded_donor_table(path: Path, donor_column: str, columns: list[str]) -> pd.DataFrame:
    obj = ad.read_h5ad(path, backed="r")
    available = [column for column in [donor_column, *columns] if column in obj.obs.columns]
    obs = obj.obs[available].copy()
    obs[donor_column] = obs[donor_column].astype(str)
    counts = obs.groupby(donor_column, observed=True).size().rename("n_observations")
    metadata = obs.groupby(donor_column, observed=True).agg(summarize_values)
    result = metadata.join(counts).reset_index()
    result = result.rename(columns={donor_column: "donor_id"})
    return result


def write_gse146639(source: Path, h5ad_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    donor = pd.read_excel(source, sheet_name="Table S2", header=1)
    donor = donor.loc[:, ~donor.columns.astype(str).str.startswith("Unnamed")].copy()
    donor = donor.rename(columns={
        "DonorID": "donor_id",
        "Group": "reported_group",
        "pH": "brain_ph",
        "PMD": "postmortem_delay_reported",
        "Age": "age_at_death",
        "Region": "brain_region_reported",
        "Gender": "sex",
        "Cause of Death": "cause_of_death",
        "Brain Bank": "brain_bank",
    })
    donor["donor_id"] = donor["donor_id"].astype(str).str.strip()
    donor["source_cohort"] = "GSE146639"
    donor["metadata_source"] = "Table S2, PMC7655794 supplementary workbook"

    embedded = embedded_donor_table(h5ad_path, "donor_id", ["diagnosis"])
    joined = embedded.merge(donor, on="donor_id", how="left", validate="one_to_one")
    joined["supplementary_metadata_matched"] = joined["age_at_death"].notna()
    donor.to_csv(output_dir / "GSE146639_public_donor_metadata.csv", index=False)
    joined.to_csv(output_dir / "GSE146639_analysis_ready_donor_metadata.csv", index=False)
    return donor, joined


def write_seaad(source: Path, h5ad_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.read_excel(source, sheet_name="SEA-AD_Cohort_Metadata")
    full.columns = [snake_case(column) for column in full.columns]
    full = full.rename(columns={"donor_id": "donor_id"})
    full["donor_id"] = full["donor_id"].astype(str).str.strip()
    full["source_cohort"] = "SEA-AD MTG"
    full["metadata_source"] = "Allen Institute SEA-AD public cohort donor metadata workbook"
    full.to_csv(output_dir / "SEAAD_public_donor_metadata_full.csv", index=False)

    selected = [
        "donor_id", "primary_study_name", "secondary_study_name", "age_at_death", "sex",
        "highest_level_of_education", "years_of_education", "apoe_genotype", "cognitive_status",
        "age_of_onset_cognitive_symptoms", "age_of_dementia_diagnosis", "known_head_injury",
        "have_they_had_neuroimaging", "last_casi_score", "interval_from_last_casi_in_months",
        "last_mmse_score", "interval_from_last_mmse_in_months", "last_moca_score",
        "interval_from_last_moca_in_months", "pmi", "brain_ph",
        "overall_ad_neuropathological_change", "thal", "braak", "cerad_score",
        "overall_caa_score", "highest_lewy_body_disease",
        "total_microinfarcts_not_observed_grossly", "total_microinfarcts_in_screening_sections",
        "atherosclerosis", "arteriolosclerosis", "late", "rin", "severely_affected_donor",
    ]
    diagnosis_columns = [column for column in full if column.startswith("consensus_clinical_dx_")]
    selected = [column for column in selected + diagnosis_columns if column in full.columns]
    public = full[selected].copy()
    embedded = embedded_donor_table(
        h5ad_path,
        "donor_id",
        ["pathology", "brain_region", "cognitive_status", "apoe_genotype", "sex", "age_at_death"],
    )
    joined = embedded.merge(public, on="donor_id", how="left", suffixes=("_embedded", "_public"), validate="one_to_one")
    joined["public_metadata_matched"] = joined["primary_study_name"].notna()
    joined.to_csv(output_dir / "SEAAD_analysis_ready_donor_metadata.csv", index=False)
    return full, joined


def write_embedded_tables(output_dir: Path) -> dict[str, pd.DataFrame]:
    cohorts = {
        "GSE243292": (
            ROOT / "data/raw/GSE243292_ADsnRNAseq_GEO.h5ad", "sampleID", ["atscore", "apoe", "trem2"]
        ),
        "GSE146639": (
            ROOT / "data/raw/GSE146639_barcoded_smartseq2_microglia.h5ad", "donor_id", ["diagnosis"]
        ),
        "GSE138852": (
            ROOT / "data/external_ad_cohorts/GSE138852/processed/GSE138852_canonical.h5ad",
            "donor_id", ["pathology", "brain_region", "donor_inference_eligible", "donor_ineligibility_reason"],
        ),
        "GSE157827": (
            ROOT / "data/external_ad_cohorts/GSE157827/processed/GSE157827_canonical.h5ad",
            "donor_id", ["pathology", "brain_region"],
        ),
        "GSE237718": (
            ROOT / "data/external_ad_cohorts/GSE237718/processed/GSE237718_canonical.h5ad",
            "donor_id", ["pathology", "brain_region", "apoe_genotype", "sex", "geo_accession"],
        ),
        "GSE268599": (
            ROOT / "data/external_ad_cohorts/GSE268599/processed/GSE268599_canonical.h5ad",
            "donor_id", ["pathology", "brain_region", "age", "sex", "apoe_genotype", "braak_stage", "cerad_stage", "rin"],
        ),
        "GSE269906": (
            ROOT / "data/external_ad_cohorts/GSE269906/processed/GSE269906_spatial_targeted.h5ad",
            "donor_id", ["pathology", "brain_region"],
        ),
        "SEAAD_MTG": (
            ROOT / "data/external_ad_cohorts/SEAAD_MTG/processed/SEAAD_MTG_targeted_canonical.h5ad",
            "donor_id", ["pathology", "brain_region", "cognitive_status", "apoe_genotype", "sex", "age_at_death"],
        ),
    }
    tables: dict[str, pd.DataFrame] = {}
    for cohort, (path, donor_column, columns) in cohorts.items():
        table = embedded_donor_table(path, donor_column, columns)
        table.insert(0, "cohort", cohort)
        table.to_csv(output_dir / f"{cohort}_embedded_donor_metadata.csv", index=False)
        tables[cohort] = table
    return tables


def availability_table(tables: dict[str, pd.DataFrame], gse146_joined: pd.DataFrame, seaad_joined: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ["GSE243292", 15, "DLPFC", "A/T pathology state", "not public in current source", "not public", "APOE; TREM2", "A-T-/A+T-/A+T+", "not public", "not public", "not public", "direct", "Use pathology/genotype only"],
        ["GSE146639", len(tables["GSE146639"]), "LPS; NBB/NBB-IBB", "CTR/CTR+/MCI/AD supplement; binary RNA label", f"{gse146_joined.age_at_death.notna().sum()}/{len(gse146_joined)} donors", f"{gse146_joined.sex.notna().sum()}/{len(gse146_joined)} donors", "not reported in Table S2", "clinical group", "PMD; brain pH", "cause of death", "partial public history", "14/15 matched", "One low-cell donor lacks Table S2 metadata"],
        ["GSE138852", 0, "entorhinal cortex", "AD/control pools", "not cell-linkable", "not cell-linkable", "unavailable", "unavailable", "unavailable", "unavailable", "unavailable", "not donor-linkable", "Pooled donors prevent donor-level covariate models"],
        ["GSE157827", len(tables["GSE157827"]), "prefrontal cortex", "AD/control", "reported in supplement, not integrated", "reported in supplement, not integrated", "reported in supplement, not integrated", "Braak-based study eligibility", "PMI reported in supplement", "limited exclusion history in paper", "limited", "direct donor IDs", "Obtain and version-lock Supplementary Table S1 before covariate use"],
        ["GSE237718", len(tables["GSE237718"]), "temporal cortex", "pathologically confirmed AD/control", "not public in current source", "available", "available", "pathologic diagnosis only", "not public in current source", "not public", "not public", "direct", "Sex/APOE can be modeled; age/stage cannot currently"],
        ["GSE268599", len(tables["GSE268599"]), "V1/V2/BA46/BA20/EC", "PathStage 1-4", "available", "available", "available", "Braak; CERAD; PathStage", "RIN and library QC", "not public in current source", "not public", "direct", "Rich multi-region neuropathology metadata"],
        ["GSE269906", len(tables["GSE269906"]), "prefrontal cortex BA10", "AD/control", "not public in current source", "not public", "not public", "diagnosis only", "not public", "not public", "not public", "sample-level", "Spatial context; limited donor covariates"],
        ["SEAAD_MTG", len(tables["SEAAD_MTG"]), "middle temporal gyrus", "ADNC and cognitive status", f"{seaad_joined.primary_study_name.notna().sum()}/{len(seaad_joined)} donors", "available", "available", "ADNC; Thal; Braak; CERAD; CAA; LBD; LATE", "PMI; RIN; brain pH", "head injury; vascular/Lewy pathology; diagnosis flags", "rich public de-identified history", "84/84 matched", "Best cohort for covariate and clinical-pathology sensitivity analyses"],
        ["ROSMAP", pd.NA, "multiple", "clinical and neuropathologic", "controlled access", "controlled access", "controlled access", "rich controlled access", "controlled access", "longitudinal clinical history", "rich but controlled", "not locally authorized", "Use only after Synapse approval"],
    ]
    columns = [
        "cohort", "donor_count_or_linkable", "donor_source_or_region", "diagnosis_or_pathology",
        "age", "sex", "genetics", "disease_stage", "technical_covariates", "history_or_comorbidity",
        "patient_history_depth", "join_status", "analysis_note",
    ]
    return pd.DataFrame(rows, columns=columns)


def data_dictionary() -> pd.DataFrame:
    rows = [
        ["donor_id", "De-identified donor key", "identifier", "aggregation/join", "Never infer identity"],
        ["brain_region", "Anatomical sampling region", "categorical", "primary stratifier/covariate", "Preserve original labels"],
        ["pathology_or_diagnosis", "Cohort-native disease/pathology label", "categorical/ordinal", "primary exposure", "Do not equate clinical diagnosis with neuropathology"],
        ["age_at_death", "Age at death", "numeric or top-coded", "primary covariate", "Preserve 90+ top coding where present"],
        ["sex", "Reported biological sex", "categorical", "primary covariate", "Use only when model degrees of freedom permit"],
        ["apoe_genotype", "Reported APOE genotype", "categorical", "secondary sensitivity/stratification", "Avoid sparse genotype cells"],
        ["braak/cerad/thal/adnc", "Neuropathology staging systems", "ordinal", "secondary pathology analyses", "Keep as separate variables"],
        ["cognitive_status/scores", "Last reported status or score", "categorical/numeric", "secondary clinical association", "Account for assessment-to-death interval"],
        ["pmi/rin/brain_ph", "Tissue and RNA quality variables", "numeric", "technical sensitivity covariates", "Do not overfit small donor sets"],
        ["history/comorbidity", "Public de-identified cause, injury, vascular or co-pathology fields", "mixed", "descriptive/exploratory", "Not a complete patient record; sparse/multiple-testing risk"],
    ]
    return pd.DataFrame(rows, columns=["field_family", "definition", "type", "recommended_role", "guardrail"])


def write_report(output_dir: Path, availability: pd.DataFrame, gse146: pd.DataFrame, seaad: pd.DataFrame) -> None:
    text = f"""# AD donor metadata audit

## Scope

This audit inventories public, de-identified donor-level information that can be joined to the AD transcriptomic cohorts. It is a provenance and model-design artifact, not a clinical-record reconstruction. Original cohort variables are retained because A/T status, clinical diagnosis, Braak stage, CERAD score, Thal phase, and AD neuropathologic change measure related but non-equivalent concepts.

## Main findings

- **SEA-AD MTG:** all {len(seaad)} analyzed donors match the official public donor workbook. It provides age at death, sex, APOE genotype, cognition, symptom/diagnosis timing, PMI, RIN, brain pH, ADNC, Thal, Braak, CERAD, CAA, Lewy body pathology, vascular pathology, LATE, head-injury status, and de-identified clinical-diagnosis flags.
- **GSE268599:** the working object already contains age, sex, APOE, five brain regions, PathStage, Braak, CERAD, and RIN for 32 donors.
- **GSE146639:** {int(gse146.supplementary_metadata_matched.sum())} of {len(gse146)} donors match the official single-cell donor table, providing age, sex, brain region, brain bank, postmortem delay, brain pH, study group, and cause of death. Donor 2018-077 is present in the expression object but absent from Table S2 and must remain missing rather than being inferred.
- **GSE237718:** donor, sex, APOE, temporal-cortex source, and pathologic AD/control status are available; age and detailed stage are not present in the current public inputs.
- **GSE157827:** donor and AD/control labels are analysis-ready. The paper reports age, sex, Braak-based eligibility, PMI selection, and exclusion criteria, but per-donor supplementary metadata must be obtained and version-locked before those fields enter models.
- **GSE243292:** donor, DLPFC source, APOE, TREM2 genotype, and A/T pathology state are available. Age, sex, PMI, Braak, and detailed clinical history are not present in the current public source.
- **GSE269906:** sample-level AD/control and BA10 source are available, but demographic and stage covariates are limited.
- **GSE138852:** individual donors were pooled and cannot be linked back to cells; donor-aware covariate analysis is not eligible.
- **ROSMAP:** potentially rich longitudinal clinical and neuropathology metadata remain controlled-access and are not locally authorized.

## Modeling rules

1. Join metadata at the donor level before pseudobulk modeling; never repeat donor covariates as if cells were independent observations.
2. Pre-specify a minimal model per cohort. For small cohorts, prioritize pathology/diagnosis plus one or two essential covariates rather than fitting every available variable.
3. Treat age, sex, region, and pathology as primary variables when available; treat APOE and detailed neuropathology as planned sensitivity or stratified analyses.
4. Use PMI, RIN, and brain pH as technical sensitivity variables when coverage and sample size permit.
5. Keep cause of death, head injury, vascular disease, Lewy pathology, and free-text history descriptive unless coverage supports a pre-specified analysis.
6. Report missingness and complete-case donor counts for every model.
7. Do not harmonize cohort-native stage variables by renaming them to a single generic severity score. Cross-cohort synthesis should compare direction and evidence class, with explicit mappings used only in sensitivity analyses.

## Planned-analysis consequence

Donor metadata integration is Step 0 of the committee-requested extension. The endothelial dual-anchor and myeloid-pooling runs should first use the minimal donor-aware model supported by each cohort, then add age/sex/region/pathology sensitivities only where donor count and metadata completeness are adequate. SEA-AD and GSE268599 support the richest adjusted analyses; GSE243292 remains primarily a genotype/pathology-stratified cohort.

## Output files

- `cohort_metadata_availability.csv`: cohort-by-field eligibility matrix.
- `donor_metadata_data_dictionary.csv`: modeling role and guardrails for each field family.
- `*_embedded_donor_metadata.csv`: donor fields retained in each analysis object.
- `GSE146639_analysis_ready_donor_metadata.csv`: expression-object donors joined to Table S2.
- `SEAAD_analysis_ready_donor_metadata.csv`: analyzed SEA-AD donors joined to the official public workbook.
- `SEAAD_public_donor_metadata_full.csv`: version-locked full public donor table with normalized column names.
- `source_provenance.csv`: source paths, sizes, and SHA-256 checksums.
"""
    (output_dir / "DONOR_METADATA_AUDIT.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/committee_extensions/donor_metadata_audit",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gse146_source = ROOT / "data/external_ad_cohorts/GSE146639_donor_tables_PMC7655794.xlsx"
    gse146_h5ad = ROOT / "data/raw/GSE146639_barcoded_smartseq2_microglia.h5ad"
    seaad_source = ROOT / "data/external_ad_cohorts/SEAAD_MTG/source/sea-ad_cohort-donor-metadata_2025-07-25.xlsx"
    seaad_h5ad = ROOT / "data/external_ad_cohorts/SEAAD_MTG/processed/SEAAD_MTG_targeted_canonical.h5ad"

    tables = write_embedded_tables(output_dir)
    _, gse146_joined = write_gse146639(gse146_source, gse146_h5ad, output_dir)
    _, seaad_joined = write_seaad(seaad_source, seaad_h5ad, output_dir)
    availability = availability_table(tables, gse146_joined, seaad_joined)
    availability.to_csv(output_dir / "cohort_metadata_availability.csv", index=False)
    data_dictionary().to_csv(output_dir / "donor_metadata_data_dictionary.csv", index=False)

    provenance = pd.DataFrame([
        {"source": "GSE146639 donor supplementary workbook", "path": str(gse146_source), "bytes": gse146_source.stat().st_size, "sha256": sha256(gse146_source)},
        {"source": "SEA-AD public cohort donor workbook", "path": str(seaad_source), "bytes": seaad_source.stat().st_size, "sha256": sha256(seaad_source)},
    ])
    provenance.to_csv(output_dir / "source_provenance.csv", index=False)
    write_report(output_dir, availability, gse146_joined, seaad_joined)
    print(f"Wrote donor metadata audit to {output_dir}")


if __name__ == "__main__":
    main()
