"""Brain atlas, regional context, and external proteomics evidence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None

try:
    import urllib3
except ImportError:
    urllib3 = None


ENSEMBL_XREF_URL = "https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{gene}"
HPA_GENE_JSON_URL = "https://www.proteinatlas.org/{ensembl_id}.json"
EXPRESSION_ATLAS_RESULTS_URL = "https://www.ebi.ac.uk/gxa/experiments/{experiment}/Results"
EXPRESSION_ATLAS_MATRIX_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/experiments/{experiment}/{experiment}.tsv"
)


@dataclass
class BrainContextEvidenceResult:
    """Outputs for brain atlas and external proteomics evidence."""

    combined_summary: pd.DataFrame
    hpa_summary: pd.DataFrame
    proteomics_summary: pd.DataFrame
    proteomics_region_level: pd.DataFrame
    combined_summary_path: Path
    hpa_summary_path: Path
    proteomics_summary_path: Path
    proteomics_region_level_path: Path
    context_label: str
    case_labels: list[str]
    control_labels: list[str]
    top_hpa_microglia_genes: list[str]
    top_case_proteomics_genes: list[str]


def run_brain_context_evidence(
    candidate_genes: list[str],
    brain_context_config: dict[str, Any],
    output_dir: Path,
) -> BrainContextEvidenceResult | None:
    """Collect Human Protein Atlas and configurable disease-context proteomics evidence."""
    if not brain_context_config.get("enabled", True):
        return None
    if requests is None:
        raise ImportError(
            "requests is required for brain-context evidence. "
            "Install the full dependency set to enable this optional layer."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    timeout_seconds = float(brain_context_config.get("timeout_seconds", 30))
    experiment_accession = str(brain_context_config.get("expression_atlas_experiment", "E-PROT-61"))
    source_accession = str(brain_context_config.get("source_accession") or experiment_accession).strip()
    context_label = str(brain_context_config.get("context_label", "Alzheimer's disease brain proteomics")).strip()
    case_labels = _normalize_label_list(brain_context_config.get("case_labels"), default=["Alzheimer's disease"])
    control_labels = _normalize_label_list(brain_context_config.get("control_labels"), default=["normal"])
    local_matrix_path = Path(str(brain_context_config.get("local_matrix_path") or "")).expanduser()
    unique_genes = list(dict.fromkeys(str(gene) for gene in candidate_genes))

    session = requests.Session()
    if urllib3 is not None:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    hpa_summary = _build_hpa_summary(
        genes=unique_genes,
        session=session,
        timeout_seconds=timeout_seconds,
    )
    if str(local_matrix_path).strip():
        proteomics_summary, proteomics_region_level = _build_local_proteomics_summary(
            genes=unique_genes,
            matrix_path=local_matrix_path,
            source_accession=source_accession,
            context_label=context_label,
            case_labels=case_labels,
            control_labels=control_labels,
            sheet_name=_optional_str(brain_context_config.get("local_matrix_sheet_name")),
            region_column=str(brain_context_config.get("local_matrix_region_column", "Brain Region")),
            group_column=str(brain_context_config.get("local_matrix_group_column", "Group")),
            sample_id_column=str(brain_context_config.get("local_matrix_sample_id_column", "Sample ID")),
            values_are_log2=bool(brain_context_config.get("local_matrix_values_are_log2", False)),
        )
    else:
        proteomics_summary, proteomics_region_level = _build_expression_atlas_proteomics_summary(
            genes=unique_genes,
            experiment_accession=experiment_accession,
            context_label=context_label,
            case_labels=case_labels,
            control_labels=control_labels,
            session=session,
            timeout_seconds=timeout_seconds,
        )

    combined_summary = pd.DataFrame({"gene": unique_genes})
    combined_summary = combined_summary.merge(hpa_summary, on="gene", how="left")
    combined_summary = combined_summary.merge(proteomics_summary, on="gene", how="left")

    combined_summary_path = output_dir / "brain_context_evidence_summary.csv"
    hpa_summary_path = output_dir / "brain_context_hpa_summary.csv"
    proteomics_summary_path = output_dir / "brain_context_proteomics_summary.csv"
    proteomics_region_level_path = output_dir / "brain_context_proteomics_region_level.csv"

    combined_summary.to_csv(combined_summary_path, index=False)
    hpa_summary.to_csv(hpa_summary_path, index=False)
    proteomics_summary.to_csv(proteomics_summary_path, index=False)
    proteomics_region_level.to_csv(proteomics_region_level_path, index=False)

    top_hpa_microglia_genes = _summarize_top_hpa_microglia_genes(combined_summary)
    top_case_proteomics_genes = _summarize_top_proteomics_genes(combined_summary)

    return BrainContextEvidenceResult(
        combined_summary=combined_summary,
        hpa_summary=hpa_summary,
        proteomics_summary=proteomics_summary,
        proteomics_region_level=proteomics_region_level,
        combined_summary_path=combined_summary_path,
        hpa_summary_path=hpa_summary_path,
        proteomics_summary_path=proteomics_summary_path,
        proteomics_region_level_path=proteomics_region_level_path,
        context_label=context_label,
        case_labels=case_labels,
        control_labels=control_labels,
        top_hpa_microglia_genes=top_hpa_microglia_genes,
        top_case_proteomics_genes=top_case_proteomics_genes,
    )


def _build_hpa_summary(
    genes: list[str],
    session: requests.Session,
    timeout_seconds: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gene in genes:
        ensembl_id = _fetch_ensembl_gene_id(gene=gene, session=session, timeout_seconds=timeout_seconds)
        record = _fetch_hpa_record(ensembl_id=ensembl_id, session=session, timeout_seconds=timeout_seconds) if ensembl_id else None
        rows.append(_summarize_hpa_record(gene=gene, ensembl_id=ensembl_id, record=record))
    return pd.DataFrame(rows)


def _fetch_ensembl_gene_id(
    gene: str,
    session: requests.Session,
    timeout_seconds: float,
) -> str | None:
    response = session.get(
        ENSEMBL_XREF_URL.format(gene=gene),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=timeout_seconds,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        return None
    for row in payload:
        if row.get("type") == "gene" and row.get("id"):
            return str(row["id"])
    for row in payload:
        if row.get("id"):
            return str(row["id"])
    return None


def _fetch_hpa_record(
    ensembl_id: str,
    session: requests.Session,
    timeout_seconds: float,
) -> dict[str, Any] | None:
    response = session.get(
        HPA_GENE_JSON_URL.format(ensembl_id=ensembl_id),
        timeout=timeout_seconds,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def _summarize_hpa_record(
    gene: str,
    ensembl_id: str | None,
    record: dict[str, Any] | None,
) -> dict[str, object]:
    if not record:
        return {
            "gene": gene,
            "ensembl_id": ensembl_id,
            "hpa_record_found": False,
            "hpa_brain_expression_cluster": np.nan,
            "hpa_brain_regional_specificity": np.nan,
            "hpa_brain_regional_distribution": np.nan,
            "hpa_single_cell_type_specificity": np.nan,
            "hpa_single_nuclei_brain_specificity": np.nan,
            "hpa_microglia_nCPM": np.nan,
            "hpa_cns_macrophage_nCPM": np.nan,
        }

    single_cell_values = record.get("RNA single cell type specific nCPM") or {}
    single_nuclei_values = record.get("RNA single nuclei brain specific nCPM") or {}
    return {
        "gene": gene,
        "ensembl_id": ensembl_id or record.get("Ensembl"),
        "hpa_record_found": True,
        "hpa_brain_expression_cluster": record.get("Brain expression cluster"),
        "hpa_brain_regional_specificity": record.get("RNA brain regional specificity"),
        "hpa_brain_regional_distribution": record.get("RNA brain regional distribution"),
        "hpa_single_cell_type_specificity": record.get("RNA single cell type specificity"),
        "hpa_single_nuclei_brain_specificity": record.get("RNA single nuclei brain specificity"),
        "hpa_microglia_nCPM": _extract_float_from_mapping(single_cell_values, "Microglia"),
        "hpa_cns_macrophage_nCPM": _extract_matching_float(single_nuclei_values, "central nervous system macrophage"),
    }


def _extract_float_from_mapping(mapping: Any, key: str) -> float:
    if not isinstance(mapping, dict):
        return np.nan
    value = mapping.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _extract_matching_float(mapping: Any, lowered_key: str) -> float:
    if not isinstance(mapping, dict):
        return np.nan
    for key, value in mapping.items():
        if lowered_key in str(key).lower():
            try:
                return float(value)
            except (TypeError, ValueError):
                return np.nan
    return np.nan


def _build_expression_atlas_proteomics_summary(
    genes: list[str],
    experiment_accession: str,
    context_label: str,
    case_labels: list[str],
    control_labels: list[str],
    session: requests.Session,
    timeout_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_metadata = _fetch_expression_atlas_group_metadata(
        experiment_accession=experiment_accession,
        session=session,
        timeout_seconds=timeout_seconds,
    )
    response = session.get(
        EXPRESSION_ATLAS_MATRIX_URL.format(experiment=experiment_accession),
        timeout=timeout_seconds,
        verify=False,
    )
    response.raise_for_status()
    matrix = pd.read_csv(StringIO(response.text), sep="\t")

    present = matrix[matrix["Gene Name"].isin(genes)].copy()
    if present.empty:
        summary = pd.DataFrame({"gene": genes})
        summary["expression_atlas_experiment"] = experiment_accession
        summary["proteomics_gene_detected"] = False
        return summary, pd.DataFrame(
            columns=[
                "gene",
                "gene_id",
                "experiment_accession",
                "proteomics_context_label",
                "proteomics_case_labels",
                "proteomics_control_labels",
                "region",
                "case_abundance",
                "control_abundance",
                "log2_fc_case_vs_control",
            ]
        )

    value_columns = [column for column in present.columns if column.startswith("g") and ".WithInSampleAbundance" in column]
    melted = present.melt(
        id_vars=["Gene ID", "Gene Name"],
        value_vars=value_columns,
        var_name="group_column",
        value_name="abundance",
    )
    melted["group_id"] = melted["group_column"].str.split(".", n=1).str[0]
    melted["abundance"] = pd.to_numeric(melted["abundance"], errors="coerce")
    melted = melted.merge(group_metadata, on="group_id", how="left")
    melted = melted.rename(columns={"Gene ID": "gene_id", "Gene Name": "gene"})
    case_labels_lower = {label.lower() for label in case_labels}
    control_labels_lower = {label.lower() for label in control_labels}
    case_labels_joined = ";".join(case_labels)
    control_labels_joined = ";".join(control_labels)

    region_rows: list[dict[str, object]] = []
    for (gene, gene_id, region), region_df in melted.groupby(["gene", "gene_id", "region"], observed=True):
        disease_labels = region_df["disease"].astype(str).str.lower()
        case_values = region_df.loc[disease_labels.isin(case_labels_lower), "abundance"].dropna()
        control_values = region_df.loc[disease_labels.isin(control_labels_lower), "abundance"].dropna()
        case_abundance = float(case_values.mean()) if not case_values.empty else np.nan
        control_abundance = float(control_values.mean()) if not control_values.empty else np.nan
        log2_fc = np.nan
        if np.isfinite(case_abundance) and np.isfinite(control_abundance):
            log2_fc = float(np.log2((case_abundance + 1.0) / (control_abundance + 1.0)))
        region_rows.append(
            {
                "gene": gene,
                "gene_id": gene_id,
                "experiment_accession": experiment_accession,
                "proteomics_context_label": context_label,
                "proteomics_case_labels": case_labels_joined,
                "proteomics_control_labels": control_labels_joined,
                "region": region,
                "case_abundance": case_abundance,
                "control_abundance": control_abundance,
                "log2_fc_case_vs_control": log2_fc,
            }
        )
    region_level = pd.DataFrame(region_rows)

    summary_rows: list[dict[str, object]] = []
    for gene in genes:
        gene_region = region_level[region_level["gene"] == gene]
        matched = gene_region["log2_fc_case_vs_control"].dropna()
        case_abundance = gene_region["case_abundance"].dropna()
        control_abundance = gene_region["control_abundance"].dropna()
        strongest_region = np.nan
        if not matched.empty:
            strongest_region = gene_region.loc[
                gene_region["log2_fc_case_vs_control"].fillna(-np.inf).idxmax(),
                "region",
            ]
        summary_rows.append(
            {
                "gene": gene,
                "expression_atlas_experiment": experiment_accession,
                "proteomics_context_label": context_label,
                "proteomics_case_labels": case_labels_joined,
                "proteomics_control_labels": control_labels_joined,
                "proteomics_gene_detected": not gene_region.empty,
                "proteomics_regions_profiled": int(gene_region.shape[0]),
                "proteomics_regions_with_matched_case_control": int(matched.shape[0]),
                "proteomics_mean_abundance_case": float(case_abundance.mean()) if not case_abundance.empty else np.nan,
                "proteomics_mean_abundance_control": float(control_abundance.mean()) if not control_abundance.empty else np.nan,
                "proteomics_mean_log2_fc_case_vs_control": float(matched.mean()) if not matched.empty else np.nan,
                "proteomics_median_log2_fc_case_vs_control": float(matched.median()) if not matched.empty else np.nan,
                "proteomics_fraction_regions_higher_in_case": float((matched > 0).mean()) if not matched.empty else np.nan,
                "proteomics_strongest_case_region": strongest_region,
            }
    )
    return pd.DataFrame(summary_rows), region_level


def _build_local_proteomics_summary(
    genes: list[str],
    matrix_path: Path,
    source_accession: str,
    context_label: str,
    case_labels: list[str],
    control_labels: list[str],
    sheet_name: str | None,
    region_column: str,
    group_column: str,
    sample_id_column: str,
    values_are_log2: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = _read_local_proteomics_matrix(matrix_path, sheet_name)
    missing_columns = [
        column
        for column in [region_column, group_column, sample_id_column]
        if column not in matrix.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Local proteomics matrix is missing required columns: {', '.join(missing_columns)}."
        )

    metadata_columns = {region_column, group_column, sample_id_column}
    gene_columns = {str(column) for column in matrix.columns if str(column) not in metadata_columns}
    case_labels_lower = {label.lower() for label in case_labels}
    control_labels_lower = {label.lower() for label in control_labels}
    case_labels_joined = ";".join(case_labels)
    control_labels_joined = ";".join(control_labels)

    region_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for gene in genes:
        if gene not in gene_columns:
            summary_rows.append(
                {
                    "gene": gene,
                    "expression_atlas_experiment": source_accession,
                    "proteomics_context_label": context_label,
                    "proteomics_case_labels": case_labels_joined,
                    "proteomics_control_labels": control_labels_joined,
                    "proteomics_gene_detected": False,
                    "proteomics_regions_profiled": 0,
                    "proteomics_regions_with_matched_case_control": 0,
                    "proteomics_mean_abundance_case": np.nan,
                    "proteomics_mean_abundance_control": np.nan,
                    "proteomics_mean_log2_fc_case_vs_control": np.nan,
                    "proteomics_median_log2_fc_case_vs_control": np.nan,
                    "proteomics_fraction_regions_higher_in_case": np.nan,
                    "proteomics_strongest_case_region": np.nan,
                }
            )
            continue

        gene_frame = matrix[[region_column, group_column, sample_id_column, gene]].copy()
        gene_frame = gene_frame.rename(
            columns={
                region_column: "region",
                group_column: "group_label",
                sample_id_column: "sample_id",
                gene: "abundance",
            }
        )
        gene_frame["abundance"] = pd.to_numeric(gene_frame["abundance"], errors="coerce")

        gene_region_rows: list[dict[str, object]] = []
        for region, region_df in gene_frame.groupby("region", observed=True):
            labels = region_df["group_label"].astype(str).str.lower()
            case_values = region_df.loc[labels.isin(case_labels_lower), "abundance"].dropna()
            control_values = region_df.loc[labels.isin(control_labels_lower), "abundance"].dropna()
            case_abundance = float(case_values.mean()) if not case_values.empty else np.nan
            control_abundance = float(control_values.mean()) if not control_values.empty else np.nan
            log2_fc = np.nan
            if np.isfinite(case_abundance) and np.isfinite(control_abundance):
                if values_are_log2:
                    log2_fc = float(case_abundance - control_abundance)
                else:
                    log2_fc = float(np.log2((case_abundance + 1.0) / (control_abundance + 1.0)))
            gene_region_rows.append(
                {
                    "gene": gene,
                    "gene_id": gene,
                    "experiment_accession": source_accession,
                    "proteomics_context_label": context_label,
                    "proteomics_case_labels": case_labels_joined,
                    "proteomics_control_labels": control_labels_joined,
                    "region": region,
                    "case_abundance": case_abundance,
                    "control_abundance": control_abundance,
                    "log2_fc_case_vs_control": log2_fc,
                }
            )
        gene_region = pd.DataFrame(gene_region_rows)
        region_rows.extend(gene_region_rows)
        matched = gene_region["log2_fc_case_vs_control"].dropna()
        case_abundance = gene_region["case_abundance"].dropna()
        control_abundance = gene_region["control_abundance"].dropna()
        strongest_region = np.nan
        if not matched.empty:
            strongest_region = gene_region.loc[
                gene_region["log2_fc_case_vs_control"].fillna(-np.inf).idxmax(),
                "region",
            ]
        summary_rows.append(
            {
                "gene": gene,
                "expression_atlas_experiment": source_accession,
                "proteomics_context_label": context_label,
                "proteomics_case_labels": case_labels_joined,
                "proteomics_control_labels": control_labels_joined,
                "proteomics_gene_detected": not gene_region.empty,
                "proteomics_regions_profiled": int(gene_region.shape[0]),
                "proteomics_regions_with_matched_case_control": int(matched.shape[0]),
                "proteomics_mean_abundance_case": float(case_abundance.mean()) if not case_abundance.empty else np.nan,
                "proteomics_mean_abundance_control": float(control_abundance.mean()) if not control_abundance.empty else np.nan,
                "proteomics_mean_log2_fc_case_vs_control": float(matched.mean()) if not matched.empty else np.nan,
                "proteomics_median_log2_fc_case_vs_control": float(matched.median()) if not matched.empty else np.nan,
                "proteomics_fraction_regions_higher_in_case": float((matched > 0).mean()) if not matched.empty else np.nan,
                "proteomics_strongest_case_region": strongest_region,
            }
        )

    region_level = pd.DataFrame(region_rows)
    if region_level.empty:
        region_level = pd.DataFrame(
            columns=[
                "gene",
                "gene_id",
                "experiment_accession",
                "proteomics_context_label",
                "proteomics_case_labels",
                "proteomics_control_labels",
                "region",
                "case_abundance",
                "control_abundance",
                "log2_fc_case_vs_control",
            ]
        )
    return pd.DataFrame(summary_rows), region_level


def _read_local_proteomics_matrix(matrix_path: Path, sheet_name: str | None) -> pd.DataFrame:
    suffix = matrix_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(matrix_path, sheet_name=sheet_name)
    if suffix in {".csv", ".tsv", ".txt"}:
        separator = "\t" if suffix in {".tsv", ".txt"} else ","
        return pd.read_csv(matrix_path, sep=separator)
    raise ValueError(
        f"Unsupported local proteomics file type for {matrix_path}. "
        "Expected .xlsx, .xls, .csv, .tsv, or .txt."
    )


def _fetch_expression_atlas_group_metadata(
    experiment_accession: str,
    session: requests.Session,
    timeout_seconds: float,
) -> pd.DataFrame:
    response = session.get(
        EXPRESSION_ATLAS_RESULTS_URL.format(experiment=experiment_accession),
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    html = response.text

    region_groups = _extract_groupings(html, "ORGANISM_PART")
    disease_groups = _extract_groupings(html, "DISEASE")
    rows: list[dict[str, object]] = []
    for region, group_ids in region_groups:
        for group_id in group_ids:
            rows.append({"group_id": group_id, "region": region})
    metadata = pd.DataFrame(rows)

    disease_rows: list[dict[str, object]] = []
    for disease, group_ids in disease_groups:
        for group_id in group_ids:
            disease_rows.append({"group_id": group_id, "disease": disease})
    disease_metadata = pd.DataFrame(disease_rows)
    if metadata.empty:
        return pd.DataFrame(columns=["group_id", "region", "disease"])
    return metadata.merge(disease_metadata, on="group_id", how="left")


def _extract_groupings(html: str, group_name: str) -> list[tuple[str, list[str]]]:
    marker = f'"name":"{group_name}"'
    marker_index = html.find(marker)
    if marker_index == -1:
        return []

    groupings_index = html.find('"groupings":[', marker_index)
    if groupings_index == -1:
        return []
    bracket_start = html.find("[", groupings_index + len('"groupings":'))
    if bracket_start == -1:
        return []
    bracket_end = _find_matching_bracket(html, bracket_start)
    if bracket_end == -1:
        return []
    groupings_payload = html[bracket_start : bracket_end + 1]
    try:
        raw_groupings = json.loads(groupings_payload)
    except json.JSONDecodeError:
        return []

    parsed: list[tuple[str, list[str]]] = []
    for item in raw_groupings:
        if not isinstance(item, list) or len(item) != 2:
            continue
        group_label = str(item[0])
        group_ids = [str(value) for value in item[1]]
        parsed.append((group_label, group_ids))
    return parsed


def _find_matching_bracket(text: str, start_index: int) -> int:
    depth = 0
    in_string = False
    escape = False
    for index in range(start_index, len(text)):
        character = text[index]
        if in_string:
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _summarize_top_hpa_microglia_genes(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return []
    sortable = summary.copy()
    sortable["hpa_microglia_context_score"] = sortable[
        ["hpa_microglia_nCPM", "hpa_cns_macrophage_nCPM"]
    ].max(axis=1, skipna=True)
    sortable = sortable.sort_values("hpa_microglia_context_score", ascending=False, na_position="last")
    return [
        f"{row.gene} ({row.hpa_microglia_context_score:.1f})"
        for row in sortable.head(5).itertuples(index=False)
        if pd.notna(row.hpa_microglia_context_score)
    ]


def _summarize_top_proteomics_genes(summary: pd.DataFrame) -> list[str]:
    if summary.empty or "proteomics_mean_log2_fc_case_vs_control" not in summary.columns:
        return []
    sortable = summary.sort_values(
        "proteomics_mean_log2_fc_case_vs_control",
        ascending=False,
        na_position="last",
    )
    return [
        f"{row.gene} ({row.proteomics_mean_log2_fc_case_vs_control:.2f})"
        for row in sortable.head(5).itertuples(index=False)
        if pd.notna(row.proteomics_mean_log2_fc_case_vs_control)
    ]


def _normalize_label_list(values: Any, default: list[str]) -> list[str]:
    if not isinstance(values, list) or not values:
        return list(default)
    normalized = [str(value).strip() for value in values if str(value).strip()]
    return normalized or list(default)


def _optional_str(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
