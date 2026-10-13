"""Consolidate orthogonal database context for the independent AD extension."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.brain_context_utils import (
    _build_expression_atlas_proteomics_summary,
    _fetch_hpa_record,
    _summarize_hpa_record,
)
from src.genetics_evidence_utils import run_genetics_evidence
from src.protein_evidence_utils import run_protein_evidence


STRING_NETWORK_URL = "https://string-db.org/api/tsv/network"


def run_consolidated_orthogonal_evidence(
    genes: list[str],
    output_dir: str | Path,
    expanded_evidence_dir: str | Path,
    timeout_seconds: int = 30,
) -> dict[str, Path]:
    """Run stable public-resource modules and export a per-gene coverage matrix."""
    output_dir = Path(output_dir)
    expanded_evidence_dir = Path(expanded_evidence_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    genes = list(dict.fromkeys(str(gene).strip().upper() for gene in genes if str(gene).strip()))
    statuses: list[dict[str, object]] = []

    protein_result = _run_module(
        "UniProt_Reactome_OmniPath",
        lambda: run_protein_evidence(
            genes,
            {
                "enabled": True,
                "organism_id": 9606,
                "reviewed_only": True,
                "timeout_seconds": timeout_seconds,
            },
            output_dir / "protein_network_ptm",
        ),
        statuses,
    )
    genetics_result = _run_module(
        "GWAS_Catalog_Open_Targets",
        lambda: run_genetics_evidence(
            genes,
            {
                "enabled": True,
                "disease_focus_label": "Alzheimer disease",
                "disease_keywords": ["alzheimer", "dementia", "cognitive", "neurodegenerative"],
                "gwas_trait_page_size": 500,
                "gwas_association_page_size": 200,
                "gwas_max_association_pages": 5,
                "opentargets_page_size": 500,
                "timeout_seconds": timeout_seconds,
            },
            output_dir / "human_genetics",
        ),
        statuses,
    )
    brain_result = _run_module(
        "HPA_Expression_Atlas",
        lambda: run_resilient_brain_context(
            genes,
            output_dir / "brain_proteomics_context",
            timeout_seconds,
            _ensembl_mapping(genetics_result),
        ),
        statuses,
    )
    string_table = _run_module(
        "STRING",
        lambda: fetch_string_network(genes, output_dir / "string", timeout_seconds),
        statuses,
    )

    coverage = pd.DataFrame({"gene": genes})
    if protein_result is not None:
        protein_columns = [
            "gene", "uniprot_found", "uniprot_accession", "annotation_score",
            "reactome_pathway_count", "annotated_ptm_feature_count",
            "candidate_panel_interaction_count", "candidate_panel_ptm_out_count",
            "candidate_panel_ptm_in_count", "known_substrate_relationship_count",
            "known_cleavage_relationship_count", "curated_literature_pmid_count",
        ]
        coverage = coverage.merge(
            protein_result.summary[[c for c in protein_columns if c in protein_result.summary.columns]],
            on="gene", how="left",
        )
    if genetics_result is not None:
        genetics_columns = [
            "gene", "gwas_matching_trait_count", "gwas_association_count", "gwas_min_p_value",
            "opentargets_matching_disease_count", "opentargets_max_score",
        ]
        coverage = coverage.merge(
            genetics_result.summary[[c for c in genetics_columns if c in genetics_result.summary.columns]],
            on="gene", how="left",
        )
    if isinstance(brain_result, pd.DataFrame):
        brain_columns = [
            "gene", "hpa_record_found", "hpa_microglia_nCPM", "hpa_cns_macrophage_nCPM",
            "proteomics_gene_detected", "proteomics_regions_profiled",
            "proteomics_mean_log2_fc_case_vs_control",
        ]
        coverage = coverage.merge(
            brain_result[
                [c for c in brain_columns if c in brain_result.columns]
            ],
            on="gene", how="left",
        )
    if isinstance(string_table, pd.DataFrame):
        string_counts = _string_gene_counts(string_table, genes)
        coverage = coverage.merge(string_counts, on="gene", how="left")

    expanded_summary = expanded_evidence_dir / "expanded_candidate_evidence_summary.csv"
    if expanded_summary.exists():
        expanded = pd.read_csv(expanded_summary)
        coverage = coverage.merge(expanded, on="gene", how="left", suffixes=("", "_expanded"))
        statuses.append(_status("Agora_SIGNOR_iPTMnet_MEROPS_ChEMBL_DGIdb_BioGRID_NIAGADS", "completed", "Imported expanded evidence snapshot."))
    else:
        statuses.append(_status("Agora_SIGNOR_iPTMnet_MEROPS_ChEMBL_DGIdb_BioGRID_NIAGADS", "not_run", f"Missing {expanded_summary}"))

    coverage_path = output_dir / "orthogonal_resource_coverage_by_gene.csv"
    status_path = output_dir / "orthogonal_resource_status.csv"
    manifest_path = output_dir / "orthogonal_evidence_manifest.json"
    coverage.to_csv(coverage_path, index=False)
    pd.DataFrame(statuses).to_csv(status_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_gene_count": len(genes),
                "direct_measurement_in_study_cohorts": False,
                "interpretation": "Contextual evidence only; database presence is not proof of activity, causality, or therapeutic efficacy.",
                "resources": [
                    "UniProtKB", "Reactome", "OmniPath", "STRING", "GWAS Catalog",
                    "Open Targets", "Human Protein Atlas", "Expression Atlas", "Agora",
                    "SIGNOR", "iPTMnet", "MEROPS", "ChEMBL", "DGIdb", "BioGRID", "NIAGADS",
                ],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return {"coverage": coverage_path, "status": status_path, "manifest": manifest_path}


def fetch_string_network(genes: list[str], output_dir: Path, timeout_seconds: int) -> pd.DataFrame:
    """Fetch STRING functional-network edges induced by the candidate panel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    response = requests.post(
        STRING_NETWORK_URL,
        data={
            "identifiers": "\r".join(genes),
            "species": 9606,
            "required_score": 400,
            "network_type": "functional",
            "caller_identity": "rhbdf2-independent-ad-extension",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    raw_path = output_dir / "string_network_raw.tsv"
    raw_path.write_text(response.text, encoding="utf-8")
    table = pd.read_csv(raw_path, sep="\t") if response.text.strip() else pd.DataFrame()
    if not table.empty:
        table["direct_measurement_in_transcriptomic_cohort"] = False
    table.to_csv(output_dir / "string_candidate_network_edges.csv", index=False)
    return table


def run_resilient_brain_context(
    genes: list[str],
    output_dir: Path,
    timeout_seconds: int,
    ensembl_by_gene: dict[str, str],
) -> pd.DataFrame:
    """Query HPA per gene and keep Expression Atlas independent from HPA failures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    retry = Retry(
        total=1,
        connect=1,
        read=1,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    hpa_frames = []
    status_rows = []
    for gene in genes:
        try:
            ensembl_id = ensembl_by_gene.get(gene)
            if not ensembl_id:
                raise LookupError("No Ensembl gene identifier was available from the completed genetics query.")
            record = _fetch_hpa_record(ensembl_id, session, min(timeout_seconds, 15))
            frame = pd.DataFrame([_summarize_hpa_record(gene, ensembl_id, record)])
            frame["hpa_query_status"] = "completed"
            frame["hpa_query_error"] = ""
        except Exception as exc:
            frame = pd.DataFrame(
                [{"gene": gene, "hpa_record_found": False, "hpa_query_status": "failed", "hpa_query_error": str(exc)}]
            )
        hpa_frames.append(frame)
        status_rows.append(
            {
                "resource": "Human Protein Atlas",
                "gene": gene,
                "status": frame.iloc[0]["hpa_query_status"],
                "detail": frame.iloc[0]["hpa_query_error"],
            }
        )
    hpa = pd.concat(hpa_frames, ignore_index=True, sort=False)
    try:
        proteomics, region_level = _build_expression_atlas_proteomics_summary(
            genes=genes,
            experiment_accession="E-PROT-61",
            context_label="Alzheimer disease brain proteomics",
            case_labels=["Alzheimer's disease"],
            control_labels=["normal"],
            session=session,
            timeout_seconds=timeout_seconds,
        )
        expression_status = "completed"
        expression_detail = ""
    except Exception as exc:
        proteomics = pd.DataFrame({"gene": genes, "proteomics_gene_detected": False})
        region_level = pd.DataFrame()
        expression_status = "failed"
        expression_detail = str(exc)
    status_rows.append(
        {
            "resource": "Expression Atlas E-PROT-61",
            "gene": "ALL",
            "status": expression_status,
            "detail": expression_detail,
        }
    )
    combined = pd.DataFrame({"gene": genes}).merge(hpa, on="gene", how="left").merge(
        proteomics, on="gene", how="left"
    )
    hpa.to_csv(output_dir / "brain_context_hpa_summary.csv", index=False)
    proteomics.to_csv(output_dir / "brain_context_expression_atlas_proteomics_summary.csv", index=False)
    region_level.to_csv(output_dir / "brain_context_expression_atlas_proteomics_region_level.csv", index=False)
    combined.to_csv(output_dir / "brain_context_evidence_summary.csv", index=False)
    pd.DataFrame(status_rows).to_csv(output_dir / "brain_context_resource_status.csv", index=False)
    return combined


def _string_gene_counts(table: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    rows = []
    for gene in genes:
        if table.empty or not {"preferredName_A", "preferredName_B"}.issubset(table.columns):
            count = 0
            mean_score = float("nan")
        else:
            selected = table[table["preferredName_A"].eq(gene) | table["preferredName_B"].eq(gene)]
            count = int(selected.shape[0])
            mean_score = pd.to_numeric(selected.get("score"), errors="coerce").mean()
        rows.append({"gene": gene, "string_candidate_edge_count": count, "string_mean_edge_score": mean_score})
    return pd.DataFrame(rows)


def _ensembl_mapping(genetics_result: Any) -> dict[str, str]:
    if genetics_result is None or not hasattr(genetics_result, "summary"):
        return {}
    summary = genetics_result.summary
    if summary.empty or not {"gene", "ensembl_id"}.issubset(summary.columns):
        return {}
    valid = summary.dropna(subset=["ensembl_id"])
    return dict(zip(valid["gene"].astype(str), valid["ensembl_id"].astype(str), strict=False))


def _run_module(name: str, function: Callable[[], Any], statuses: list[dict[str, object]]) -> Any:
    try:
        result = function()
        row_count = int(result.shape[0]) if isinstance(result, pd.DataFrame) else None
        statuses.append(_status(name, "completed", "", row_count))
        return result
    except Exception as exc:
        statuses.append(_status(name, "failed", str(exc)))
        return None


def _status(resource: str, status: str, detail: str, row_count: int | None = None) -> dict[str, object]:
    return {
        "resource_group": resource,
        "status": status,
        "row_count": row_count,
        "detail": detail,
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "direct_measurement_in_study_cohorts": False,
    }
