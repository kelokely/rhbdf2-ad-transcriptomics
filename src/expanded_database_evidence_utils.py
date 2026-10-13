"""Expanded AD, signaling, PTM, proteolysis, interaction, and drug evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import csv
from io import StringIO
from pathlib import Path
import json
import os
import re
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


AGORA_BASE = "https://agora.adknowledgeportal.org/api/v1"
IPTMNET_BASE = "https://research.bioinformatics.udel.edu/iptmnet/api"
SIGNOR_BASE = "https://signor.uniroma2.it"
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
DGIDB_URL = "https://dgidb.org/api/graphql"
MEROPS_ADAM17_URL = "https://www.ebi.ac.uk/merops/cgi-bin/pepsum?id=M12.217"
NIAGADS_GENE_URL = "https://www.niagads.org/genomics/app/record/gene/{gene}"
BIOGRID_URL = "https://webservice.thebiogrid.org/interactions/"


def run_expanded_database_evidence(
    genes: list[str],
    output_dir: str | Path,
    timeout_seconds: int = 20,
) -> dict[str, pd.DataFrame]:
    """Query all configured evidence resources and always emit status tables."""
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    unique_genes = list(dict.fromkeys(str(gene).strip().upper() for gene in genes if str(gene).strip()))
    session = _session()

    agora, gene_ids, uniprot = query_agora(session, unique_genes, raw_dir / "agora", timeout_seconds)
    signor = query_signor(session, unique_genes, uniprot, raw_dir / "signor", timeout_seconds)
    iptmnet = query_iptmnet(session, unique_genes, raw_dir / "iptmnet", timeout_seconds)
    merops = query_merops(session, unique_genes, raw_dir / "merops", timeout_seconds)
    chembl = query_chembl(session, unique_genes, raw_dir / "chembl", timeout_seconds)
    dgidb = query_dgidb(session, unique_genes, raw_dir / "dgidb", timeout_seconds)
    biogrid = query_biogrid(session, unique_genes, raw_dir / "biogrid", timeout_seconds)
    niagads = query_niagads_availability(session, unique_genes, raw_dir / "niagads", timeout_seconds)

    tables = {
        "agora": agora,
        "signor": signor,
        "iptmnet": iptmnet,
        "merops": merops,
        "chembl": chembl,
        "dgidb": dgidb,
        "biogrid": biogrid,
        "niagads": niagads,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}_evidence.csv", index=False)
    build_resource_status(tables).to_csv(output_dir / "expanded_database_status.csv", index=False)
    build_candidate_summary(unique_genes, tables).to_csv(
        output_dir / "expanded_candidate_evidence_summary.csv", index=False
    )
    (output_dir / "README.md").write_text(_evidence_readme(), encoding="utf-8")
    return tables


def query_agora(
    session: requests.Session,
    genes: list[str],
    raw_dir: Path,
    timeout: int,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, list[str]]]:
    """Retrieve AD-specific multi-omic and nomination context from Agora."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    gene_ids: dict[str, str] = {}
    uniprot: dict[str, list[str]] = {}
    for gene in genes:
        try:
            search = session.get(
                f"{AGORA_BASE}/genes/search/enhanced", params={"q": gene}, timeout=timeout
            )
            search.raise_for_status()
            hits = search.json()
            exact = next((hit for hit in hits if str(hit.get("hgnc_symbol", "")).upper() == gene), None)
            if exact is None:
                rows.append(_status_row(gene, "Agora", "not_found", "No exact HGNC-symbol match."))
                continue
            ensg = exact["id"]
            detail_response = session.get(f"{AGORA_BASE}/genes/{ensg}", timeout=timeout)
            detail_response.raise_for_status()
            detail = detail_response.json()
            _write_json(raw_dir / f"{gene}.json", detail)
            gene_ids[gene] = ensg
            uniprot[gene] = [str(item) for item in detail.get("uniprotkb_accessions", [])]
            rna = detail.get("rna_differential_expression") or []
            protein = (
                (detail.get("proteomics_LFQ") or [])
                + (detail.get("proteomics_SRM") or [])
                + (detail.get("proteomics_TMT") or [])
            )
            rows.append(
                {
                    **_status_row(gene, "Agora", "completed", ""),
                    "ensembl_gene_id": ensg,
                    "is_eqtl": detail.get("is_eqtl"),
                    "is_igap": detail.get("is_igap"),
                    "is_any_rna_changed_in_ad_brain": detail.get("is_any_rna_changed_in_ad_brain"),
                    "is_any_protein_changed_in_ad_brain": detail.get("is_any_protein_changed_in_ad_brain"),
                    "rna_results_count": len(rna),
                    "rna_fdr_lt_0_05_count": sum(
                        _numeric(item.get("adj_p_val"), 1.0) < 0.05 for item in rna
                    ),
                    "proteomics_results_count": len(protein),
                    "proteomics_fdr_lt_0_05_count": sum(
                        min(
                            _numeric(item.get("adj_p_val"), 1.0),
                            _numeric(item.get("fdr"), 1.0),
                        )
                        < 0.05
                        for item in protein
                    ),
                    "metabolomics_results_count": len(detail.get("metabolomics") or []),
                    "target_nomination_count": int(detail.get("total_nominations") or 0),
                    "target_nomination_sources": ";".join(
                        sorted({str(item.get("source")) for item in detail.get("target_nominations") or []})
                    ),
                    "bio_domains": ";".join(detail.get("biodomains") or []),
                    "pharos_class": ";".join((detail.get("druggability") or {}).get("pharos_class") or []),
                    "uniprot_accessions": ";".join(uniprot[gene]),
                }
            )
        except Exception as exc:
            rows.append(_status_row(gene, "Agora", "failed", str(exc)))
    return pd.DataFrame(rows), gene_ids, uniprot


def query_signor(
    session: requests.Session,
    genes: list[str],
    uniprot: dict[str, list[str]],
    raw_dir: Path,
    timeout: int,
) -> pd.DataFrame:
    """Retrieve signed, directional causal interactions from SIGNOR."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    status_rows = []
    for gene in genes:
        accessions = uniprot.get(gene, [])
        if not accessions:
            status_rows.append(_status_row(gene, "SIGNOR", "not_eligible", "No UniProt accession available."))
            continue
        try:
            response = session.get(
                f"{SIGNOR_BASE}/getData.php",
                params={"organism": "9606", "id": accessions[0]},
                timeout=timeout,
            )
            response.raise_for_status()
            (raw_dir / f"{gene}.tsv").write_text(response.text, encoding="utf-8")
            if response.text.strip().lower().startswith("no result found"):
                status_rows.append(_status_row(gene, "SIGNOR", "completed_empty", "No interactions returned."))
                continue
            columns = [
                "source_entity", "source_type", "source_id", "source_database",
                "target_entity", "target_type", "target_id", "target_database",
                "effect", "mechanism", "residue", "sequence", "taxon_id",
                "cell_context", "tissue_context", "context_16", "context_17",
                "context_18", "context_19", "context_20", "context_21", "pmid",
                "directness", "context_24", "curator", "evidence_sentence",
                "signor_id", "confidence_score", "context_29",
            ]
            parsed = list(csv.reader(StringIO(response.text), delimiter="\t"))
            parsed = [(row + [""] * len(columns))[: len(columns)] for row in parsed if any(row)]
            frame = pd.DataFrame(parsed, columns=columns)
            if frame.empty:
                status_rows.append(_status_row(gene, "SIGNOR", "completed_empty", "No interactions returned."))
                continue
            frame.insert(0, "gene", gene)
            frame.insert(1, "query_gene", gene)
            frame.insert(2, "resource", "SIGNOR")
            frame.insert(3, "status", "completed")
            frame["direct_measurement_in_transcriptomic_cohort"] = False
            frames.append(frame)
        except Exception as exc:
            status_rows.append(_status_row(gene, "SIGNOR", "failed", str(exc)))
    return _combine_frames_and_status(frames, status_rows)


def query_iptmnet(
    session: requests.Session,
    genes: list[str],
    raw_dir: Path,
    timeout: int,
) -> pd.DataFrame:
    """Retrieve PTM/proteoform network entries from iPTMnet."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for gene in genes:
        try:
            response = session.get(
                f"{IPTMNET_BASE}/v1/search",
                params={
                    "search_term": gene,
                    "term_type": "Protein/Gene Name",
                    "role": "Enzyme or Substrate",
                    "organism": "9606",
                },
                timeout=timeout,
            )
            response.raise_for_status()
            hits = response.json()
            _write_json(raw_dir / f"{gene}.json", hits)
            exact = [
                item
                for item in hits
                if gene in {str(value).upper() for value in item.get("gene_name", [])}
                or str(item.get("gene_name", "")).upper() == gene
            ]
            rows.append(
                {
                    **_status_row(gene, "iPTMnet", "completed", ""),
                    "search_hit_count": len(hits),
                    "exact_gene_hit_count": len(exact),
                    "uniprot_accessions": ";".join(
                        sorted({str(item.get("uniprot_ac")) for item in exact if item.get("uniprot_ac")})
                    ),
                }
            )
        except Exception as exc:
            rows.append(_status_row(gene, "iPTMnet", "failed", str(exc)))
    return pd.DataFrame(rows)


def query_merops(
    session: requests.Session,
    genes: list[str],
    raw_dir: Path,
    timeout: int,
) -> pd.DataFrame:
    """Record ADAM17 protease/substrate evidence from MEROPS."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for gene in genes:
        if gene != "ADAM17":
            rows.append(_status_row(gene, "MEROPS", "not_applicable", "Candidate is not the configured protease."))
            continue
        try:
            response = session.get(MEROPS_ADAM17_URL, timeout=timeout)
            response.raise_for_status()
            (raw_dir / "ADAM17.html").write_text(response.text, encoding="utf-8")
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", response.text))
            rows.append(
                {
                    **_status_row(gene, "MEROPS", "completed", ""),
                    "merops_identifier": "M12.217",
                    "adam17_name_present": "ADAM17" in text,
                    "substrate_section_present": "Substrate" in text,
                    "source_url": MEROPS_ADAM17_URL,
                }
            )
        except Exception as exc:
            rows.append(_status_row(gene, "MEROPS", "failed", str(exc)))
    return pd.DataFrame(rows)


def query_chembl(
    session: requests.Session,
    genes: list[str],
    raw_dir: Path,
    timeout: int,
) -> pd.DataFrame:
    """Retrieve target and measured bioactivity records from ChEMBL."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    def query_gene(gene: str) -> dict[str, object]:
        worker_session = _session()
        try:
            search = worker_session.get(
                f"{CHEMBL_BASE}/target/search.json", params={"q": gene, "limit": 20}, timeout=timeout
            )
            search.raise_for_status()
            targets = search.json().get("targets", [])
            exact = [
                target
                for target in targets
                if gene
                in {
                    str(component.get("component_synonym", "")).upper()
                    for component in target.get("target_components", [])
                    for component in component.get("target_component_synonyms", [])
                }
            ]
            selected = exact[0] if exact else (targets[0] if targets else None)
            payload: dict[str, Any] = {"targets": targets, "activities": []}
            activity_count = 0
            target_id = None
            if selected:
                target_id = selected.get("target_chembl_id")
                activity = worker_session.get(
                    f"{CHEMBL_BASE}/activity.json",
                    params={"target_chembl_id": target_id, "limit": 100},
                    timeout=timeout,
                )
                activity.raise_for_status()
                payload["activities"] = activity.json().get("activities", [])
                activity_count = len(payload["activities"])
            _write_json(raw_dir / f"{gene}.json", payload)
            return {
                **_status_row(gene, "ChEMBL", "completed" if selected else "not_found", ""),
                "target_chembl_id": target_id,
                "target_match_count": len(exact),
                "returned_activity_count": activity_count,
                "measured_bioactivity_present": activity_count > 0,
            }
        except Exception as exc:
            return _status_row(gene, "ChEMBL", "failed", str(exc))
    with ThreadPoolExecutor(max_workers=min(8, len(genes))) as executor:
        rows = list(executor.map(query_gene, genes))
    return pd.DataFrame(rows)


def query_dgidb(
    session: requests.Session,
    genes: list[str],
    raw_dir: Path,
    timeout: int,
) -> pd.DataFrame:
    """Retrieve aggregated drug-gene interaction hypotheses from DGIdb."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    query = """
    query GeneInteractions($names: [String!]!) {
      genes(names: $names) {
        nodes {
          name
          interactions {
            drug { name conceptId }
            interactionScore
            interactionTypes { type directionality }
            sources { sourceDbName }
            publications { pmid }
          }
        }
      }
    }
    """
    try:
        response = session.post(DGIDB_URL, json={"query": query, "variables": {"names": genes}}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        _write_json(raw_dir / "candidate_panel.json", payload)
        by_gene = {
            str(node.get("name", "")).upper(): node
            for node in (((payload.get("data") or {}).get("genes") or {}).get("nodes") or [])
        }
        rows = []
        for gene in genes:
            node = by_gene.get(gene, {})
            interactions = node.get("interactions") or []
            rows.append(
                {
                    **_status_row(gene, "DGIdb", "completed", ""),
                    "interaction_count": len(interactions),
                    "named_interaction_count": sum(
                        bool((item.get("drug") or {}).get("name")) for item in interactions
                    ),
                    "drugs": ";".join(
                        sorted({str((item.get("drug") or {}).get("name")) for item in interactions if (item.get("drug") or {}).get("name")})
                    ),
                    "source_databases": ";".join(
                        sorted(
                            {
                                str(source.get("sourceDbName"))
                                for item in interactions
                                for source in item.get("sources") or []
                                if source.get("sourceDbName")
                            }
                        )
                    ),
                }
            )
        return pd.DataFrame(rows)
    except Exception as exc:
        return pd.DataFrame([_status_row(gene, "DGIdb", "failed", str(exc)) for gene in genes])


def query_biogrid(
    session: requests.Session,
    genes: list[str],
    raw_dir: Path,
    timeout: int,
) -> pd.DataFrame:
    """Query BioGRID when the required access key is available."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    access_key = os.getenv("BIOGRID_ACCESS_KEY", "").strip()
    if not access_key:
        return pd.DataFrame(
            [_status_row(gene, "BioGRID", "missing_access_key", "Set BIOGRID_ACCESS_KEY to enable REST queries.") for gene in genes]
        )
    try:
        response = session.get(
            BIOGRID_URL,
            params={
                "accesskey": access_key,
                "geneList": "|".join(genes),
                "taxId": 9606,
                "format": "json",
                "includeInteractors": "true",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        _write_json(raw_dir / "candidate_panel.json", payload)
        return pd.DataFrame(
            [
                {
                    **_status_row(gene, "BioGRID", "completed", ""),
                    "panel_interaction_count": len(payload),
                }
                for gene in genes
            ]
        )
    except Exception as exc:
        return pd.DataFrame([_status_row(gene, "BioGRID", "failed", str(exc)) for gene in genes])


def query_niagads_availability(
    session: requests.Session,
    genes: list[str],
    raw_dir: Path,
    timeout: int,
) -> pd.DataFrame:
    """Record NIAGADS candidate-page availability without scraping dynamic scores."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    probe_url = NIAGADS_GENE_URL.format(gene=genes[0])
    try:
        response = session.get(probe_url, timeout=min(timeout, 10))
        available = response.ok and len(response.content) > 500
        status = "page_available" if available else "unavailable"
        reason = "No stable public candidate API; dynamic scores are not scraped."
        http_status: int | None = response.status_code
    except Exception as exc:
        status = "unavailable"
        reason = f"No stable public candidate API; service probe failed: {exc}"
        http_status = None
    rows = [
        {
            **_status_row(gene, "NIAGADS", status, reason),
            "http_status": http_status,
            "source_url": NIAGADS_GENE_URL.format(gene=gene),
        }
        for gene in genes
    ]
    return pd.DataFrame(rows)


def build_resource_status(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize resource completion without converting context into biological support."""
    rows = []
    for resource, table in tables.items():
        statuses = table["status"].astype(str).value_counts().to_dict() if "status" in table else {}
        rows.append(
            {
                "resource": resource,
                "rows": int(table.shape[0]),
                "status_counts": json.dumps({str(key): int(value) for key, value in statuses.items()}, sort_keys=True),
                "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def build_candidate_summary(genes: list[str], tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build provenance-only coverage indicators; do not create a proof score."""
    rows = []
    for gene in genes:
        row: dict[str, object] = {"gene": gene}
        for resource, table in tables.items():
            gene_rows = table[table.get("gene", pd.Series(dtype=str)).astype(str).str.upper() == gene]
            row[f"{resource}_status"] = ";".join(sorted(gene_rows["status"].astype(str).unique())) if not gene_rows.empty else "not_returned"
            row[f"{resource}_rows"] = int(gene_rows.shape[0])
            row[f"{resource}_completed_rows"] = int(
                gene_rows["status"].astype(str).eq("completed").sum()
            ) if not gene_rows.empty else 0
        rows.append(row)
    return pd.DataFrame(rows)


def _combine_frames_and_status(frames: list[pd.DataFrame], status_rows: list[dict[str, object]]) -> pd.DataFrame:
    pieces = [*frames]
    if status_rows:
        pieces.append(pd.DataFrame(status_rows))
    return pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()


def _status_row(gene: str, resource: str, status: str, reason: str) -> dict[str, object]:
    return {
        "gene": gene,
        "resource": resource,
        "status": status,
        "reason": reason,
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "direct_measurement_in_transcriptomic_cohort": False,
    }


def _session() -> requests.Session:
    retry = Retry(total=1, connect=1, read=1, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "rhbdf2-ad-evidence/1.0 (research; provenance-locked)"})
    return session


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _numeric(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _evidence_readme() -> str:
    return """# Expanded external evidence

These files contain contextual evidence retrieved from versioned or access-dated external resources. They are not measurements from the analyzed transcriptomic donors and are not counted as independent cohort replications.

- Agora: AD-specific RNA, protein, metabolite, nomination, and druggability context.
- SIGNOR: literature-curated signed causal relationships.
- iPTMnet: PTM and proteoform-network context.
- MEROPS: protease, substrate, and cleavage context for ADAM17.
- ChEMBL and DGIdb: bioactivity and drug-gene hypotheses, not treatment efficacy.
- BioGRID: requires `BIOGRID_ACCESS_KEY`; absence is reported explicitly.
- NIAGADS: candidate-page availability is recorded, but dynamic scores are not scraped because a stable public candidate API is unavailable.
"""
