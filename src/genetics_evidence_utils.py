"""Human genetics evidence helpers built on GWAS Catalog and Open Targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import requests
except ImportError:
    requests = None


GWAS_BASE_URL = "https://www.ebi.ac.uk/gwas/rest/api/v2"
OPEN_TARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
DEFAULT_DISEASE_KEYWORDS = [
    "alzheimer",
    "dementia",
    "cognitive",
    "neurodegenerative",
]


@dataclass
class GeneticsEvidenceResult:
    """Outputs for the human genetics evidence layer."""

    summary_path: Path
    gwas_traits_path: Path
    gwas_associations_path: Path
    opentargets_path: Path
    summary: pd.DataFrame
    gwas_traits: pd.DataFrame
    gwas_associations: pd.DataFrame
    opentargets_diseases: pd.DataFrame
    top_genes: list[str]


def run_genetics_evidence(
    candidate_genes: list[str],
    genetics_config: dict[str, Any],
    output_dir: Path,
) -> GeneticsEvidenceResult | None:
    """Fetch disease-context genetics evidence for the candidate panel."""
    if not genetics_config.get("enabled", False):
        return None
    if requests is None:
        raise ImportError(
            "requests is required for genetics evidence. "
            "Install the full dependency set to enable this optional layer."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    disease_focus_label = str(genetics_config.get("disease_focus_label", "Alzheimer's disease")).strip()
    unique_genes = list(dict.fromkeys(str(gene) for gene in candidate_genes))
    keywords = [str(value).lower() for value in genetics_config.get("disease_keywords", DEFAULT_DISEASE_KEYWORDS)]
    gwas_trait_page_size = int(genetics_config.get("gwas_trait_page_size", 500))
    gwas_association_page_size = int(genetics_config.get("gwas_association_page_size", 200))
    gwas_max_association_pages = int(genetics_config.get("gwas_max_association_pages", 5))
    opentargets_page_size = int(genetics_config.get("opentargets_page_size", 500))
    timeout_seconds = float(genetics_config.get("timeout_seconds", 30))

    session = requests.Session()
    summary_rows: list[dict[str, object]] = []
    gwas_trait_rows: list[dict[str, object]] = []
    gwas_association_rows: list[dict[str, object]] = []
    opentargets_rows: list[dict[str, object]] = []

    for gene in unique_genes:
        gene_record = fetch_gwas_gene_record(gene, session=session, timeout_seconds=timeout_seconds)
        ensembl_id = None
        if gene_record is not None:
            ensembl_ids = gene_record.get("ensembl_gene_ids") or []
            ensembl_id = str(ensembl_ids[0]) if ensembl_ids else None

        gene_traits = fetch_matching_gwas_traits(
            gene=gene,
            keywords=keywords,
            page_size=gwas_trait_page_size,
            session=session,
            timeout_seconds=timeout_seconds,
        )
        for row in gene_traits:
            row["gene"] = gene
        gwas_trait_rows.extend(gene_traits)

        gene_associations = fetch_matching_gwas_associations(
            gene=gene,
            matching_traits=gene_traits,
            page_size=gwas_association_page_size,
            max_pages=gwas_max_association_pages,
            session=session,
            timeout_seconds=timeout_seconds,
        )
        gwas_association_rows.extend(gene_associations)

        gene_opentargets = fetch_matching_opentargets_diseases(
            gene=gene,
            ensembl_id=ensembl_id,
            keywords=keywords,
            page_size=opentargets_page_size,
            session=session,
            timeout_seconds=timeout_seconds,
        )
        opentargets_rows.extend(gene_opentargets)

        summary_rows.append(
            summarize_gene_genetics(
                gene=gene,
                ensembl_id=ensembl_id,
                gwas_traits=gene_traits,
                gwas_associations=gene_associations,
                opentargets_rows=gene_opentargets,
                disease_focus_label=disease_focus_label,
            )
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["combined_genetics_support", "opentargets_max_score", "gwas_association_count", "gene"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    gwas_traits = pd.DataFrame(gwas_trait_rows)
    gwas_associations = pd.DataFrame(gwas_association_rows)
    opentargets_diseases = pd.DataFrame(opentargets_rows)

    summary_path = output_dir / "genetics_evidence_summary.csv"
    gwas_traits_path = output_dir / "genetics_gwas_matching_traits.csv"
    gwas_associations_path = output_dir / "genetics_gwas_associations.csv"
    opentargets_path = output_dir / "genetics_opentargets_diseases.csv"

    summary.to_csv(summary_path, index=False)
    gwas_traits.to_csv(gwas_traits_path, index=False)
    gwas_associations.to_csv(gwas_associations_path, index=False)
    opentargets_diseases.to_csv(opentargets_path, index=False)

    top_genes = [
        f"{row.gene} (GWAS={int(row.gwas_association_count)}, OT={row.opentargets_max_score:.3f})"
        for row in summary.itertuples(index=False)
        if bool(row.combined_genetics_support)
    ][:5]

    return GeneticsEvidenceResult(
        summary_path=summary_path,
        gwas_traits_path=gwas_traits_path,
        gwas_associations_path=gwas_associations_path,
        opentargets_path=opentargets_path,
        summary=summary,
        gwas_traits=gwas_traits,
        gwas_associations=gwas_associations,
        opentargets_diseases=opentargets_diseases,
        top_genes=top_genes,
    )


def fetch_gwas_gene_record(
    gene: str,
    session: requests.Session,
    timeout_seconds: float,
) -> dict[str, Any] | None:
    """Fetch the GWAS Catalog gene record for a symbol if it exists."""
    response = session.get(
        f"{GWAS_BASE_URL}/genes/{gene}",
        timeout=timeout_seconds,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def fetch_matching_gwas_traits(
    gene: str,
    keywords: list[str],
    page_size: int,
    session: requests.Session,
    timeout_seconds: float,
) -> list[dict[str, object]]:
    """Fetch GWAS Catalog traits mapped to a gene and keep only disease-relevant ones."""
    response = session.get(
        f"{GWAS_BASE_URL}/efo-traits",
        params={"mapped_gene": gene, "size": page_size, "page": 0},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    traits = payload.get("_embedded", {}).get("efo_traits", [])

    matching_rows = []
    for trait in traits:
        trait_name = str(trait.get("efo_trait") or "")
        lowered = trait_name.lower()
        matched_keywords = [keyword for keyword in keywords if keyword in lowered]
        if not matched_keywords:
            continue
        matching_rows.append(
            {
                "efo_id": trait.get("efo_id"),
                "efo_trait": trait_name,
                "uri": trait.get("uri"),
                "matched_keywords": ";".join(matched_keywords),
            }
        )
    return matching_rows


def fetch_matching_gwas_associations(
    gene: str,
    matching_traits: list[dict[str, object]],
    page_size: int,
    max_pages: int,
    session: requests.Session,
    timeout_seconds: float,
) -> list[dict[str, object]]:
    """Fetch GWAS associations for the matching trait subset."""
    rows: list[dict[str, object]] = []
    for trait in matching_traits:
        efo_id = trait.get("efo_id")
        if not efo_id:
            continue
        for page in range(max_pages):
            response = session.get(
                f"{GWAS_BASE_URL}/associations",
                params={
                    "mapped_gene": gene,
                    "efo_id": efo_id,
                    "size": page_size,
                    "page": page,
                },
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            associations = payload.get("_embedded", {}).get("associations", [])
            for association in associations:
                rows.append(
                    {
                        "gene": gene,
                        "efo_id": efo_id,
                        "efo_trait": trait.get("efo_trait"),
                        "reported_trait": "; ".join(association.get("reported_trait", [])),
                        "accession_id": association.get("accession_id"),
                        "association_id": association.get("association_id"),
                        "pubmed_id": association.get("pubmed_id"),
                        "first_author": association.get("first_author"),
                        "p_value": association.get("p_value"),
                        "or_value": association.get("or_value"),
                        "beta": association.get("beta"),
                        "mapped_genes": ";".join(association.get("mapped_genes", [])),
                        "locations": ";".join(association.get("locations", [])),
                    }
                )
            page_meta = payload.get("page", {})
            total_pages = int(page_meta.get("totalPages", 0))
            if page + 1 >= total_pages:
                break
    return rows


def fetch_matching_opentargets_diseases(
    gene: str,
    ensembl_id: str | None,
    keywords: list[str],
    page_size: int,
    session: requests.Session,
    timeout_seconds: float,
) -> list[dict[str, object]]:
    """Fetch disease-relevant Open Targets disease rows for a gene."""
    if not ensembl_id:
        return []

    query = """
    query($ensemblId: String!, $pageSize: Int!) {
      target(ensemblId: $ensemblId) {
        approvedSymbol
        associatedDiseases(page: {index: 0, size: $pageSize}) {
          count
          rows {
            disease { id name }
            score
          }
        }
      }
    }
    """
    response = session.post(
        OPEN_TARGETS_GRAPHQL_URL,
        json={"query": query, "variables": {"ensemblId": ensembl_id, "pageSize": page_size}},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    target = payload.get("data", {}).get("target") or {}
    rows = target.get("associatedDiseases", {}).get("rows", [])

    matched_rows = []
    for row in rows:
        disease = row.get("disease") or {}
        disease_name = str(disease.get("name") or "")
        lowered = disease_name.lower()
        matched_keywords = [keyword for keyword in keywords if keyword in lowered]
        if not matched_keywords:
            continue
        matched_rows.append(
            {
                "gene": gene,
                "ensembl_id": ensembl_id,
                "disease_id": disease.get("id"),
                "disease_name": disease_name,
                "score": row.get("score"),
                "matched_keywords": ";".join(matched_keywords),
            }
        )
    return matched_rows


def summarize_gene_genetics(
    gene: str,
    ensembl_id: str | None,
    gwas_traits: list[dict[str, object]],
    gwas_associations: list[dict[str, object]],
    opentargets_rows: list[dict[str, object]],
    disease_focus_label: str = "Alzheimer's disease",
) -> dict[str, object]:
    """Collapse gene-level genetics evidence into one summary row."""
    best_gwas = None
    if gwas_associations:
        best_gwas = min(
            gwas_associations,
            key=lambda row: float(row.get("p_value") if row.get("p_value") is not None else 1.0),
        )
    best_ot = None
    if opentargets_rows:
        best_ot = max(
            opentargets_rows,
            key=lambda row: float(row.get("score") if row.get("score") is not None else 0.0),
        )

    return {
        "gene": gene,
        "ensembl_id": ensembl_id,
        "disease_focus_label": disease_focus_label,
        "gwas_matching_trait_count": len(gwas_traits),
        "gwas_association_count": len(gwas_associations),
        "gwas_min_p_value": best_gwas.get("p_value") if best_gwas else None,
        "gwas_best_trait": best_gwas.get("efo_trait") if best_gwas else None,
        "gwas_best_reported_trait": best_gwas.get("reported_trait") if best_gwas else None,
        "opentargets_matching_disease_count": len(opentargets_rows),
        "opentargets_max_score": best_ot.get("score") if best_ot else 0.0,
        "opentargets_best_disease": best_ot.get("disease_name") if best_ot else None,
        "combined_genetics_support": bool(gwas_associations or opentargets_rows),
    }
