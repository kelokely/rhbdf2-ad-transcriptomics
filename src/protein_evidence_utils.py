"""Programmatic protein and post-translational evidence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import re
from typing import Any

import pandas as pd

try:
    import requests
except ImportError:
    requests = None


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
OMNIPATH_INTERACTIONS_URL = "https://omnipathdb.org/interactions"
OMNIPATH_ENZSUB_URL = "https://omnipathdb.org/enzsub"
PTM_FEATURE_TYPES = {
    "Modified residue",
    "Glycosylation",
    "Lipidation",
    "Disulfide bond",
    "Cross-link",
}


@dataclass
class ProteinEvidenceResult:
    """Paths and lightweight summaries for protein evidence outputs."""

    summary_path: Path
    ptm_features_path: Path
    reactome_path: Path
    candidate_network_path: Path
    candidate_ptm_path: Path
    candidate_panel_ptm_path: Path
    interaction_ptm_summary_path: Path
    literature_context_path: Path
    known_substrate_relationships_path: Path
    summary: pd.DataFrame
    ptm_features: pd.DataFrame
    candidate_network: pd.DataFrame
    candidate_ptm: pd.DataFrame
    candidate_panel_ptm: pd.DataFrame
    interaction_ptm_summary: pd.DataFrame
    literature_context_summary: pd.DataFrame
    known_substrate_relationships: pd.DataFrame
    top_ptm_genes: list[str]
    top_candidate_edges: list[str]
    top_candidate_panel_ptm_edges: list[str]
    top_literature_genes: list[str]


def run_protein_evidence(
    candidate_genes: list[str],
    protein_config: dict[str, Any],
    output_dir: Path,
) -> ProteinEvidenceResult | None:
    """Fetch and export protein-level evidence for the configured candidate gene panel."""
    if not protein_config.get("enabled", False):
        return None
    if requests is None:
        raise ImportError(
            "requests is required for protein/PTM evidence. "
            "Install the full dependency set to enable this optional layer."
        )

    unique_genes = list(dict.fromkeys(str(gene) for gene in candidate_genes))
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    protein_rows: list[dict[str, object]] = []
    ptm_rows: list[dict[str, object]] = []
    reactome_rows: list[dict[str, object]] = []

    organism_id = int(protein_config.get("organism_id", 9606))
    reviewed_only = bool(protein_config.get("reviewed_only", True))
    timeout_seconds = float(protein_config.get("timeout_seconds", 30))

    for gene in unique_genes:
        record = fetch_uniprot_record(
            gene=gene,
            organism_id=organism_id,
            reviewed_only=reviewed_only,
            timeout_seconds=timeout_seconds,
            session=session,
        )
        protein_row, gene_ptm_rows, gene_reactome_rows = parse_uniprot_record(gene, record)
        protein_rows.append(protein_row)
        ptm_rows.extend(gene_ptm_rows)
        reactome_rows.extend(gene_reactome_rows)

    protein_summary = pd.DataFrame(protein_rows)
    ptm_features = pd.DataFrame(ptm_rows)
    reactome_table = pd.DataFrame(reactome_rows)

    candidate_network = fetch_candidate_network_edges(
        candidate_genes=unique_genes,
        organism_id=organism_id,
        timeout_seconds=timeout_seconds,
    )
    candidate_ptm = fetch_candidate_ptm_edges(
        candidate_genes=unique_genes,
        organism_id=organism_id,
        timeout_seconds=timeout_seconds,
    )
    candidate_panel_ptm = candidate_ptm[
        candidate_ptm["enzyme_genesymbol"].isin(unique_genes)
        & candidate_ptm["substrate_genesymbol"].isin(unique_genes)
    ].copy()
    candidate_panel_ptm = candidate_panel_ptm.copy()
    if not candidate_panel_ptm.empty:
        candidate_panel_ptm["relationship_type"] = candidate_panel_ptm["modification"].apply(classify_ptm_relationship)

    network_summary = summarize_candidate_network(unique_genes, candidate_network, candidate_ptm, candidate_panel_ptm)
    protein_summary = protein_summary.merge(network_summary, on="gene", how="left")
    literature_context_summary = summarize_literature_context(protein_summary)
    protein_summary = protein_summary.merge(literature_context_summary, on="gene", how="left")
    interaction_ptm_summary = summarize_interaction_ptm_context(unique_genes, candidate_network, candidate_panel_ptm)
    protein_summary = protein_summary.merge(interaction_ptm_summary, on="gene", how="left")

    summary_path = output_dir / "protein_evidence_summary.csv"
    ptm_features_path = output_dir / "protein_ptm_features.csv"
    reactome_path = output_dir / "protein_reactome_cross_references.csv"
    candidate_network_path = output_dir / "protein_candidate_network_edges.csv"
    candidate_ptm_path = output_dir / "protein_candidate_ptm_edges.csv"
    candidate_panel_ptm_path = output_dir / "protein_candidate_panel_ptm_edges.csv"
    interaction_ptm_summary_path = output_dir / "protein_interaction_ptm_summary.csv"
    literature_context_path = output_dir / "protein_literature_context_summary.csv"
    known_substrate_relationships_path = output_dir / "protein_known_substrate_cleavage_relationships.csv"

    protein_summary.sort_values(["uniprot_found", "gene"], ascending=[False, True]).to_csv(summary_path, index=False)
    ptm_features.to_csv(ptm_features_path, index=False)
    reactome_table.to_csv(reactome_path, index=False)
    candidate_network.to_csv(candidate_network_path, index=False)
    candidate_ptm.to_csv(candidate_ptm_path, index=False)
    candidate_panel_ptm.to_csv(candidate_panel_ptm_path, index=False)
    interaction_ptm_summary.to_csv(interaction_ptm_summary_path, index=False)
    literature_context_summary.to_csv(literature_context_path, index=False)
    candidate_panel_ptm.to_csv(known_substrate_relationships_path, index=False)

    top_ptm = [
        f"{row.gene} ({int(row.annotated_ptm_feature_count)} PTM features)"
        for row in protein_summary.sort_values("annotated_ptm_feature_count", ascending=False).head(5).itertuples(index=False)
        if int(getattr(row, "annotated_ptm_feature_count", 0) or 0) > 0
    ]
    top_edges = [
        f"{row.source_genesymbol}->{row.target_genesymbol}"
        for row in candidate_network.head(5).itertuples(index=False)
    ]
    top_panel_ptm = [
        f"{row.enzyme_genesymbol}->{row.substrate_genesymbol} ({row.modification} {row.residue_type}{int(row.residue_offset)})"
        for row in candidate_panel_ptm.head(5).itertuples(index=False)
    ]
    top_literature = [
        f"{row.gene} ({int(row.curated_literature_pmid_count)} curated PMIDs)"
        for row in literature_context_summary.sort_values("curated_literature_pmid_count", ascending=False).head(5).itertuples(index=False)
        if int(getattr(row, "curated_literature_pmid_count", 0) or 0) > 0
    ]

    return ProteinEvidenceResult(
        summary_path=summary_path,
        ptm_features_path=ptm_features_path,
        reactome_path=reactome_path,
        candidate_network_path=candidate_network_path,
        candidate_ptm_path=candidate_ptm_path,
        candidate_panel_ptm_path=candidate_panel_ptm_path,
        interaction_ptm_summary_path=interaction_ptm_summary_path,
        literature_context_path=literature_context_path,
        known_substrate_relationships_path=known_substrate_relationships_path,
        summary=protein_summary,
        ptm_features=ptm_features,
        candidate_network=candidate_network,
        candidate_ptm=candidate_ptm,
        candidate_panel_ptm=candidate_panel_ptm,
        interaction_ptm_summary=interaction_ptm_summary,
        literature_context_summary=literature_context_summary,
        known_substrate_relationships=candidate_panel_ptm,
        top_ptm_genes=top_ptm,
        top_candidate_edges=top_edges,
        top_candidate_panel_ptm_edges=top_panel_ptm,
        top_literature_genes=top_literature,
    )


def fetch_uniprot_record(
    gene: str,
    organism_id: int,
    reviewed_only: bool,
    timeout_seconds: float,
    session: requests.Session | None = None,
) -> dict[str, Any] | None:
    """Fetch the first matching UniProtKB record for a human gene symbol."""
    session = session or requests.Session()
    query = f"gene:{gene} AND organism_id:{organism_id}"
    if reviewed_only:
        query += " AND reviewed:true"
    response = session.get(
        UNIPROT_SEARCH_URL,
        params={
            "query": query,
            "format": "json",
            "size": 1,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    return results[0] if results else None


def parse_uniprot_record(
    gene: str,
    record: dict[str, Any] | None,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Flatten one UniProt record into summary, PTM, and Reactome rows."""
    if record is None:
        return (
            {
                "gene": gene,
                "uniprot_found": False,
                "uniprot_accession": None,
                "uniprot_entry": None,
                "protein_name": None,
                "annotation_score": None,
                "protein_existence": None,
                "function_summary": None,
                "subcellular_location_summary": None,
                "disease_summary": None,
                "reactome_pathway_count": 0,
                "annotated_ptm_feature_count": 0,
                "annotated_ptm_types": None,
                "annotated_ptm_positions": None,
                "annotated_ptm_pmids": None,
            },
            [],
            [],
        )

    function_summary = summarize_uniprot_comments(record.get("comments", []), "FUNCTION")
    subcellular_summary = summarize_uniprot_subcellular_locations(record.get("comments", []))
    disease_summary = summarize_uniprot_disease_comments(record.get("comments", []))
    ptm_features = [feature for feature in record.get("features", []) if feature.get("type") in PTM_FEATURE_TYPES]
    reactome_refs = [xref for xref in record.get("uniProtKBCrossReferences", []) if xref.get("database") == "Reactome"]

    ptm_rows: list[dict[str, object]] = []
    ptm_positions: list[str] = []
    ptm_types: list[str] = []
    ptm_pmids: set[str] = set()
    for feature in ptm_features:
        start = feature.get("location", {}).get("start", {}).get("value")
        description = str(feature.get("description") or feature.get("type"))
        ptm_positions.append(f"{feature.get('type')}@{start}")
        ptm_types.append(str(feature.get("type")))
        evidence_pmids = sorted(
            {
                str(evidence.get("id"))
                for evidence in feature.get("evidences", [])
                if evidence.get("source") == "PubMed" and evidence.get("id")
            }
        )
        ptm_pmids.update(evidence_pmids)
        ptm_rows.append(
            {
                "gene": gene,
                "uniprot_accession": record.get("primaryAccession"),
                "feature_type": feature.get("type"),
                "position": start,
                "description": description,
                "evidence_pmids": ";".join(evidence_pmids),
            }
        )

    reactome_rows = []
    for ref in reactome_refs:
        pathway_name = None
        for prop in ref.get("properties", []):
            if prop.get("key") == "PathwayName":
                pathway_name = prop.get("value")
                break
        reactome_rows.append(
            {
                "gene": gene,
                "uniprot_accession": record.get("primaryAccession"),
                "reactome_id": ref.get("id"),
                "pathway_name": pathway_name,
            }
        )

    summary = {
        "gene": gene,
        "uniprot_found": True,
        "uniprot_accession": record.get("primaryAccession"),
        "uniprot_entry": record.get("uniProtkbId"),
        "protein_name": extract_uniprot_protein_name(record),
        "annotation_score": record.get("annotationScore"),
        "protein_existence": record.get("proteinExistence"),
        "function_summary": function_summary,
        "subcellular_location_summary": subcellular_summary,
        "disease_summary": disease_summary,
        "reactome_pathway_count": len(reactome_rows),
        "annotated_ptm_feature_count": len(ptm_rows),
        "annotated_ptm_types": "; ".join(sorted(dict.fromkeys(ptm_types))) if ptm_types else None,
        "annotated_ptm_positions": "; ".join(ptm_positions) if ptm_positions else None,
        "annotated_ptm_pmids": ";".join(sorted(ptm_pmids)) if ptm_pmids else None,
        "function_literature_pmids": ";".join(sorted(extract_pubmed_ids(function_summary))) if function_summary else None,
        "disease_literature_pmids": ";".join(sorted(extract_pubmed_ids(disease_summary))) if disease_summary else None,
    }
    return summary, ptm_rows, reactome_rows


def extract_uniprot_protein_name(record: dict[str, Any]) -> str | None:
    """Extract a readable protein name from a UniProt record."""
    protein_description = record.get("proteinDescription", {})
    recommended = protein_description.get("recommendedName", {})
    full_name = recommended.get("fullName", {}).get("value")
    if full_name:
        return str(full_name)
    alternatives = protein_description.get("alternativeNames", [])
    for alternative in alternatives:
        alt_full_name = alternative.get("fullName", {}).get("value")
        if alt_full_name:
            return str(alt_full_name)
    return None


def summarize_uniprot_comments(comments: list[dict[str, Any]], comment_type: str) -> str | None:
    """Join value-bearing text from comment blocks of a requested type."""
    values: list[str] = []
    for comment in comments:
        if comment.get("commentType") != comment_type:
            continue
        for text_entry in comment.get("texts", []):
            value = text_entry.get("value")
            if value:
                values.append(clean_whitespace(str(value)))
    return "; ".join(dict.fromkeys(values)) if values else None


def summarize_uniprot_subcellular_locations(comments: list[dict[str, Any]]) -> str | None:
    """Compact subcellular location comments into a readable string."""
    locations: list[str] = []
    for comment in comments:
        if comment.get("commentType") != "SUBCELLULAR LOCATION":
            continue
        for entry in comment.get("subcellularLocations", []):
            pieces = []
            location = entry.get("location", {}).get("value")
            topology = entry.get("topology", {}).get("value")
            orientation = entry.get("orientation", {}).get("value")
            note_texts = [
                clean_whitespace(str(text_entry.get("value")))
                for text_entry in entry.get("note", {}).get("texts", [])
                if text_entry.get("value")
            ]
            for value in [location, topology, orientation]:
                if value:
                    pieces.append(str(value))
            if note_texts:
                pieces.append("note: " + "; ".join(note_texts))
            if pieces:
                locations.append(" | ".join(pieces))
    return "; ".join(dict.fromkeys(locations)) if locations else None


def summarize_uniprot_disease_comments(comments: list[dict[str, Any]]) -> str | None:
    """Flatten UniProt disease comments while keeping only the high-signal text."""
    disease_text: list[str] = []
    for comment in comments:
        if comment.get("commentType") != "DISEASE":
            continue
        disease = comment.get("disease", {})
        disease_id = disease.get("diseaseId")
        description = disease.get("description")
        pieces = []
        if disease_id:
            pieces.append(str(disease_id))
        if description:
            pieces.append(clean_whitespace(str(description)))
        for text_entry in comment.get("note", {}).get("texts", []):
            value = text_entry.get("value")
            if value:
                pieces.append(clean_whitespace(str(value)))
        if pieces:
            disease_text.append(" | ".join(pieces))
    return "; ".join(dict.fromkeys(disease_text)) if disease_text else None


def fetch_candidate_network_edges(
    candidate_genes: list[str],
    organism_id: int,
    timeout_seconds: float,
) -> pd.DataFrame:
    """Fetch curated OmniPath interactions touching the candidate gene panel."""
    url = (
        f"{OMNIPATH_INTERACTIONS_URL}?genesymbols=yes&organisms={organism_id}"
        f"&partners={','.join(candidate_genes)}&fields=sources,references,curation_effort"
    )
    interactions = read_remote_tsv(url, timeout_seconds=timeout_seconds)
    if "source_genesymbol" not in interactions.columns:
        raise RuntimeError(
            "OmniPath interaction query failed. The response did not contain interaction columns."
        )
    candidate_set = set(candidate_genes)
    interactions = interactions[
        interactions["source_genesymbol"].isin(candidate_set)
        & interactions["target_genesymbol"].isin(candidate_set)
    ].copy()
    interactions = interactions.sort_values(
        ["curation_effort", "source_genesymbol", "target_genesymbol"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return interactions


def fetch_candidate_ptm_edges(
    candidate_genes: list[str],
    organism_id: int,
    timeout_seconds: float,
) -> pd.DataFrame:
    """Fetch OmniPath enzyme-substrate edges where a candidate is enzyme or substrate."""
    tables = []
    for axis_name in ["enzymes", "substrates"]:
        url = (
            f"{OMNIPATH_ENZSUB_URL}?genesymbols=yes&organisms={organism_id}"
            f"&{axis_name}={','.join(candidate_genes)}&fields=sources,references"
        )
        tables.append(read_remote_tsv(url, timeout_seconds=timeout_seconds))

    ptm = pd.concat(tables, ignore_index=True).drop_duplicates().copy()
    candidate_set = set(candidate_genes)
    ptm = ptm[
        ptm["enzyme_genesymbol"].isin(candidate_set)
        | ptm["substrate_genesymbol"].isin(candidate_set)
    ].copy()
    ptm = ptm.sort_values(
        ["enzyme_genesymbol", "substrate_genesymbol", "residue_offset", "modification"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return ptm


def summarize_candidate_network(
    candidate_genes: list[str],
    candidate_network: pd.DataFrame,
    candidate_ptm: pd.DataFrame,
    candidate_panel_ptm: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize how each candidate sits in the curated protein interaction/PTM network."""
    rows: list[dict[str, object]] = []
    for gene in candidate_genes:
        direct_edges = candidate_network[
            (candidate_network["source_genesymbol"] == gene)
            | (candidate_network["target_genesymbol"] == gene)
        ]
        partners = sorted(
            {
                *(direct_edges.loc[direct_edges["source_genesymbol"] == gene, "target_genesymbol"].tolist()),
                *(direct_edges.loc[direct_edges["target_genesymbol"] == gene, "source_genesymbol"].tolist()),
            }
        )

        ptm_as_enzyme = candidate_ptm[candidate_ptm["enzyme_genesymbol"] == gene]
        ptm_as_substrate = candidate_ptm[candidate_ptm["substrate_genesymbol"] == gene]
        panel_ptm_as_enzyme = candidate_panel_ptm[candidate_panel_ptm["enzyme_genesymbol"] == gene]
        panel_ptm_as_substrate = candidate_panel_ptm[candidate_panel_ptm["substrate_genesymbol"] == gene]

        rows.append(
            {
                "gene": gene,
                "candidate_panel_interaction_count": int(len(direct_edges)),
                "candidate_panel_partners": "; ".join(partners) if partners else None,
                "candidate_adjacent_ptm_out_count": int(len(ptm_as_enzyme)),
                "candidate_adjacent_ptm_in_count": int(len(ptm_as_substrate)),
                "candidate_panel_ptm_out_count": int(len(panel_ptm_as_enzyme)),
                "candidate_panel_ptm_in_count": int(len(panel_ptm_as_substrate)),
            }
        )
    return pd.DataFrame(rows)


def summarize_literature_context(protein_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize curated literature support embedded in UniProt/PTM annotations."""
    rows: list[dict[str, object]] = []
    for row in protein_summary.itertuples(index=False):
        function_pmids = extract_pubmed_ids(getattr(row, "function_summary", None))
        disease_pmids = extract_pubmed_ids(getattr(row, "disease_summary", None))
        ptm_pmids = extract_pubmed_ids(getattr(row, "annotated_ptm_pmids", None), allow_plain_ids=True)
        curated_pmids = sorted(function_pmids | disease_pmids | ptm_pmids)
        rows.append(
            {
                "gene": row.gene,
                "function_literature_pmid_count": len(function_pmids),
                "disease_literature_pmid_count": len(disease_pmids),
                "ptm_literature_pmid_count": len(ptm_pmids),
                "curated_literature_pmid_count": len(curated_pmids),
                "curated_literature_pmids": ";".join(curated_pmids) if curated_pmids else None,
                "literature_supported_context": bool(curated_pmids),
            }
        )
    return pd.DataFrame(rows)


def summarize_interaction_ptm_context(
    candidate_genes: list[str],
    candidate_network: pd.DataFrame,
    candidate_panel_ptm: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize within-panel interaction and substrate/cleavage evidence."""
    rows: list[dict[str, object]] = []
    for gene in candidate_genes:
        network_edges = candidate_network[
            (candidate_network["source_genesymbol"] == gene)
            | (candidate_network["target_genesymbol"] == gene)
        ]
        ptm_out = candidate_panel_ptm[candidate_panel_ptm["enzyme_genesymbol"] == gene]
        ptm_in = candidate_panel_ptm[candidate_panel_ptm["substrate_genesymbol"] == gene]
        cleavage_edges = candidate_panel_ptm[
            ((candidate_panel_ptm["enzyme_genesymbol"] == gene) | (candidate_panel_ptm["substrate_genesymbol"] == gene))
            & candidate_panel_ptm["relationship_type"].eq("cleavage_or_shedding")
        ]
        rows.append(
            {
                "gene": gene,
                "within_panel_interaction_edge_count": int(len(network_edges)),
                "known_substrate_relationship_count": int(len(ptm_out) + len(ptm_in)),
                "known_cleavage_relationship_count": int(len(cleavage_edges)),
            }
        )
    return pd.DataFrame(rows)


def classify_ptm_relationship(modification: object) -> str:
    """Reduce OmniPath PTM labels to coarse relationship classes."""
    text = str(modification or "").lower()
    if any(keyword in text for keyword in ["cleavage", "shedding", "proteol", "processing"]):
        return "cleavage_or_shedding"
    return "post_translational_modification"


def extract_pubmed_ids(value: object, allow_plain_ids: bool = False) -> set[str]:
    """Extract PubMed identifiers from UniProt text or semicolon-delimited ID strings."""
    text = str(value or "")
    if not text:
        return set()
    ids = set(re.findall(r"PubMed:(\d+)", text))
    if allow_plain_ids:
        ids.update(re.findall(r"\b\d{6,9}\b", text))
    return ids


def clean_whitespace(value: str) -> str:
    """Collapse internal whitespace for compact CSV/report output."""
    return " ".join(value.split())


def read_remote_tsv(url: str, timeout_seconds: float) -> pd.DataFrame:
    """Read a remote tab-delimited table through requests instead of urllib."""
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text), sep="\t")
