"""Helpers for exporting enrichment-ready gene lists and database-backed enrichments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from gprofiler import GProfiler
except ImportError:
    GProfiler = None

try:
    import gseapy as gp
except ImportError:
    gp = None


@dataclass
class SingleEnrichmentResult:
    """Result bundle for one enrichment query."""

    label: str
    query_genes: list[str]
    results: pd.DataFrame
    results_path: Path
    summary_path: Path
    top_terms: list[str]


@dataclass
class EnrichmentBundleResult:
    """Up/down enrichment result bundle."""

    backend_name: str
    mode_label: str
    up: SingleEnrichmentResult
    down: SingleEnrichmentResult


def export_enrichment_gene_lists(
    positive_results: pd.DataFrame,
    negative_results: pd.DataFrame,
    output_dir: Path,
    prefix: str = "",
) -> list[Path]:
    """Export clean up and down gene lists as plain text."""
    output_dir.mkdir(parents=True, exist_ok=True)

    up_path = output_dir / f"{prefix}top_up_genes.txt"
    down_path = output_dir / f"{prefix}top_down_genes.txt"

    up_genes = _extract_gene_names(positive_results)
    down_genes = _extract_gene_names(negative_results)

    up_path.write_text("\n".join(up_genes), encoding="utf-8")
    down_path.write_text("\n".join(down_genes), encoding="utf-8")
    return [up_path, down_path]


def export_ranked_gene_lists(
    results: pd.DataFrame,
    output_dir: Path,
    effect_column: str,
    padj_column: str | None,
    prefix: str = "",
    gene_column: str = "names",
) -> list[Path]:
    """Export GSEA-ready ranked gene lists from a full result table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    effect_path = output_dir / f"{prefix}ranked_genes_by_{_sanitize_stub(effect_column)}.rnk"
    effect_table = _build_ranked_table(
        results=results,
        gene_column=gene_column,
        score=_coerce_numeric(results.get(effect_column)),
    )
    effect_table.to_csv(effect_path, sep="\t", index=False, header=False)
    written_paths.append(effect_path)

    if padj_column and padj_column in results.columns:
        signed_score = _build_signed_neg_log10_score(
            effect=_coerce_numeric(results.get(effect_column)),
            padj=_coerce_numeric(results.get(padj_column)),
        )
        signed_path = output_dir / f"{prefix}ranked_genes_by_signed_neg_log10_{_sanitize_stub(padj_column)}.rnk"
        signed_table = _build_ranked_table(
            results=results,
            gene_column=gene_column,
            score=signed_score,
        )
        signed_table.to_csv(signed_path, sep="\t", index=False, header=False)
        written_paths.append(signed_path)

    return written_paths


def _extract_gene_names(df: pd.DataFrame) -> list[str]:
    if "names" not in df.columns:
        return []
    return [str(value) for value in df["names"].dropna().astype(str).tolist()]


def _build_ranked_table(
    results: pd.DataFrame,
    gene_column: str,
    score: pd.Series,
) -> pd.DataFrame:
    if gene_column not in results.columns:
        return pd.DataFrame(columns=["gene", "score"])
    ranked = pd.DataFrame(
        {
            "gene": results[gene_column].astype(str),
            "score": score,
        }
    )
    ranked = ranked.dropna(subset=["gene", "score"])
    ranked = ranked.drop_duplicates(subset=["gene"], keep="first")
    ranked = ranked.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
    return ranked


def _build_signed_neg_log10_score(effect: pd.Series, padj: pd.Series) -> pd.Series:
    clipped = padj.clip(lower=1e-300, upper=1.0)
    return np.sign(effect.fillna(0.0)) * (-np.log10(clipped))


def _coerce_numeric(values) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(values, errors="coerce")


def _sanitize_stub(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(value)).strip("_").lower()


def run_gprofiler_enrichment(
    positive_results: pd.DataFrame,
    negative_results: pd.DataFrame,
    enrichment_config: dict[str, Any],
    output_dir: Path,
    background_genes: list[str] | None = None,
    prefix: str = "",
) -> EnrichmentBundleResult | None:
    """Run g:Profiler enrichment for up and down gene lists."""
    if not enrichment_config.get("enabled", True):
        return None
    if GProfiler is None:
        raise ImportError(
            "gprofiler-official is required for database-backed enrichment. "
            "Install it with: pip install gprofiler-official"
        )

    up_genes = _extract_gene_names(positive_results)
    down_genes = _extract_gene_names(negative_results)
    gp = GProfiler(return_dataframe=True)

    up_result = _run_single_gprofiler_query(
        gp=gp,
        query_genes=up_genes,
        enrichment_config=enrichment_config,
        output_dir=output_dir,
        background_genes=background_genes,
        file_stub=f"{prefix}gprofiler_up_enrichment",
        label=f"{prefix}up".strip("_"),
    )
    down_result = _run_single_gprofiler_query(
        gp=gp,
        query_genes=down_genes,
        enrichment_config=enrichment_config,
        output_dir=output_dir,
        background_genes=background_genes,
        file_stub=f"{prefix}gprofiler_down_enrichment",
        label=f"{prefix}down".strip("_"),
    )
    return EnrichmentBundleResult(
        backend_name="gprofiler",
        mode_label=prefix.rstrip("_") or "cell_level",
        up=up_result,
        down=down_result,
    )


def run_enrichr_enrichment(
    positive_results: pd.DataFrame,
    negative_results: pd.DataFrame,
    enrichr_config: dict[str, Any],
    output_dir: Path,
    background_genes: list[str] | None = None,
    prefix: str = "",
) -> EnrichmentBundleResult | None:
    """Run Enrichr-backed enrichment for Hallmark and cell-type style libraries."""
    if not enrichr_config.get("enabled", True):
        return None
    if gp is None:
        raise ImportError("gseapy is required for Enrichr-backed enrichment. Install it with: pip install gseapy")

    up_result = _run_single_enrichr_query(
        query_genes=_extract_gene_names(positive_results),
        enrichr_config=enrichr_config,
        output_dir=output_dir,
        background_genes=background_genes,
        file_stub=f"{prefix}enrichr_up_enrichment",
        label=f"{prefix}up".strip("_"),
    )
    down_result = _run_single_enrichr_query(
        query_genes=_extract_gene_names(negative_results),
        enrichr_config=enrichr_config,
        output_dir=output_dir,
        background_genes=background_genes,
        file_stub=f"{prefix}enrichr_down_enrichment",
        label=f"{prefix}down".strip("_"),
    )
    return EnrichmentBundleResult(
        backend_name="enrichr",
        mode_label=prefix.rstrip("_") or "cell_level",
        up=up_result,
        down=down_result,
    )


def _run_single_gprofiler_query(
    gp: GProfiler,
    query_genes: list[str],
    enrichment_config: dict[str, Any],
    output_dir: Path,
    background_genes: list[str] | None,
    file_stub: str,
    label: str,
) -> SingleEnrichmentResult:
    unique_query = list(dict.fromkeys(query_genes))
    background = None
    if enrichment_config.get("use_dataset_background", True) and background_genes:
        background = list(dict.fromkeys(background_genes))
    domain_scope = "custom" if background else str(enrichment_config.get("domain_scope", "annotated"))

    results_path = output_dir / f"{file_stub}.csv"
    summary_path = output_dir / f"{file_stub}_summary.yaml"

    if not unique_query:
        empty = pd.DataFrame()
        _write_enrichment_summary(
            summary_path,
            {
                "label": label,
                "query_gene_count": 0,
                "background_gene_count": len(background or []),
                "sources": enrichment_config["sources"],
                "note": "No genes were available for enrichment.",
            },
        )
        empty.to_csv(results_path, index=False)
        return SingleEnrichmentResult(
            label=label,
            query_genes=[],
            results=empty,
            results_path=results_path,
            summary_path=summary_path,
            top_terms=[],
        )

    try:
        results = gp.profile(
            organism=str(enrichment_config.get("organism", "hsapiens")),
            query=unique_query,
            sources=list(enrichment_config["sources"]),
            user_threshold=float(enrichment_config.get("user_threshold", 0.05)),
            significance_threshold_method=str(
                enrichment_config.get("significance_threshold_method", "g_SCS")
            ),
            domain_scope=domain_scope,
            background=background,
            no_iea=bool(enrichment_config.get("no_iea", False)),
        )
    except AssertionError as exc:
        raise RuntimeError(f"g:Profiler enrichment failed for '{label}': {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"g:Profiler enrichment failed for '{label}': {exc}") from exc

    if not results.empty:
        results = results.sort_values("p_value", ascending=True, na_position="last").reset_index(drop=True)
    results.to_csv(results_path, index=False)

    top_terms = _format_top_terms(results, int(enrichment_config.get("max_terms_report", 5)))
    _write_enrichment_summary(
        summary_path,
        {
            "label": label,
            "query_gene_count": len(unique_query),
            "background_gene_count": len(background or []),
            "sources": enrichment_config["sources"],
            "top_terms": top_terms,
        },
    )
    return SingleEnrichmentResult(
        label=label,
        query_genes=unique_query,
        results=results,
        results_path=results_path,
        summary_path=summary_path,
        top_terms=top_terms,
    )


def _run_single_enrichr_query(
    query_genes: list[str],
    enrichr_config: dict[str, Any],
    output_dir: Path,
    background_genes: list[str] | None,
    file_stub: str,
    label: str,
) -> SingleEnrichmentResult:
    unique_query = _normalize_gene_list(query_genes)
    results_path = output_dir / f"{file_stub}.csv"
    summary_path = output_dir / f"{file_stub}_summary.yaml"
    requested_background = (
        _normalize_gene_list(background_genes)
        if enrichr_config.get("use_dataset_background", True) and background_genes
        else []
    )

    if not unique_query:
        empty = pd.DataFrame()
        _write_enrichment_summary(
            summary_path,
            {
                "label": label,
                "query_gene_count": 0,
                "background_gene_count": len(background_genes or []),
                "libraries": enrichr_config["libraries"],
                "note": "No genes were available for enrichment.",
            },
        )
        empty.to_csv(results_path, index=False)
        return SingleEnrichmentResult(
            label=label,
            query_genes=[],
            results=empty,
            results_path=results_path,
            summary_path=summary_path,
            top_terms=[],
        )

    background = requested_background or None
    fallback_note = None
    try:
        results = _execute_enrichr_query(
            query_genes=unique_query,
            enrichr_config=enrichr_config,
            background=background,
        )
    except Exception as exc:
        if background is None:
            raise RuntimeError(f"Enrichr enrichment failed for '{label}': {exc}") from exc
        try:
            results = _execute_enrichr_query(
                query_genes=unique_query,
                enrichr_config=enrichr_config,
                background=None,
            )
        except Exception as retry_exc:
            raise RuntimeError(
                f"Enrichr enrichment failed for '{label}' with and without dataset background. "
                f"Background error: {exc}. Retry error: {retry_exc}"
            ) from retry_exc
        fallback_note = (
            "Custom Enrichr background submission failed and the query was retried "
            f"without dataset background. Requested background genes: {len(requested_background)}. "
            f"Original error: {exc}"
        )
    if not results.empty:
        results = results.sort_values("Adjusted P-value", ascending=True, na_position="last").reset_index(drop=True)
    results.to_csv(results_path, index=False)

    top_terms = _format_enrichr_top_terms(results, int(enrichr_config.get("max_terms_report", 5)))
    _write_enrichment_summary(
        summary_path,
        {
            "label": label,
            "query_gene_count": len(unique_query),
            "background_gene_count_requested": len(requested_background),
            "background_gene_count_used": len(background or []),
            "background_fallback_applied": fallback_note is not None,
            "libraries": enrichr_config["libraries"],
            "top_terms": top_terms,
            "note": fallback_note,
        },
    )
    return SingleEnrichmentResult(
        label=label,
        query_genes=unique_query,
        results=results,
        results_path=results_path,
        summary_path=summary_path,
        top_terms=top_terms,
    )


def _execute_enrichr_query(
    query_genes: list[str],
    enrichr_config: dict[str, Any],
    background: list[str] | None,
) -> pd.DataFrame:
    result = gp.enrichr(
        gene_list=query_genes,
        gene_sets=list(enrichr_config["libraries"]),
        organism=str(enrichr_config.get("organism", "human")),
        outdir=None,
        background=background,
        cutoff=float(enrichr_config.get("cutoff", 0.05)),
        no_plot=True,
        verbose=False,
    )
    return result.results.copy()


def _normalize_gene_list(genes: list[str] | None) -> list[str]:
    if not genes:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for gene in genes:
        value = str(gene).strip()
        if not value or value.lower() == "nan" or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def _format_top_terms(results: pd.DataFrame, max_terms: int) -> list[str]:
    if results.empty:
        return []
    return [
        f"{row.source}:{row.name} (p={row.p_value:.2e}, intersection={row.intersection_size})"
        for row in results.head(max_terms).itertuples(index=False)
    ]


def _format_enrichr_top_terms(results: pd.DataFrame, max_terms: int) -> list[str]:
    if results.empty:
        return []
    return [
        f"{row.get('Gene_set', 'unknown')}:{row.get('Term', 'unknown term')} "
        f"(adj_p={float(row.get('Adjusted P-value', 1.0)):.2e}, overlap={row.get('Overlap', 'n/a')})"
        for _, row in results.head(max_terms).iterrows()
    ]


def _write_enrichment_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
