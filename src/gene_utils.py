"""Gene-level helpers for sparse and dense AnnData matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr


def ensure_unique_var_names(adata) -> None:
    """Raise a clear error when duplicate gene names are present."""
    duplicated = pd.Index(adata.var_names).duplicated(keep=False)
    if not np.any(duplicated):
        return
    duplicate_names = pd.Index(adata.var_names)[duplicated].astype(str).unique().tolist()
    preview = ", ".join(duplicate_names[:10])
    raise ValueError(
        "Duplicate gene names were found in adata.var_names. "
        f"Resolve duplicates before running the pipeline. Example duplicates: {preview}"
    )


def check_gene_presence(adata, gene: str) -> bool:
    """Return True when a gene is present in adata.var_names."""
    return bool(gene in adata.var_names)


def gene_presence_table(adata, genes: list[str]) -> pd.DataFrame:
    """Build a table showing which requested genes are present."""
    unique_genes = list(dict.fromkeys(genes))
    return pd.DataFrame(
        [{"gene": gene, "present": check_gene_presence(adata, gene)} for gene in unique_genes]
    )


def require_gene_present(adata, gene: str) -> None:
    """Raise a clear error if the gene of interest is absent."""
    if not check_gene_presence(adata, gene):
        raise ValueError(
            f"Gene of interest '{gene}' was not found in adata.var_names. "
            "Check the dataset's gene naming scheme before continuing."
        )


def extract_expression_vector(adata, gene: str) -> np.ndarray:
    """Safely extract a 1D expression vector for a gene from sparse or dense matrices."""
    matches = np.flatnonzero(adata.var_names.to_numpy() == gene)
    if matches.size == 0:
        raise KeyError(f"Gene not found in adata.var_names: {gene}")
    if matches.size > 1:
        raise ValueError(f"Gene '{gene}' appears multiple times in adata.var_names.")

    matrix = adata[:, matches[0]].X
    if sparse.issparse(matrix):
        values = matrix.toarray()
    else:
        values = np.asarray(matrix)
    return np.asarray(values, dtype=float).reshape(-1)


def summarize_candidate_genes_by_group(
    adata,
    genes: list[str],
    group_column: str,
    output_path: Path,
) -> pd.DataFrame:
    """Summarize mean expression and detection rate for candidate genes by group."""
    if group_column not in adata.obs.columns:
        raise ValueError(f"Group column '{group_column}' not found in adata.obs.")

    present_genes = [gene for gene in dict.fromkeys(genes) if check_gene_presence(adata, gene)]
    group_values = pd.Series(adata.obs[group_column]).dropna().astype(str).unique().tolist()
    rows: list[dict[str, object]] = []
    for gene in present_genes:
        expression = extract_expression_vector(adata, gene)
        for group_name in group_values:
            mask = adata.obs[group_column].astype(str) == group_name
            grouped_expression = expression[np.asarray(mask)]
            rows.append(
                {
                    "group_column": group_column,
                    "group_name": str(group_name),
                    "gene": gene,
                    "mean_expression": float(np.mean(grouped_expression)) if grouped_expression.size else np.nan,
                    "median_expression": float(np.median(grouped_expression)) if grouped_expression.size else np.nan,
                    "fraction_expressing": float(np.mean(grouped_expression > 0)) if grouped_expression.size else np.nan,
                    "n_cells": int(grouped_expression.size),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)
    return summary


def compute_candidate_gene_correlations(
    adata,
    anchor_gene: str,
    candidate_genes: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations against the anchor gene."""
    require_gene_present(adata, anchor_gene)
    present_genes = [gene for gene in dict.fromkeys(candidate_genes) if check_gene_presence(adata, gene)]

    anchor_values = extract_expression_vector(adata, anchor_gene)
    rows: list[dict[str, object]] = []
    for gene in present_genes:
        gene_values = extract_expression_vector(adata, gene)
        if np.std(anchor_values) == 0 or np.std(gene_values) == 0:
            pearson_r = np.nan
            spearman_r = np.nan
        else:
            pearson_r = float(np.corrcoef(anchor_values, gene_values)[0, 1])
            spearman_r = float(spearmanr(anchor_values, gene_values).statistic)
        rows.append(
            {
                "gene": gene,
                "pearson_r": pearson_r,
                "spearman_r": spearman_r,
            }
        )

    correlations = (
        pd.DataFrame(rows)
        .sort_values(["pearson_r", "spearman_r"], ascending=[False, False], na_position="last")
        .reset_index(drop=True)
    )
    correlations.to_csv(output_path, index=False)
    return correlations
