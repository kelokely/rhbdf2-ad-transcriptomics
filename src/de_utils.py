"""Differential expression helpers built around Scanpy conventions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DEResult:
    """Differential expression results and exported file paths."""

    full_results: pd.DataFrame
    positive_results: pd.DataFrame
    negative_results: pd.DataFrame
    full_results_path: Path
    positive_path: Path
    negative_path: Path


def run_differential_expression(
    adata,
    de_config: dict[str, object],
    output_dir: Path,
) -> DEResult:
    """Run exploratory cell-level DE on normalized/log-transformed data."""
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("scanpy is required to run differential expression.") from exc

    group_column = str(de_config["group_column"])
    group_a = str(de_config["group_a"])
    group_b = str(de_config["group_b"])
    method = str(de_config.get("method", "wilcoxon"))
    top_n_export = int(de_config.get("top_n_export", 100))

    if group_column not in adata.obs.columns:
        raise ValueError(f"DE group column '{group_column}' not found in adata.obs.")

    counts = adata.obs[group_column].value_counts(dropna=False)
    if group_a not in counts or counts[group_a] == 0:
        raise ValueError(f"DE group '{group_a}' has no cells.")
    if group_b not in counts or counts[group_b] == 0:
        raise ValueError(f"DE group '{group_b}' has no cells.")

    sc.tl.rank_genes_groups(
        adata,
        groupby=group_column,
        groups=[group_a],
        reference=group_b,
        method=method,
    )
    full_results = sc.get.rank_genes_groups_df(adata, group=group_a)
    full_results = full_results.sort_values(
        ["logfoldchanges", "pvals_adj"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    comparison_stub = f"de_{group_a.lower()}_vs_{group_b.lower()}"
    full_results_path = output_dir / f"{comparison_stub}.csv"
    positive_path = output_dir / "top_positive_genes.csv"
    negative_path = output_dir / "top_negative_genes.csv"

    full_results.to_csv(full_results_path, index=False)

    positive_results = (
        full_results[full_results["logfoldchanges"] > 0]
        .sort_values(["logfoldchanges", "pvals_adj"], ascending=[False, True], na_position="last")
        .head(top_n_export)
        .reset_index(drop=True)
    )
    negative_results = (
        full_results[full_results["logfoldchanges"] < 0]
        .sort_values(["logfoldchanges", "pvals_adj"], ascending=[True, True], na_position="last")
        .head(top_n_export)
        .reset_index(drop=True)
    )

    positive_results.to_csv(positive_path, index=False)
    negative_results.to_csv(negative_path, index=False)

    return DEResult(
        full_results=full_results,
        positive_results=positive_results,
        negative_results=negative_results,
        full_results_path=full_results_path,
        positive_path=positive_path,
        negative_path=negative_path,
    )
