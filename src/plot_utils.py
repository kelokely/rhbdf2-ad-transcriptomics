"""Plot generation helpers that save figures without requiring an interactive backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def save_violin_plot(
    adata,
    gene: str,
    groupby: str,
    output_path: Path,
    plot_config: dict[str, Any] | None = None,
) -> Path | None:
    """Save a Scanpy violin plot if the grouping column exists."""
    if groupby not in adata.obs.columns:
        return None

    sc, plt = _load_plot_dependencies()
    plot_config = plot_config or {}
    dpi = int(plot_config.get("dpi", 300))
    rotation = float(plot_config.get("violin_rotation", 35))
    n_groups = _group_count(adata, groupby)
    width = max(6.0, 1.15 * n_groups + 1.5)
    height = 4.8
    try:
        ax = sc.pl.violin(
            adata,
            keys=gene,
            groupby=groupby,
            show=False,
            stripplot=False,
            rotation=rotation,
        )
        figure = ax.figure if hasattr(ax, "figure") else plt.gcf()
        figure.set_size_inches(width, height, forward=True)
        _rotate_x_labels(ax, rotation)
        figure.subplots_adjust(bottom=0.28, left=0.14, right=0.96, top=0.92)
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        return output_path
    finally:
        plt.close("all")


def save_dotplot(
    adata,
    genes: list[str],
    groupby: str,
    output_path: Path,
    plot_config: dict[str, Any] | None = None,
) -> Path | None:
    """Save a candidate-gene dotplot if there are genes and a valid grouping column."""
    if groupby not in adata.obs.columns:
        return None

    present = [gene for gene in dict.fromkeys(genes) if gene in adata.var_names]
    if not present:
        return None

    sc, plt = _load_plot_dependencies()
    plot_config = plot_config or {}
    dpi = int(plot_config.get("dpi", 300))
    rotation = float(plot_config.get("category_label_rotation", 45))
    n_groups = _group_count(adata, groupby)
    figsize = _matrix_like_figsize(n_groups=n_groups, n_features=len(present))
    try:
        dotplot = sc.pl.dotplot(
            adata,
            present,
            groupby=groupby,
            show=False,
            return_fig=True,
            figsize=figsize,
            var_group_rotation=rotation,
        )
        dotplot.savefig(output_path, dpi=dpi)
        return output_path
    finally:
        plt.close("all")


def save_matrixplot(
    adata,
    genes: list[str],
    groupby: str,
    output_path: Path,
    plot_config: dict[str, Any] | None = None,
) -> Path | None:
    """Save a candidate-gene matrixplot if there are genes and a valid grouping column."""
    if groupby not in adata.obs.columns:
        return None

    present = [gene for gene in dict.fromkeys(genes) if gene in adata.var_names]
    if not present:
        return None

    sc, plt = _load_plot_dependencies()
    plot_config = plot_config or {}
    dpi = int(plot_config.get("dpi", 300))
    rotation = float(plot_config.get("category_label_rotation", 45))
    n_groups = _group_count(adata, groupby)
    figsize = _matrix_like_figsize(n_groups=n_groups, n_features=len(present))
    try:
        matrixplot = sc.pl.matrixplot(
            adata,
            present,
            groupby=groupby,
            show=False,
            return_fig=True,
            figsize=figsize,
            var_group_rotation=rotation,
        )
        matrixplot.savefig(output_path, dpi=dpi)
        return output_path
    finally:
        plt.close("all")


def save_umap_plots_if_available(
    adata,
    genes: list[str],
    output_dir: Path,
    plot_config: dict[str, Any] | None = None,
) -> list[Path]:
    """Save UMAP plots for a small gene subset when an embedding is available."""
    if "X_umap" not in adata.obsm:
        return []

    sc, plt = _load_plot_dependencies()
    plot_config = plot_config or {}
    dpi = int(plot_config.get("dpi", 300))
    max_umap_genes = int(plot_config.get("max_umap_genes", 4))
    saved_paths: list[Path] = []
    present = [gene for gene in dict.fromkeys(genes) if gene in adata.var_names]
    for gene in present[:max_umap_genes]:
        output_path = output_dir / f"umap_{gene.lower()}.png"
        try:
            sc.pl.umap(adata, color=gene, show=False)
            figure = plt.gcf()
            figure.set_size_inches(6.8, 5.8, forward=True)
            figure.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.95)
            figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
            saved_paths.append(output_path)
        finally:
            plt.close("all")
    return saved_paths


def save_correlation_heatmap(correlations: pd.DataFrame, output_path: Path) -> Path | None:
    """Save a small heatmap summarizing candidate-gene correlations."""
    if correlations.empty:
        return None

    _, plt = _load_plot_dependencies()
    import seaborn as sns

    heatmap_df = correlations.set_index("gene")[["pearson_r", "spearman_r"]]
    plt.figure(figsize=(4.5, max(3.5, heatmap_df.shape[0] * 0.35)))
    sns.heatmap(
        heatmap_df,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("Candidate Gene Correlations With RHBDF2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close("all")
    return output_path


def _load_plot_dependencies():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scanpy as sc

    return sc, plt


def _group_count(adata, groupby: str) -> int:
    return int(pd.Series(adata.obs[groupby]).dropna().astype(str).nunique())


def _matrix_like_figsize(n_groups: int, n_features: int) -> tuple[float, float]:
    width = max(7.0, 0.48 * n_features + 0.42 * n_groups + 2.2)
    height = max(4.8, 0.42 * n_groups + 2.0)
    return (width, height)


def _rotate_x_labels(ax, rotation: float) -> None:
    if not hasattr(ax, "get_xticklabels"):
        return
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_horizontalalignment("right")
