#!/usr/bin/env python3
"""Run donor-aware descriptive spatial RHBDF2 context analysis for GSE269906."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ad_cohort_utils import load_cohort_registry, prepare_registered_cohort
from src.stereo_spatial_utils import (
    compare_spatial_detection_by_pathology,
    spatial_donor_detection_summary,
    spatial_neighbor_composition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/ad_external_cohorts.yaml")
    parser.add_argument("--cohort", default="GSE269906")
    parser.add_argument("--output", default="results/independent_ad_cohorts/GSE269906_spatial")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_cohort_registry(args.registry)
    root = Path(registry["_project_root"])
    input_path = prepare_registered_cohort(registry, args.cohort, overwrite=False)
    output = _resolve(root, args.output)
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}")
        shutil.rmtree(output)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_path)
    donor_summary = spatial_donor_detection_summary(adata)
    pathology = compare_spatial_detection_by_pathology(donor_summary)
    neighborhood = spatial_neighbor_composition(adata)
    donor_summary.to_csv(tables / "rhbdf2_spatial_donor_cell_type_summary.csv", index=False)
    pathology.to_csv(tables / "rhbdf2_spatial_pathology_comparison.csv", index=False)
    neighborhood.to_csv(tables / "rhbdf2_detected_cell_neighborhood_composition.csv", index=False)
    _plot_spatial_facets(adata, figures / "rhbdf2_detection_spatial_facets.png")
    _plot_detection_by_cell_type(donor_summary, figures / "rhbdf2_detection_by_cell_type.png")
    _write_report(output, adata, donor_summary, pathology, neighborhood)
    print(f"Spatial AD context completed: {output}")
    return 0


def _plot_spatial_facets(adata: ad.AnnData, output_path: Path) -> None:
    expression = np.asarray(adata[:, "RHBDF2"].X.toarray()).reshape(-1)
    donors = sorted(adata.obs["donor_id"].astype(str).unique())
    columns = 4
    rows = math.ceil(len(donors) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.4 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(-1)
    coordinates = np.asarray(adata.obsm["spatial"])
    for axis, donor in zip(axes, donors, strict=False):
        mask = adata.obs["donor_id"].astype(str).eq(donor).to_numpy()
        index = np.flatnonzero(mask)
        background = index[:: max(1, math.ceil(len(index) / 50000))]
        detected = index[expression[index] > 0]
        axis.scatter(coordinates[background, 0], coordinates[background, 1], s=0.15, c="#d7d3c8", rasterized=True)
        axis.scatter(coordinates[detected, 0], coordinates[detected, 1], s=1.2, c="#b4462a", rasterized=True)
        pathology = adata.obs.loc[mask, "pathology"].iloc[0]
        axis.set_title(f"{donor} | {pathology}", fontsize=9)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.axis("off")
    for axis in axes[len(donors):]:
        axis.axis("off")
    fig.suptitle("RHBDF2-detected spatial cells (rust) in Stereo-seq BA10", fontsize=14)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_detection_by_cell_type(summary: pd.DataFrame, output_path: Path) -> None:
    order = summary.groupby("cell_type")["detection_fraction"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    positions = {cell_type: index for index, cell_type in enumerate(order)}
    colors = {"AD": "#b4462a", "Control": "#2f6f79"}
    offsets = {"AD": -0.12, "Control": 0.12}
    for pathology, frame in summary.groupby("pathology"):
        x = [positions[value] + offsets.get(pathology, 0) for value in frame["cell_type"]]
        ax.scatter(x, frame["detection_fraction"], s=28, alpha=0.75, label=pathology, color=colors.get(pathology))
    ax.set_xticks(range(len(order)), order, rotation=35, ha="right")
    ax.set_ylabel("Donor-level RHBDF2 detection fraction")
    ax.set_xlabel("")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_report(
    output: Path,
    adata: ad.AnnData,
    donor_summary: pd.DataFrame,
    pathology: pd.DataFrame,
    neighborhood: pd.DataFrame,
) -> None:
    fdr_hits = pathology[pathology["pvals_adj"].fillna(1.0).lt(0.05)]
    lines = [
        "# GSE269906 spatial RHBDF2 context",
        "",
        "## Dataset",
        "",
        f"- Spatial cells after full-GEF gene-count QC: {adata.n_obs:,}",
        f"- Targeted genes retained: {adata.n_vars:,}",
        f"- Donors: {adata.obs['donor_id'].nunique()} (six AD and six control expected from GEO)",
        f"- Marker-derived cell classes: {adata.obs['cell_type'].nunique()}",
        "",
        "## Factual outputs",
        "",
        f"- Donor-by-cell-type detection rows: {donor_summary.shape[0]:,}",
        f"- AD-control cell-type comparisons at FDR < 0.05: {fdr_hits.shape[0]:,}",
        f"- Descriptive RHBDF2-detected neighborhood rows: {neighborhood.shape[0]:,}",
        "",
        "## Interpretation boundary",
        "",
        "This is a targeted, marker-derived spatial-support analysis. It does not count as snRNA-seq replication, does not infer ligand-receptor signaling, and does not establish protein activity, mechanism, or causality.",
    ]
    (output / "spatial_context_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
