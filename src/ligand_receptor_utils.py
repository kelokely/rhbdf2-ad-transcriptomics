"""Donor-aware node overlay on curated OmniPath ligand-receptor interactions."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

from .independent_ad_synthesis_utils import canonical_cell_type


OMNIPATH_LR_URL = "https://omnipathdb.org/interactions"


def fetch_omnipath_ligand_receptor(output_path: str | Path, timeout_seconds: int = 60) -> pd.DataFrame:
    response = requests.get(
        OMNIPATH_LR_URL,
        params={"format": "tsv", "genesymbols": 1, "datasets": "ligrecextra"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.text, encoding="utf-8")
    table = pd.read_csv(StringIO(response.text), sep="\t")
    required = {"source_genesymbol", "target_genesymbol"}
    if not required.issubset(table.columns):
        raise ValueError(f"OmniPath ligand-receptor response is missing {sorted(required - set(table.columns))}.")
    table = table.dropna(subset=["source_genesymbol", "target_genesymbol"]).copy()
    table["source_genesymbol"] = table["source_genesymbol"].astype(str)
    table["target_genesymbol"] = table["target_genesymbol"].astype(str)
    table["direct_measurement_in_study_cohorts"] = False
    return table


def build_donor_aware_lr_overlay(
    cohort_directories: dict[str, Path],
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require FDR-supported donor associations at both ends of each directed LR edge."""
    ligand_genes = set(interactions["source_genesymbol"])
    receptor_genes = set(interactions["target_genesymbol"])
    all_edges = []
    input_status = []
    for cohort, cohort_dir in cohort_directories.items():
        node_tables = _load_node_tables(cohort_dir, ligand_genes, receptor_genes)
        matrix_scope = _cohort_matrix_scope(cohort_dir)
        input_status.append(
            {
                "cohort": cohort,
                "status": (
                    f"included_{matrix_scope}"
                    if node_tables
                    else "no_primary_node_tables"
                ),
                "matrix_scope": matrix_scope,
                "node_tables": len(node_tables),
                "path": str(cohort_dir),
            }
        )
        for sender in node_tables:
            sender_nodes = sender["table"]
            if sender_nodes.empty:
                continue
            sender_nodes = sender_nodes[sender_nodes["names"].isin(ligand_genes)]
            for receiver in node_tables:
                if sender["cell_type"] == receiver["cell_type"]:
                    continue
                receiver_nodes = receiver["table"]
                receiver_nodes = receiver_nodes[receiver_nodes["names"].isin(receptor_genes)]
                if sender_nodes.empty or receiver_nodes.empty:
                    continue
                edges = interactions.merge(
                    sender_nodes[["names", "effect", "pvals_adj"]].rename(
                        columns={"names": "source_genesymbol", "effect": "ligand_effect", "pvals_adj": "ligand_padj"}
                    ),
                    on="source_genesymbol",
                    how="inner",
                ).merge(
                    receiver_nodes[["names", "effect", "pvals_adj"]].rename(
                        columns={"names": "target_genesymbol", "effect": "receptor_effect", "pvals_adj": "receptor_padj"}
                    ),
                    on="target_genesymbol",
                    how="inner",
                )
                if edges.empty:
                    continue
                edges["cohort"] = cohort
                edges["sender_cell_type"] = sender["cell_type"]
                edges["receiver_cell_type"] = receiver["cell_type"]
                edges["sender_context"] = sender["context"]
                edges["receiver_context"] = receiver["context"]
                edges["node_direction"] = np.select(
                    [
                        (edges["ligand_effect"] > 0) & (edges["receptor_effect"] > 0),
                        (edges["ligand_effect"] < 0) & (edges["receptor_effect"] < 0),
                    ],
                    ["both_positive", "both_negative"],
                    default="discordant",
                )
                edges["joint_node_strength"] = np.minimum(
                    -np.log10(edges["ligand_padj"].clip(lower=np.finfo(float).tiny)),
                    -np.log10(edges["receptor_padj"].clip(lower=np.finfo(float).tiny)),
                )
                all_edges.append(edges)
    result = pd.concat(all_edges, ignore_index=True, sort=False) if all_edges else pd.DataFrame()
    return result, pd.DataFrame(input_status)


def summarize_lr_replication(edges: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_genesymbol", "target_genesymbol", "sender_cell_type", "receiver_cell_type",
        "node_direction", "replicating_cohort_count", "replicating_cohorts",
        "best_joint_node_strength", "replication_status",
    ]
    if edges.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    group_columns = [
        "source_genesymbol", "target_genesymbol", "sender_cell_type", "receiver_cell_type", "node_direction"
    ]
    for key, frame in edges.groupby(group_columns, observed=True):
        cohorts = sorted(frame["cohort"].astype(str).unique())
        rows.append(
            {
                **dict(zip(group_columns, key, strict=False)),
                "replicating_cohort_count": len(cohorts),
                "replicating_cohorts": ";".join(cohorts),
                "best_joint_node_strength": float(frame["joint_node_strength"].max()),
                "replication_status": "replicated_two_or_more_cohorts" if len(cohorts) >= 2 else "single_cohort",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["replicating_cohort_count", "best_joint_node_strength"], ascending=[False, False]
    )


def plot_lr_replication(summary: pd.DataFrame, output_path: str | Path, maximum_edges: int = 30) -> None:
    replicated = summary[summary["replicating_cohort_count"].ge(2)].head(maximum_edges).copy()
    if replicated.empty:
        return
    replicated["interaction"] = replicated["source_genesymbol"] + " -> " + replicated["target_genesymbol"]
    replicated["cell_pair"] = replicated["sender_cell_type"] + " -> " + replicated["receiver_cell_type"]
    matrix = replicated.pivot_table(
        index="interaction", columns="cell_pair", values="replicating_cohort_count", aggfunc="max", fill_value=0
    )
    width = max(9, matrix.shape[1] * 0.55 + 4)
    height = max(6, matrix.shape[0] * 0.34 + 2)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    sns.heatmap(matrix, cmap="YlGnBu", vmin=0, annot=True, fmt=".0f", ax=ax, cbar_kws={"label": "replicating cohorts"})
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=55, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _load_node_tables(
    cohort_dir: Path, ligand_genes: set[str], receptor_genes: set[str]
) -> list[dict[str, object]]:
    required_genes = ligand_genes | receptor_genes
    tables = []
    for path in sorted(cohort_dir.glob("cell_types/**/donor_pseudobulk/donor_continuous_pseudobulk_all_genes.csv")):
        if "sensitivity_" in str(path):
            continue
        relative = path.relative_to(cohort_dir / "cell_types")
        cell_type = canonical_cell_type(relative.parts[0].replace("_", " "))
        context = "|".join(relative.parts[1:-2]) or "all"
        table = pd.read_csv(path, usecols=lambda name: name in {"names", "effect_per_log1p_rhbdf2_detection_per_1000_cells", "pvals_adj"})
        table = table.rename(columns={"effect_per_log1p_rhbdf2_detection_per_1000_cells": "effect"})
        table = table[
            table["names"].isin(required_genes)
            & pd.to_numeric(table["pvals_adj"], errors="coerce").lt(0.05)
        ].copy()
        tables.append({"cell_type": cell_type, "context": context, "table": table})
    return tables


def _cohort_matrix_scope(cohort_dir: Path) -> str:
    report = cohort_dir / "independent_cohort_report.md"
    if report.exists() and "Matrix scope: targeted" in report.read_text(encoding="utf-8"):
        return "targeted_gene_scope"
    return "full_feature_scope"
