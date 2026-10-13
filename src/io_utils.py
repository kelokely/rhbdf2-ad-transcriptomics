"""Input and output helpers for AnnData-based analysis runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

import pandas as pd
import yaml

from .config_utils import public_config


@dataclass
class RunPaths:
    """Filesystem locations for one pipeline run."""

    run_name: str
    run_dir: Path
    summaries_dir: Path
    tables_dir: Path
    plots_dir: Path
    enrichment_dir: Path
    pseudobulk_dir: Path
    continuous_dir: Path
    activity_proxies_dir: Path
    external_evidence_dir: Path
    report_copy_path: Path
    module_status_path: Path
    plot_status_path: Path


def create_run_paths(config: dict[str, Any]) -> RunPaths:
    """Create output folders for a pipeline run."""
    output_root = Path(config["output_root"])
    report_root = Path(config["report_root"])
    run_name = str(config["run_name"])
    run_dir = output_root / run_name
    overwrite_existing_run = bool(config.get("overwrite_existing_run", False))
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite_existing_run:
        raise FileExistsError(
            f"Run directory already exists and is not empty: {run_dir}. "
            "Set 'overwrite_existing_run: true' in the config to reuse this run folder."
        )
    if run_dir.exists() and overwrite_existing_run:
        for child in run_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    summaries_dir = run_dir / "summaries"
    tables_dir = run_dir / "tables"
    plots_dir = run_dir / "plots"
    enrichment_dir = run_dir / "enrichment"
    pseudobulk_dir = run_dir / "pseudobulk"
    continuous_dir = run_dir / "continuous"
    activity_proxies_dir = run_dir / "activity_proxies"
    external_evidence_dir = run_dir / "external_evidence"

    for path in [
        output_root,
        report_root,
        run_dir,
        summaries_dir,
        tables_dir,
        plots_dir,
        enrichment_dir,
        pseudobulk_dir,
        continuous_dir,
        activity_proxies_dir,
        external_evidence_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_name=run_name,
        run_dir=run_dir,
        summaries_dir=summaries_dir,
        tables_dir=tables_dir,
        plots_dir=plots_dir,
        enrichment_dir=enrichment_dir,
        pseudobulk_dir=pseudobulk_dir,
        continuous_dir=continuous_dir,
        activity_proxies_dir=activity_proxies_dir,
        external_evidence_dir=external_evidence_dir,
        report_copy_path=report_root / f"{run_name}.md",
        module_status_path=summaries_dir / "module_status_summary.csv",
        plot_status_path=summaries_dir / "plot_status_summary.csv",
    )


def load_h5ad(input_path: str | Path):
    """Load an H5AD file with a clear failure message if it is missing."""
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("scanpy is required to load .h5ad files.") from exc

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"H5AD input file not found: {input_path}. "
            "Place the processed dataset at the configured path or update the config."
        )
    if input_path.suffix.lower() != ".h5ad":
        raise ValueError(
            f"Unsupported input type for '{input_path}'. "
            "The stabilized v1 pipeline supports processed '.h5ad' inputs only."
        )

    try:
        return sc.read_h5ad(str(input_path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read '{input_path}' as a valid H5AD file. "
            "The file may be corrupted, incomplete, or not actually in H5AD format."
        ) from exc


def summarize_anndata(adata, gene_of_interest: str) -> dict[str, Any]:
    """Build a lightweight summary of the loaded AnnData object."""
    raw_var_names = getattr(getattr(adata, "raw", None), "var_names", [])
    return {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "obs_columns": [str(column) for column in adata.obs.columns],
        "var_columns": [str(column) for column in adata.var.columns],
        "obsm_keys": [str(key) for key in adata.obsm.keys()],
        "layers": [str(key) for key in adata.layers.keys()],
        "raw_available": bool(getattr(adata, "raw", None) is not None),
        "raw_n_genes": int(len(raw_var_names)) if getattr(adata, "raw", None) is not None else 0,
        "gene_of_interest": gene_of_interest,
        "gene_of_interest_present": bool(gene_of_interest in adata.var_names),
        "gene_of_interest_present_in_raw": bool(
            getattr(adata, "raw", None) is not None and gene_of_interest in raw_var_names
        ),
    }


def save_dataset_summary(adata, summary: dict[str, Any], run_dir: Path) -> list[Path]:
    """Persist dataset-level summary files for sanity-checking the loaded object."""
    dataset_summary_path = run_dir / "dataset_summary.txt"
    var_summary_path = run_dir / "var_summary.csv"
    obs_columns_path = run_dir / "obs_columns.txt"

    lines = [
        "AnnData Dataset Summary",
        "======================",
        f"Cells: {summary['n_cells']}",
        f"Genes: {summary['n_genes']}",
        f"Gene of interest: {summary['gene_of_interest']}",
        f"Gene of interest present: {summary['gene_of_interest_present']}",
        f"adata.raw available: {summary['raw_available']}",
        f"Gene of interest present in adata.raw: {summary['gene_of_interest_present_in_raw']}",
        "",
        "obs columns:",
        *summary["obs_columns"],
        "",
        "var columns:",
        *summary["var_columns"],
        "",
        "obsm keys:",
        *(summary["obsm_keys"] or ["<none>"]),
        "",
        "layers:",
        *(summary["layers"] or ["<none>"]),
    ]
    dataset_summary_path.write_text("\n".join(lines), encoding="utf-8")
    obs_columns_path.write_text("\n".join(summary["obs_columns"]), encoding="utf-8")

    var_summary = pd.DataFrame(
        {
            "var_column": [str(column) for column in adata.var.columns],
            "dtype": [str(dtype) for dtype in adata.var.dtypes],
            "non_null_values": [int(adata.var[column].notna().sum()) for column in adata.var.columns],
        }
    )
    var_summary.to_csv(var_summary_path, index=False)
    return [dataset_summary_path, var_summary_path, obs_columns_path]


def save_config_snapshot(config: dict[str, Any], destination: Path) -> Path:
    """Write the resolved run configuration used for the analysis."""
    destination.write_text(
        yaml.safe_dump(public_config(config), sort_keys=False),
        encoding="utf-8",
    )
    return destination


def list_relative_outputs(run_dir: Path) -> list[str]:
    """Return a deterministic list of generated files relative to the run directory."""
    return sorted(
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(run_dir).parts)
    )


def save_status_table(rows: list[dict[str, object]], destination: Path) -> Path:
    """Persist a deterministic status table for run modules or plots."""
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(list(frame.columns)).reset_index(drop=True)
    frame.to_csv(destination, index=False)
    return destination
