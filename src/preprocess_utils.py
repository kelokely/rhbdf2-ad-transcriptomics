"""Preprocessing helpers for normalized and log-transformed exploratory analysis."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import sparse


@dataclass
class PreprocessingResult:
    """Normalized AnnData plus a small preprocessing summary."""

    adata: object
    source_adata: object
    summary: dict[str, Any]
    summary_path: Path


def prepare_analysis_anndata(adata, preprocessing_config: dict[str, object], output_dir: Path) -> PreprocessingResult:
    """Create a normalized/log1p copy of the AnnData object for analysis."""
    try:
        import anndata as ad
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("scanpy is required to preprocess AnnData objects.") from exc

    source_adata, source_summary = _select_analysis_matrix(adata=adata, preprocessing_config=preprocessing_config, ad_module=ad)
    processed = source_adata.copy()
    normalize_total = bool(preprocessing_config.get("normalize_total", True))
    log1p = bool(preprocessing_config.get("log1p", True))
    target_sum = float(preprocessing_config.get("target_sum", 10000))

    pre_counts = _row_sums(processed.X)
    summary = {
        **source_summary,
        "normalize_total": normalize_total,
        "target_sum": target_sum if normalize_total else None,
        "log1p": log1p,
        "input_total_counts_min": float(np.min(pre_counts)),
        "input_total_counts_median": float(np.median(pre_counts)),
        "input_total_counts_mean": float(np.mean(pre_counts)),
        "input_total_counts_max": float(np.max(pre_counts)),
    }

    if normalize_total:
        sc.pp.normalize_total(processed, target_sum=target_sum)
        normalized_counts = _row_sums(processed.X)
        summary["post_normalize_total_counts_min"] = float(np.min(normalized_counts))
        summary["post_normalize_total_counts_median"] = float(np.median(normalized_counts))
        summary["post_normalize_total_counts_mean"] = float(np.mean(normalized_counts))
        summary["post_normalize_total_counts_max"] = float(np.max(normalized_counts))

    if log1p:
        sc.pp.log1p(processed)
        values = processed.X.toarray() if sparse.issparse(processed.X) else np.asarray(processed.X)
        summary["post_log1p_min"] = float(np.min(values))
        summary["post_log1p_max"] = float(np.max(values))

    summary_path = output_dir / "preprocessing_summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    return PreprocessingResult(
        adata=processed,
        source_adata=source_adata,
        summary=summary,
        summary_path=summary_path,
    )


def _select_analysis_matrix(adata, preprocessing_config: dict[str, object], ad_module):
    matrix_source = str(preprocessing_config.get("matrix_source", "X")).strip().lower()
    layer_name = preprocessing_config.get("layer_name")

    if matrix_source == "x":
        source_adata = adata.copy()
        summary = {
            "matrix_source": "X",
            "matrix_source_detail": "Using adata.X as the downstream analysis matrix.",
            "matrix_source_layer_name": None,
            "selected_n_genes": int(source_adata.n_vars),
        }
        return source_adata, summary

    if matrix_source == "layer":
        if not layer_name:
            raise ValueError(
                "preprocessing.layer_name must be set when preprocessing.matrix_source='layer'."
            )
        if str(layer_name) not in adata.layers:
            raise ValueError(
                f"Requested preprocessing layer '{layer_name}' was not found in adata.layers."
            )
        source_adata = adata.copy()
        source_adata.X = _copy_matrix(adata.layers[str(layer_name)])
        summary = {
            "matrix_source": "layer",
            "matrix_source_detail": f"Using adata.layers['{layer_name}'] as the downstream analysis matrix.",
            "matrix_source_layer_name": str(layer_name),
            "selected_n_genes": int(source_adata.n_vars),
        }
        return source_adata, summary

    if matrix_source == "raw":
        if adata.raw is None:
            raise ValueError(
                "preprocessing.matrix_source='raw' was requested, but adata.raw is not available."
            )
        source_adata = ad_module.AnnData(
            X=_copy_matrix(adata.raw.X),
            obs=adata.obs.copy(),
            var=adata.raw.var.copy(),
        )
        for key in adata.obsm.keys():
            source_adata.obsm[key] = _copy_matrix(adata.obsm[key])
        for key in adata.uns.keys():
            source_adata.uns[key] = copy.deepcopy(adata.uns[key])
        summary = {
            "matrix_source": "raw",
            "matrix_source_detail": "Using adata.raw.X as the downstream analysis matrix.",
            "matrix_source_layer_name": None,
            "selected_n_genes": int(source_adata.n_vars),
        }
        return source_adata, summary

    raise ValueError(
        f"Unsupported preprocessing.matrix_source '{preprocessing_config.get('matrix_source')}'. "
        "Choose one of: X, raw, layer."
    )


def _copy_matrix(matrix):
    if hasattr(matrix, "copy"):
        return matrix.copy()
    return np.asarray(matrix).copy()


def _row_sums(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        values = np.asarray(matrix.sum(axis=1)).reshape(-1)
    else:
        values = np.asarray(matrix).sum(axis=1).reshape(-1)
    return values.astype(float)
