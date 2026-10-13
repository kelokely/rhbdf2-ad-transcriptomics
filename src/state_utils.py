"""State assignment helpers for RHBDF2 high versus low labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .gene_utils import extract_expression_vector


@dataclass
class StateAssignmentResult:
    """Summary of one expression-based state assignment."""

    column_name: str
    threshold_method: str
    threshold_value: float
    high_label: str
    low_label: str
    fallback_used: bool
    note: str


def assign_expression_state(adata, gene: str, state_config: dict[str, object]):
    """Assign high and low states for a gene and return a summary object."""
    if getattr(adata, "is_view", False):
        adata = adata.copy()

    method = str(state_config.get("method", "median"))
    quantile = float(state_config.get("quantile", 0.75))
    column_name = str(state_config.get("new_column_name", f"{gene}_state"))
    high_label = str(state_config.get("high_label", f"{gene}_high"))
    low_label = str(state_config.get("low_label", f"{gene}_low"))

    expression = extract_expression_vector(adata, gene)
    threshold = compute_threshold(expression, method, quantile)
    state_labels = np.where(expression > threshold, high_label, low_label)
    fallback_used = False
    note = "Assigned using the configured threshold."

    if method in {"median", "quantile"} and _has_empty_group(state_labels, high_label, low_label):
        state_labels = _rank_based_split(expression, method, quantile, high_label, low_label)
        fallback_used = True
        note = (
            "Threshold values alone produced an empty group because of tied expression values; "
            "a rank-based fallback was applied so exploratory comparisons remain possible."
        )

    adata.obs[column_name] = pd.Categorical(state_labels, categories=[high_label, low_label])
    result = StateAssignmentResult(
        column_name=column_name,
        threshold_method=method,
        threshold_value=float(threshold),
        high_label=high_label,
        low_label=low_label,
        fallback_used=fallback_used,
        note=note,
    )
    return adata, result


def build_state_summary(adata, result: StateAssignmentResult) -> pd.DataFrame:
    """Create a state-count summary table for export."""
    counts = (
        adata.obs[result.column_name]
        .value_counts(dropna=False)
        .rename_axis(result.column_name)
        .reset_index(name="count")
    )
    counts["threshold_method"] = result.threshold_method
    counts["threshold_value"] = result.threshold_value
    counts["fallback_used"] = result.fallback_used
    counts["note"] = result.note
    return counts


def compute_threshold(expression: np.ndarray, method: str, quantile: float) -> float:
    """Compute the configured threshold value."""
    if method == "median":
        return float(np.median(expression))
    if method == "nonzero":
        return 0.0
    if method == "quantile":
        return float(np.quantile(expression, quantile))
    raise ValueError(f"Unsupported state definition method: {method}")


def _has_empty_group(labels: np.ndarray, high_label: str, low_label: str) -> bool:
    return not np.any(labels == high_label) or not np.any(labels == low_label)


def _rank_based_split(
    expression: np.ndarray,
    method: str,
    quantile: float,
    high_label: str,
    low_label: str,
) -> np.ndarray:
    ordered_index = np.argsort(expression, kind="mergesort")
    labels = np.full(expression.shape[0], low_label, dtype=object)

    if method == "median":
        n_high = max(1, expression.shape[0] // 2)
    else:
        n_high = max(1, int(round(expression.shape[0] * (1.0 - quantile))))

    labels[ordered_index[-n_high:]] = high_label
    return labels
