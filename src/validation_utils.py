"""Cross-dataset validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def build_cross_dataset_candidate_summary(
    run_dirs: list[str | Path],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, Path]:
    """Merge candidate evidence tables across multiple finished run directories."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged: pd.DataFrame | None = None
    dataset_rows: list[dict[str, Any]] = []

    for run_dir_value in run_dirs:
        run_dir = Path(run_dir_value)
        config = load_run_config(run_dir / "summaries" / "config_used.yaml")
        dataset_name = str(config["dataset_name"])
        evidence = pd.read_csv(run_dir / "tables" / "candidate_evidence_summary.csv")
        continuous = _load_optional_csv(run_dir / "continuous" / "continuous_association_with_rhbdf2.csv")
        state_summary = pd.read_csv(run_dir / "summaries" / "state_summary.csv")
        high_fraction = derive_high_fraction(state_summary)

        dataset_rows.append(
            {
                "dataset_name": dataset_name,
                "run_dir": str(run_dir),
                "rhbdf2_high_fraction": high_fraction,
                "has_continuous_model": continuous is not None,
                "has_subpopulation_pseudobulk": "subpopulation_pseudobulk_n_subgroups_significant" in evidence.columns,
            }
        )

        compact = _build_dataset_compact(
            dataset_name=dataset_name,
            evidence=evidence,
            continuous=continuous,
        )

        merged = compact if merged is None else merged.merge(compact, on="gene", how="outer")

    if merged is None:
        raise ValueError("No run directories were provided for cross-dataset validation.")

    dataset_names = [row["dataset_name"] for row in dataset_rows]
    merged["n_datasets"] = len(dataset_names)

    boolean_metric_suffixes = [
        "cell_level_significant_fdr_0_05",
        "pseudobulk_significant_fdr_0_05",
        "continuous_significant_fdr_0_05",
        "detection_significant_fdr_0_05",
        "positive_expression_significant_fdr_0_05",
        "subpopulation_supported",
        "protein_context_supported",
        "genetics_supported",
        "brain_context_supported",
        "public_proteomics_overlap_supported",
    ]
    for suffix in boolean_metric_suffixes:
        merged[f"n_{suffix.removesuffix('_fdr_0_05')}_datasets"] = _count_true_by_suffix(
            merged=merged,
            dataset_names=dataset_names,
            suffix=suffix,
        )

    direction_metric_map = {
        "cell_level_direction_consistent": "cell_level_logfoldchange",
        "pseudobulk_direction_consistent": "pseudobulk_effect_log1p_cpm",
        "continuous_direction_consistent": "continuous_slope_per_log1p_rhbdf2",
        "detection_direction_consistent": "detection_coef_per_log1p_rhbdf2",
        "positive_expression_direction_consistent": "positive_expression_coef_per_log1p_rhbdf2",
        "subpopulation_direction_consistent": "subpopulation_pseudobulk_mean_effect_log1p_cpm",
    }
    for output_column, suffix in direction_metric_map.items():
        merged[output_column] = merged.apply(
            lambda row: signs_are_consistent([row.get(f"{dataset}_{suffix}") for dataset in dataset_names]),
            axis=1,
        )

    score_metric_suffixes = [
        "weighted_transcript_evidence_score",
        "weighted_context_evidence_score",
        "weighted_total_evidence_score",
        "transcript_support_count",
        "context_support_count",
        "evidence_support_count",
    ]
    for suffix in score_metric_suffixes:
        columns = [f"{dataset}_{suffix}" for dataset in dataset_names if f"{dataset}_{suffix}" in merged.columns]
        if not columns:
            continue
        merged[f"mean_{suffix}"] = merged[columns].mean(axis=1, skipna=True)
    if "mean_weighted_total_evidence_score" not in merged.columns:
        merged["mean_weighted_total_evidence_score"] = float("nan")

    summary_path = output_dir / "cross_dataset_candidate_replication.csv"
    merged.sort_values(
        [
            "n_pseudobulk_significant_datasets",
            "n_continuous_significant_datasets",
            "mean_weighted_total_evidence_score",
            "n_detection_significant_datasets",
            "n_positive_expression_significant_datasets",
            "n_cell_level_significant_datasets",
            "gene",
        ],
        ascending=[False, False, False, False, False, False, True],
    ).to_csv(summary_path, index=False)

    dataset_summary_path = output_dir / "cross_dataset_run_summary.csv"
    pd.DataFrame(dataset_rows).to_csv(dataset_summary_path, index=False)
    return merged, summary_path


def load_run_config(config_path: str | Path) -> dict[str, Any]:
    """Load a config snapshot from a finished run directory."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def derive_high_fraction(state_summary: pd.DataFrame) -> float:
    """Derive the fraction of cells assigned to the high state."""
    if state_summary.empty:
        return float("nan")
    columns = list(state_summary.columns)
    state_column = next((column for column in columns if column != "count"), columns[0])
    total = float(state_summary["count"].sum())
    high_count = float(
        state_summary.loc[state_summary[state_column].astype(str).str.contains("high", case=False), "count"].sum()
    )
    return high_count / total if total else float("nan")


def signs_are_consistent(values: list[object]) -> bool:
    """Return True when the non-zero signs point in the same direction across datasets."""
    signs = []
    for value in values:
        if pd.isna(value):
            continue
        numeric = float(value)
        if numeric == 0:
            continue
        signs.append(numeric > 0)
    return len(set(signs)) <= 1


def _load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _build_dataset_compact(
    dataset_name: str,
    evidence: pd.DataFrame,
    continuous: pd.DataFrame | None,
) -> pd.DataFrame:
    compact = evidence.copy()
    base_columns = [
        "gene",
        "cell_level_logfoldchange",
        "cell_level_padj",
        "continuous_slope_per_log1p_rhbdf2",
        "continuous_padj",
        "pseudobulk_effect_log1p_cpm",
        "pseudobulk_padj",
        "cell_level_significant_fdr_0_05",
        "continuous_significant_fdr_0_05",
        "pseudobulk_significant_fdr_0_05",
        "detection_coef_per_log1p_rhbdf2",
        "detection_padj",
        "positive_expression_coef_per_log1p_rhbdf2",
        "positive_expression_padj",
        "subpopulation_pseudobulk_n_subgroups_significant",
        "subpopulation_pseudobulk_mean_effect_log1p_cpm",
        "protein_context_supported",
        "genetics_supported",
        "brain_context_supported",
        "public_proteomics_overlap_supported",
        "weighted_transcript_evidence_score",
        "weighted_context_evidence_score",
        "weighted_total_evidence_score",
        "transcript_support_count",
        "context_support_count",
        "evidence_support_count",
    ]
    base_columns = [column for column in base_columns if column in compact.columns]
    compact = compact[base_columns].copy()

    if "detection_padj" in compact.columns:
        compact["detection_significant_fdr_0_05"] = compact["detection_padj"].fillna(1.0) < 0.05
    if "positive_expression_padj" in compact.columns:
        compact["positive_expression_significant_fdr_0_05"] = compact["positive_expression_padj"].fillna(1.0) < 0.05
    if "subpopulation_pseudobulk_n_subgroups_significant" in compact.columns:
        compact["subpopulation_supported"] = (
            compact["subpopulation_pseudobulk_n_subgroups_significant"].fillna(0).astype(float) > 0
        )

    if continuous is not None and not continuous.empty:
        continuous_compact = continuous[
            ["names", "slope_per_log1p_rhbdf2", "pvals_adj"]
        ].rename(
            columns={
                "names": "gene",
                "slope_per_log1p_rhbdf2": "continuous_slope_per_log1p_rhbdf2",
                "pvals_adj": "continuous_padj",
            }
        )
        continuous_compact["continuous_significant_fdr_0_05"] = (
            continuous_compact["continuous_padj"].fillna(1.0) < 0.05
        )
        missing_columns = [
            column
            for column in [
                "continuous_slope_per_log1p_rhbdf2",
                "continuous_padj",
                "continuous_significant_fdr_0_05",
            ]
            if column not in compact.columns
        ]
        if missing_columns:
            compact = compact.merge(
                continuous_compact[["gene", *missing_columns]],
                on="gene",
                how="left",
            )

    rename_map = {
        column: f"{dataset_name}_{column}"
        for column in compact.columns
        if column != "gene"
    }
    return compact.rename(columns=rename_map)


def _count_true_by_suffix(
    merged: pd.DataFrame,
    dataset_names: list[str],
    suffix: str,
) -> pd.Series:
    columns = [f"{dataset}_{suffix}" for dataset in dataset_names if f"{dataset}_{suffix}" in merged.columns]
    if not columns:
        return pd.Series(0, index=merged.index, dtype=int)
    total = pd.Series(0, index=merged.index, dtype=int)
    for column in columns:
        series = merged[column].astype("boolean")
        total = total + series.fillna(False).astype(int)
    return total
