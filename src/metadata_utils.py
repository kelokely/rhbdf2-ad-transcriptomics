"""Metadata validation and summary helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_metadata_columns(adata, metadata_config: dict[str, str | None]) -> pd.DataFrame:
    """Validate configured metadata columns without assuming column names in advance."""
    rows: list[dict[str, object]] = []
    for config_key, column_name in metadata_config.items():
        configured = column_name is not None
        exists = bool(configured and column_name in adata.obs.columns)
        if configured and not exists:
            raise ValueError(
                f"Configured metadata column '{column_name}' for '{config_key}' "
                "was not found in adata.obs."
            )
        rows.append(
            {
                "config_key": config_key,
                "column_name": column_name,
                "configured": configured,
                "exists": exists,
            }
        )
    return pd.DataFrame(rows)


def summarize_metadata(adata) -> pd.DataFrame:
    """Summarize all available obs metadata columns."""
    rows: list[dict[str, object]] = []
    for column in adata.obs.columns:
        series = adata.obs[column]
        rows.append(
            {
                "column_name": str(column),
                "dtype": str(series.dtype),
                "n_missing": int(series.isna().sum()),
                "n_unique": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values("column_name").reset_index(drop=True)


def export_value_counts(
    adata,
    columns: list[str | None],
    output_dir: Path,
) -> list[Path]:
    """Export value-count tables for configured metadata columns that exist."""
    written: list[Path] = []
    for column in columns:
        if not column or column not in adata.obs.columns:
            continue
        value_counts = (
            adata.obs[column]
            .value_counts(dropna=False)
            .rename_axis(column)
            .reset_index(name="count")
        )
        output_path = output_dir / f"value_counts_{column}.csv"
        value_counts.to_csv(output_path, index=False)
        written.append(output_path)
    return written


def suggest_metadata_columns(adata) -> pd.DataFrame:
    """Suggest likely diagnosis, donor, cluster, and sample columns from adata.obs names."""
    heuristics = {
        "diagnosis": ["diagnosis", "disease", "condition", "case_control", "pathology", "atscore"],
        "donor": ["donor", "patient", "subject", "individual"],
        "cluster": ["cluster", "subpopulation", "celltype", "cell_type", "annotation", "state"],
        "sample": ["sample", "library", "batch", "specimen"],
    }

    rows: list[dict[str, object]] = []
    for column in adata.obs.columns:
        lowered = str(column).lower()
        series = adata.obs[column]
        n_unique = int(series.nunique(dropna=True))
        for role, keywords in heuristics.items():
            keyword_hits = [keyword for keyword in keywords if keyword in lowered]
            if not keyword_hits:
                continue
            rows.append(
                {
                    "suggested_role": role,
                    "column_name": str(column),
                    "keyword_hits": ";".join(keyword_hits),
                    "n_unique": n_unique,
                    "dtype": str(series.dtype),
                    "suggestion_score": len(keyword_hits),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "suggested_role",
                "column_name",
                "keyword_hits",
                "n_unique",
                "dtype",
                "suggestion_score",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["suggested_role", "suggestion_score", "n_unique", "column_name"], ascending=[True, False, False, True])
        .reset_index(drop=True)
    )
