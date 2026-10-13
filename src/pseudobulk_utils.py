"""Donor-aware pseudobulk comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

try:
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests
except ImportError:
    sm = None
    multipletests = None


@dataclass
class PseudobulkResult:
    """Pseudobulk outputs and summaries."""

    sample_metadata: pd.DataFrame
    candidate_expression: pd.DataFrame
    paired_candidate_deltas: pd.DataFrame
    full_results: pd.DataFrame
    positive_results: pd.DataFrame
    negative_results: pd.DataFrame
    sample_metadata_path: Path
    candidate_expression_path: Path
    paired_candidate_deltas_path: Path
    full_results_path: Path
    positive_path: Path
    negative_path: Path
    n_pseudobulk_samples: int
    n_paired_donors: int


@dataclass
class SubpopulationPseudobulkResult:
    """Subpopulation-aware pseudobulk summaries."""

    run_summary: pd.DataFrame
    candidate_summary: pd.DataFrame
    gene_summary: pd.DataFrame
    run_summary_path: Path
    candidate_summary_path: Path
    gene_summary_path: Path


def run_pseudobulk_differential_expression(
    adata,
    pseudobulk_config: dict[str, object],
    output_dir: Path,
    candidate_genes: list[str],
    state_column: str,
    fallback_donor_column: str | None,
) -> PseudobulkResult:
    """Aggregate counts by donor/state and run donor-blocked OLS on log1p CPM."""
    if sm is None or multipletests is None:
        raise ImportError(
            "statsmodels is required for pseudobulk analysis. "
            "Install the full dependency set to enable this optional layer."
        )

    donor_column = pseudobulk_config.get("donor_column") or fallback_donor_column
    if not donor_column:
        raise ValueError(
            "Pseudobulk analysis requires a donor column. Set metadata.donor_column "
            "or pseudobulk.donor_column in the config."
        )

    group_column = str(pseudobulk_config.get("group_column") or state_column)
    group_a = str(pseudobulk_config.get("group_a") or "RHBDF2_high")
    group_b = str(pseudobulk_config.get("group_b") or "RHBDF2_low")
    paired_only = bool(pseudobulk_config.get("paired_only", True))
    min_cells_per_sample = int(pseudobulk_config.get("min_cells_per_sample", 20))
    min_donors_per_group = int(pseudobulk_config.get("min_donors_per_group", 2))
    normalization_target_sum = float(pseudobulk_config.get("normalization_target_sum", 1000000))
    top_n_export = int(pseudobulk_config.get("top_n_export", 100))

    _validate_required_columns(adata, donor_column, group_column)
    aggregated_counts, sample_metadata = _build_pseudobulk_samples(
        adata=adata,
        donor_column=str(donor_column),
        group_column=group_column,
        group_a=group_a,
        group_b=group_b,
        min_cells_per_sample=min_cells_per_sample,
    )

    if paired_only:
        valid_donors = (
            sample_metadata.groupby("donor_id", observed=True)[group_column]
            .nunique()
        )
        valid_donors = valid_donors[valid_donors >= 2].index.tolist()
        mask = sample_metadata["donor_id"].isin(valid_donors).to_numpy()
        sample_metadata = sample_metadata.loc[mask].reset_index(drop=True)
        aggregated_counts = aggregated_counts.loc[sample_metadata["sample_id"]]

    donors_per_group = (
        sample_metadata.groupby(group_column, observed=True)["donor_id"].nunique().to_dict()
    )
    if donors_per_group.get(group_a, 0) < min_donors_per_group or donors_per_group.get(group_b, 0) < min_donors_per_group:
        raise ValueError(
            "Pseudobulk comparison does not have enough donors per group after filtering. "
            f"Found {donors_per_group} with min_donors_per_group={min_donors_per_group}."
        )

    log_cpm = _normalize_log_cpm(aggregated_counts, normalization_target_sum)
    sample_metadata_path = output_dir / "pseudobulk_sample_metadata.csv"
    sample_metadata.to_csv(sample_metadata_path, index=False)

    present_candidates = [gene for gene in dict.fromkeys(candidate_genes) if gene in log_cpm.columns]
    candidate_expression = pd.concat(
        [sample_metadata.reset_index(drop=True), log_cpm[present_candidates].reset_index(drop=True)],
        axis=1,
    )
    candidate_expression_path = output_dir / "pseudobulk_candidate_expression.csv"
    candidate_expression.to_csv(candidate_expression_path, index=False)

    paired_candidate_deltas = _compute_paired_candidate_deltas(
        sample_metadata=sample_metadata,
        log_cpm=log_cpm,
        donor_column="donor_id",
        group_column=group_column,
        group_a=group_a,
        group_b=group_b,
        candidate_genes=present_candidates,
    )
    paired_candidate_deltas_path = output_dir / "pseudobulk_paired_candidate_deltas.csv"
    paired_candidate_deltas.to_csv(paired_candidate_deltas_path, index=False)

    full_results = _fit_donor_blocked_models(
        log_cpm=log_cpm,
        sample_metadata=sample_metadata,
        donor_column="donor_id",
        group_column=group_column,
        group_a=group_a,
        group_b=group_b,
    )
    full_results = full_results.sort_values(
        ["effect_size_log1p_cpm", "pvals_adj"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    full_results_path = output_dir / f"pseudobulk_de_{group_a.lower()}_vs_{group_b.lower()}.csv"
    positive_path = output_dir / "pseudobulk_top_positive_genes.csv"
    negative_path = output_dir / "pseudobulk_top_negative_genes.csv"
    full_results.to_csv(full_results_path, index=False)

    positive_results = (
        full_results[full_results["effect_size_log1p_cpm"] > 0]
        .sort_values(["effect_size_log1p_cpm", "pvals_adj"], ascending=[False, True], na_position="last")
        .head(top_n_export)
        .reset_index(drop=True)
    )
    negative_results = (
        full_results[full_results["effect_size_log1p_cpm"] < 0]
        .sort_values(["effect_size_log1p_cpm", "pvals_adj"], ascending=[True, True], na_position="last")
        .head(top_n_export)
        .reset_index(drop=True)
    )
    positive_results.to_csv(positive_path, index=False)
    negative_results.to_csv(negative_path, index=False)

    paired_donor_count = int(
        sample_metadata.groupby("donor_id", observed=True)[group_column].nunique().ge(2).sum()
    )

    return PseudobulkResult(
        sample_metadata=sample_metadata,
        candidate_expression=candidate_expression,
        paired_candidate_deltas=paired_candidate_deltas,
        full_results=full_results,
        positive_results=positive_results,
        negative_results=negative_results,
        sample_metadata_path=sample_metadata_path,
        candidate_expression_path=candidate_expression_path,
        paired_candidate_deltas_path=paired_candidate_deltas_path,
        full_results_path=full_results_path,
        positive_path=positive_path,
        negative_path=negative_path,
        n_pseudobulk_samples=int(sample_metadata.shape[0]),
        n_paired_donors=paired_donor_count,
    )


def run_within_subpopulation_pseudobulk(
    adata,
    pseudobulk_config: dict[str, object],
    output_dir: Path,
    candidate_genes: list[str],
    state_column: str,
    fallback_donor_column: str | None,
    subgroup_column: str | None,
) -> SubpopulationPseudobulkResult | None:
    """Run donor-aware pseudobulk separately within each microglia subpopulation."""
    if not pseudobulk_config.get("enabled", True) or not pseudobulk_config.get("within_subpopulation", True):
        return None
    if not subgroup_column or subgroup_column not in adata.obs.columns:
        return None

    subgroup_root = output_dir / "subpopulation_pseudobulk"
    subgroup_root.mkdir(parents=True, exist_ok=True)
    min_cells_per_subpopulation = int(pseudobulk_config.get("min_cells_per_subpopulation", 100))
    donor_column = str(pseudobulk_config.get("donor_column") or fallback_donor_column or "")
    subgroup_values = (
        pd.Series(adata.obs[subgroup_column])
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    run_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for subgroup_value in subgroup_values:
        mask = adata.obs[subgroup_column].astype(str) == subgroup_value
        n_cells = int(np.sum(mask))
        subgroup_pseudobulk_config = dict(pseudobulk_config)
        subgroup_min_cells_per_sample = pseudobulk_config.get("within_subpopulation_min_cells_per_sample")
        subgroup_min_donors_per_group = pseudobulk_config.get("within_subpopulation_min_donors_per_group")
        if subgroup_min_cells_per_sample is not None:
            subgroup_pseudobulk_config["min_cells_per_sample"] = int(subgroup_min_cells_per_sample)
        if subgroup_min_donors_per_group is not None:
            subgroup_pseudobulk_config["min_donors_per_group"] = int(subgroup_min_donors_per_group)
        run_row = {
            "subgroup_column": subgroup_column,
            "subgroup_value": subgroup_value,
            "n_cells": n_cells,
            "n_donors": int(adata.obs.loc[mask, donor_column].nunique()) if donor_column and donor_column in adata.obs.columns else np.nan,
            "status": "skipped",
            "message": "",
            "effective_min_cells_per_sample": int(subgroup_pseudobulk_config.get("min_cells_per_sample", 20)),
            "effective_min_donors_per_group": int(subgroup_pseudobulk_config.get("min_donors_per_group", 2)),
            "n_pseudobulk_samples": np.nan,
            "n_paired_donors": np.nan,
        }
        if n_cells < min_cells_per_subpopulation:
            run_row["message"] = (
                f"Skipped because only {n_cells} cells were available and "
                f"min_cells_per_subpopulation={min_cells_per_subpopulation}."
            )
            run_rows.append(run_row)
            continue

        subgroup_dir = subgroup_root / _sanitize_subgroup_value(subgroup_value)
        subgroup_dir.mkdir(parents=True, exist_ok=True)
        subset = adata[np.asarray(mask)].copy()
        try:
            result = run_pseudobulk_differential_expression(
                adata=subset,
                pseudobulk_config=subgroup_pseudobulk_config,
                output_dir=subgroup_dir,
                candidate_genes=candidate_genes,
                state_column=state_column,
                fallback_donor_column=fallback_donor_column,
            )
        except Exception as exc:
            run_row["status"] = "failed"
            run_row["message"] = str(exc)
            run_rows.append(run_row)
            continue

        run_row["status"] = "completed"
        run_row["n_pseudobulk_samples"] = result.n_pseudobulk_samples
        run_row["n_paired_donors"] = result.n_paired_donors
        run_rows.append(run_row)

        candidate_subset = result.full_results[result.full_results["names"].isin(candidate_genes)].copy()
        if candidate_subset.empty:
            continue
        candidate_subset["subgroup_column"] = subgroup_column
        candidate_subset["subgroup_value"] = subgroup_value
        candidate_subset["n_cells"] = n_cells
        candidate_subset["n_pseudobulk_samples"] = result.n_pseudobulk_samples
        candidate_subset["n_paired_donors"] = result.n_paired_donors
        candidate_rows.extend(candidate_subset.to_dict(orient="records"))

    run_summary = pd.DataFrame(run_rows)
    candidate_summary = pd.DataFrame(candidate_rows)
    gene_summary = _summarize_subpopulation_candidate_support(candidate_summary=candidate_summary)

    run_summary_path = subgroup_root / "subpopulation_pseudobulk_run_summary.csv"
    candidate_summary_path = subgroup_root / "subpopulation_pseudobulk_candidate_summary.csv"
    gene_summary_path = subgroup_root / "subpopulation_pseudobulk_gene_summary.csv"

    run_summary.to_csv(run_summary_path, index=False)
    candidate_summary.to_csv(candidate_summary_path, index=False)
    gene_summary.to_csv(gene_summary_path, index=False)

    return SubpopulationPseudobulkResult(
        run_summary=run_summary,
        candidate_summary=candidate_summary,
        gene_summary=gene_summary,
        run_summary_path=run_summary_path,
        candidate_summary_path=candidate_summary_path,
        gene_summary_path=gene_summary_path,
    )


def _validate_required_columns(adata, donor_column: str, group_column: str) -> None:
    for column in [donor_column, group_column]:
        if column not in adata.obs.columns:
            raise ValueError(f"Required pseudobulk column '{column}' not found in adata.obs.")


def _build_pseudobulk_samples(
    adata,
    donor_column: str,
    group_column: str,
    group_a: str,
    group_b: str,
    min_cells_per_sample: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = adata.obs[[donor_column, group_column]].copy()
    obs["_position"] = np.arange(adata.n_obs)
    obs = obs[obs[donor_column].notna() & obs[group_column].isin([group_a, group_b])]

    grouped_positions = (
        obs.groupby([donor_column, group_column], observed=True)["_position"]
        .apply(list)
        .to_dict()
    )

    sample_rows: list[dict[str, object]] = []
    count_rows: list[np.ndarray] = []
    sample_ids: list[str] = []

    for (donor_id, group_name), positions in grouped_positions.items():
        if len(positions) < min_cells_per_sample:
            continue
        summed_counts = _sum_rows(adata.X[positions])
        sample_id = f"{donor_id}__{group_name}"
        sample_ids.append(sample_id)
        count_rows.append(summed_counts)
        sample_rows.append(
            {
                "sample_id": sample_id,
                "donor_id": str(donor_id),
                group_column: str(group_name),
                "n_cells": int(len(positions)),
                "raw_library_size": float(np.sum(summed_counts)),
            }
        )

    if not sample_rows:
        raise ValueError("No pseudobulk samples were retained after donor/state filtering.")

    sample_metadata = pd.DataFrame(sample_rows)
    aggregated_counts = pd.DataFrame(
        np.vstack(count_rows),
        index=sample_ids,
        columns=adata.var_names.astype(str),
    )
    return aggregated_counts, sample_metadata


def _normalize_log_cpm(aggregated_counts: pd.DataFrame, normalization_target_sum: float) -> pd.DataFrame:
    library_sizes = aggregated_counts.sum(axis=1).replace(0, np.nan)
    normalized = aggregated_counts.div(library_sizes, axis=0).fillna(0.0) * normalization_target_sum
    return np.log1p(normalized)


def _fit_donor_blocked_models(
    log_cpm: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    donor_column: str,
    group_column: str,
    group_a: str,
    group_b: str,
) -> pd.DataFrame:
    design = pd.DataFrame(index=sample_metadata["sample_id"])
    design["intercept"] = 1.0
    design["state_indicator"] = (sample_metadata[group_column] == group_a).astype(float).to_numpy()
    donor_dummies = pd.get_dummies(sample_metadata[donor_column], prefix="donor", drop_first=True, dtype=float)
    design = pd.concat([design.reset_index(drop=True), donor_dummies.reset_index(drop=True)], axis=1)

    grouped_means = (
        log_cpm.assign(group=sample_metadata[group_column].to_numpy())
        .groupby("group", observed=True)
        .mean(numeric_only=True)
    )

    rows: list[dict[str, object]] = []
    for gene in log_cpm.columns:
        y = log_cpm[gene].to_numpy(dtype=float)
        if np.allclose(y, y[0]):
            effect_size = 0.0
            pvalue = 1.0
            t_statistic = 0.0
        else:
            model = sm.OLS(y, design.to_numpy(dtype=float)).fit()
            effect_size = float(model.params[1])
            pvalue = float(model.pvalues[1])
            t_statistic = float(model.tvalues[1])
            if not np.isfinite(pvalue):
                pvalue = 1.0
            if not np.isfinite(t_statistic):
                t_statistic = 0.0
        rows.append(
            {
                "names": str(gene),
                "effect_size_log1p_cpm": effect_size,
                "t_statistic": t_statistic,
                "pvals": pvalue,
                "group_a_mean_log1p_cpm": float(grouped_means.loc[group_a, gene]) if group_a in grouped_means.index else np.nan,
                "group_b_mean_log1p_cpm": float(grouped_means.loc[group_b, gene]) if group_b in grouped_means.index else np.nan,
                "n_pseudobulk_samples": int(sample_metadata.shape[0]),
            }
        )

    results = pd.DataFrame(rows)
    results["pvals_adj"] = multipletests(results["pvals"], method="fdr_bh")[1]
    return results


def _compute_paired_candidate_deltas(
    sample_metadata: pd.DataFrame,
    log_cpm: pd.DataFrame,
    donor_column: str,
    group_column: str,
    group_a: str,
    group_b: str,
    candidate_genes: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not candidate_genes:
        return pd.DataFrame(columns=["donor_id", "gene", "group_a_expression", "group_b_expression", "delta_high_minus_low"])

    for donor_id, donor_df in sample_metadata.groupby(donor_column, observed=True):
        groups_present = set(donor_df[group_column].tolist())
        if {group_a, group_b} - groups_present:
            continue

        group_a_sample = donor_df.loc[donor_df[group_column] == group_a, "sample_id"].iloc[0]
        group_b_sample = donor_df.loc[donor_df[group_column] == group_b, "sample_id"].iloc[0]
        for gene in candidate_genes:
            group_a_expression = float(log_cpm.loc[group_a_sample, gene])
            group_b_expression = float(log_cpm.loc[group_b_sample, gene])
            rows.append(
                {
                    "donor_id": donor_id,
                    "gene": gene,
                    "group_a_expression": group_a_expression,
                    "group_b_expression": group_b_expression,
                    "delta_high_minus_low": group_a_expression - group_b_expression,
                }
            )

    return pd.DataFrame(rows)


def _sum_rows(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        summed = np.asarray(matrix.sum(axis=0)).reshape(-1)
    else:
        summed = np.asarray(matrix).sum(axis=0).reshape(-1)
    return summed.astype(float)


def _summarize_subpopulation_candidate_support(candidate_summary: pd.DataFrame) -> pd.DataFrame:
    if candidate_summary.empty:
        return pd.DataFrame(
            columns=[
                "gene",
                "subpopulation_pseudobulk_n_subgroups_tested",
                "subpopulation_pseudobulk_n_subgroups_significant",
                "subpopulation_pseudobulk_n_subgroups_positive",
                "subpopulation_pseudobulk_mean_effect_log1p_cpm",
                "subpopulation_pseudobulk_median_effect_log1p_cpm",
                "subpopulation_pseudobulk_best_subgroup",
            ]
        )

    rows: list[dict[str, object]] = []
    for gene, gene_df in candidate_summary.groupby("names", observed=True):
        effects = gene_df["effect_size_log1p_cpm"].astype(float)
        padj = gene_df["pvals_adj"].astype(float)
        best_subgroup = np.nan
        if not effects.empty:
            best_index = effects.abs().idxmax()
            best_subgroup = gene_df.loc[best_index, "subgroup_value"]
        rows.append(
            {
                "gene": str(gene),
                "subpopulation_pseudobulk_n_subgroups_tested": int(gene_df["subgroup_value"].nunique()),
                "subpopulation_pseudobulk_n_subgroups_significant": int((padj < 0.05).sum()),
                "subpopulation_pseudobulk_n_subgroups_positive": int((effects > 0).sum()),
                "subpopulation_pseudobulk_mean_effect_log1p_cpm": float(effects.mean()),
                "subpopulation_pseudobulk_median_effect_log1p_cpm": float(effects.median()),
                "subpopulation_pseudobulk_best_subgroup": best_subgroup,
            }
        )
    return pd.DataFrame(rows).sort_values("gene").reset_index(drop=True)


def _sanitize_subgroup_value(value: str) -> str:
    sanitized = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value.strip())
    return sanitized.strip("_") or "subgroup"
