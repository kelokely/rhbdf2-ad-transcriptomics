#!/usr/bin/env python3
"""Run the committee-requested dual-anchor, vascular, myeloid, and metadata analyses."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import shutil
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yaml
from scipy import sparse
from scipy.stats import f as f_distribution
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from src.enrich_utils import export_ranked_gene_lists, run_enrichr_enrichment, run_gprofiler_enrichment
from src.multicell_ad_utils import (
    load_backed_population,
    load_backed_subset,
    run_cell_level_continuous_layer,
    run_continuous_donor_pseudobulk,
)


ROOT = Path(__file__).resolve().parent


def key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/committee_extensions.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed eligibility and primary manifests.")
    parser.add_argument("--skip-network-enrichment", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["_config_path"] = str(path.resolve())
    return config


def extract_backed_gene(obj, gene: str) -> np.ndarray:
    if gene not in obj.var_names:
        return np.full(obj.n_obs, np.nan)
    matrix = obj[:, obj.var_names.get_loc(gene)].X
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray()).reshape(-1).astype(float)
    return np.asarray(matrix).reshape(-1).astype(float)


def unique_value(series: pd.Series) -> object:
    values = [str(value) for value in series.dropna().unique() if str(value) not in {"", "nan", "None"}]
    return values[0] if len(values) == 1 else ";".join(sorted(values)) if values else pd.NA


def build_eligibility(config: dict, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = config["thresholds"]
    rows: list[dict[str, object]] = []
    donor_rows: list[dict[str, object]] = []
    for cohort, spec in config["cohorts"].items():
        path = ROOT / spec["path"]
        obj = ad.read_h5ad(path, backed="r")
        try:
            obs_columns = list(dict.fromkeys([
                spec["cell_type_column"], spec["donor_column"], spec["pathology_column"],
                *spec.get("metadata_variables", []),
            ]))
            obs_columns = [column for column in obs_columns if column in obj.obs.columns]
            obs = obj.obs[obs_columns].copy()
            expressions = {anchor: extract_backed_gene(obj, anchor) for anchor in config["anchors"]}
            for population, label in spec["populations"].items():
                mask = obs[spec["cell_type_column"]].astype(str).eq(str(label)).to_numpy()
                population_obs = obs.loc[mask].copy()
                for anchor in config["anchors"]:
                    expression = expressions[anchor][mask]
                    present = bool(np.isfinite(expression).any())
                    detected = np.isfinite(expression) & (expression > 0)
                    donor_detected = population_obs.loc[detected, spec["donor_column"]].astype(str).nunique() if present else 0
                    pathology_coverage = (
                        population_obs[[spec["donor_column"], spec["pathology_column"]]]
                        .drop_duplicates()
                        .groupby(spec["pathology_column"], observed=True)[spec["donor_column"]]
                        .nunique()
                        .to_dict()
                    )
                    eligible = (
                        present
                        and mask.sum() >= int(thresholds["minimum_cells"])
                        and detected.sum() >= int(thresholds["minimum_detected_cells"])
                        and donor_detected >= int(thresholds["minimum_donors_with_detection"])
                    )
                    rows.append({
                        "cohort": cohort,
                        "population": population,
                        "anchor_gene": anchor,
                        "matrix_scope": spec["matrix_scope"],
                        "n_genes": int(obj.n_vars),
                        "n_cells": int(mask.sum()),
                        "n_donors": int(population_obs[spec["donor_column"]].astype(str).nunique()),
                        "detected_cells": int(detected.sum()),
                        "detection_fraction": float(detected.mean()) if expression.size else np.nan,
                        "donors_with_detection": int(donor_detected),
                        "pathology_donor_coverage": json.dumps({str(k): int(v) for k, v in pathology_coverage.items()}, sort_keys=True),
                        "cell_continuous_eligible": eligible,
                        "donor_continuous_eligible": eligible and population_obs[spec["donor_column"]].nunique() >= int(thresholds["minimum_donors"]),
                        "status": "eligible" if eligible else "descriptive_only",
                    })
                    if not present:
                        continue
                    donor_frame = population_obs.copy()
                    donor_frame["_anchor_expression"] = expression
                    for donor, frame in donor_frame.groupby(spec["donor_column"], observed=True):
                        donor_expression = frame["_anchor_expression"].to_numpy(dtype=float)
                        n_cells = len(frame)
                        n_detected = int((donor_expression > 0).sum())
                        metadata = {
                            column: unique_value(frame[column])
                            for column in obs_columns
                            if column not in {spec["cell_type_column"], spec["donor_column"]}
                        }
                        donor_rows.append({
                            "cohort": cohort,
                            "population": population,
                            "anchor_gene": anchor,
                            "donor_id": str(donor),
                            "n_cells": n_cells,
                            "detected_cells": n_detected,
                            "detection_fraction": n_detected / n_cells if n_cells else np.nan,
                            "detection_log1p_per_1000": math.log1p(1000 * n_detected / n_cells) if n_cells else np.nan,
                            "summed_counts": float(np.nansum(donor_expression)),
                            **metadata,
                        })
        finally:
            obj.file.close()
    eligibility = pd.DataFrame(rows)
    donor_metrics = pd.DataFrame(donor_rows)
    eligibility.to_csv(output_dir / "dual_anchor_population_eligibility.csv", index=False)
    donor_metrics.to_csv(output_dir / "dual_anchor_donor_metrics.csv", index=False)
    return eligibility, donor_metrics


def top_direction_tables(full: pd.DataFrame, anchor: str, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    effect = f"slope_per_log1p_{key(anchor)}"
    without_self = full[full["names"].astype(str).ne(anchor)].copy()
    positive = without_self[without_self[effect] > 0].sort_values([effect, "pvals_adj"], ascending=[False, True]).head(top_n)
    negative = without_self[without_self[effect] < 0].sort_values([effect, "pvals_adj"], ascending=[True, True]).head(top_n)
    return positive, negative


def save_analysis_matrix(result, candidate_genes: list[str], path: Path) -> None:
    metadata = result.sample_metadata.reset_index(drop=True).copy()
    present = [gene for gene in candidate_genes if gene in result.log_cpm.columns]
    expression = result.log_cpm[present].reset_index(drop=True).add_prefix("expr__")
    pd.concat([metadata, expression], axis=1).to_csv(path, index=False)


def run_enrichment_layer(
    full: pd.DataFrame,
    var_names: list[str],
    anchor: str,
    config: dict,
    output_dir: Path,
) -> list[dict[str, object]]:
    statuses: list[dict[str, object]] = []
    top_n = int(config["enrichment"]["top_genes_per_direction"])
    positive, negative = top_direction_tables(full, anchor, top_n)
    gprofiler_config = {
        "enabled": True,
        "organism": "hsapiens",
        "sources": config["enrichment"]["gprofiler_sources"],
        "user_threshold": 0.05,
        "significance_threshold_method": "g_SCS",
        "domain_scope": "custom",
        "use_dataset_background": True,
        "max_terms_report": int(config["enrichment"]["max_terms_report"]),
    }
    enrichr_config = {
        "enabled": True,
        "organism": "human",
        "libraries": config["enrichment"]["enrichr_libraries"],
        "cutoff": 0.05,
        "use_dataset_background": True,
        "max_terms_report": int(config["enrichment"]["max_terms_report"]),
    }
    for backend, function, backend_config in [
        ("gprofiler", run_gprofiler_enrichment, gprofiler_config),
        ("enrichr", run_enrichr_enrichment, enrichr_config),
    ]:
        try:
            function(positive, negative, backend_config, output_dir, background_genes=var_names, prefix=f"{key(anchor)}_")
            statuses.append({"anchor_gene": anchor, "backend": backend, "status": "completed", "reason": ""})
        except Exception as exc:
            statuses.append({"anchor_gene": anchor, "backend": backend, "status": "failed", "reason": str(exc)})
    return statuses


def run_one_population(
    raw: ad.AnnData,
    cohort: str,
    population: str,
    spec: dict,
    config: dict,
    output_dir: Path,
    enable_enrichment: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for anchor in config["anchors"]:
        anchor_dir = output_dir / key(cohort) / key(population) / key(anchor)
        cell_dir = anchor_dir / "cell_continuous"
        donor_dir = anchor_dir / "donor_pseudobulk"
        enrichment_dir = anchor_dir / "enrichment"
        cell_result = run_cell_level_continuous_layer(
            raw_adata=raw,
            output_dir=cell_dir,
            candidate_genes=config["candidate_genes"],
            donor_column=spec["donor_column"],
            anchor_gene=anchor,
        )
        donor_result = run_continuous_donor_pseudobulk(
            adata=raw,
            output_dir=donor_dir,
            candidate_genes=config["candidate_genes"],
            donor_column=spec["donor_column"],
            pathology_column=spec["pathology_column"],
            minimum_donors=int(config["thresholds"]["minimum_donors"]),
            anchor_gene=anchor,
        )
        effect = f"slope_per_log1p_{key(anchor)}"
        export_ranked_gene_lists(cell_result.full_results, cell_dir, effect, "pvals_adj", prefix=f"{key(anchor)}_")
        matrix_path = donor_dir / f"donor_analysis_matrix_{key(anchor)}.csv"
        if donor_result.status == "completed":
            save_analysis_matrix(donor_result, config["candidate_genes"], matrix_path)
        enrichment_status: list[dict[str, object]] = []
        if enable_enrichment and spec["matrix_scope"] == "full_transcriptome":
            enrichment_dir.mkdir(parents=True, exist_ok=True)
            enrichment_status = run_enrichment_layer(
                cell_result.full_results,
                raw.var_names.astype(str).tolist(),
                anchor,
                config,
                enrichment_dir,
            )
            pd.DataFrame(enrichment_status).to_csv(enrichment_dir / "enrichment_status.csv", index=False)
        records.append({
            "cohort": cohort,
            "population": population,
            "anchor_gene": anchor,
            "matrix_scope": spec["matrix_scope"],
            "n_cells": int(raw.n_obs),
            "n_genes": int(raw.n_vars),
            "cell_results": str(cell_result.full_results_path),
            "hurdle_results": str(cell_result.candidate_hurdle_path),
            "donor_results": str(donor_dir / ("donor_continuous_pseudobulk_all_genes.csv" if anchor == "RHBDF2" else f"donor_continuous_pseudobulk_all_genes_{key(anchor)}.csv")),
            "donor_candidate_results": str(donor_dir / ("donor_continuous_pseudobulk_candidates.csv" if anchor == "RHBDF2" else f"donor_continuous_pseudobulk_candidates_{key(anchor)}.csv")),
            "donor_analysis_matrix": str(matrix_path),
            "donor_status": donor_result.status,
            "donor_reason": donor_result.reason,
            "enrichment_status": json.dumps(enrichment_status),
        })
    return records


def run_primary_analyses(config: dict, output_dir: Path, skip_network: bool) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for cohort, population in config["primary_runs"]:
        spec = config["cohorts"][cohort]
        raw = load_backed_population(ROOT / spec["path"], spec["cell_type_column"], spec["populations"][population])
        print(f"Primary: {cohort} / {population}: {raw.n_obs:,} cells x {raw.n_vars:,} genes", flush=True)
        enable_enrichment = (
            not skip_network
            and bool(config["enrichment"]["enabled"])
            and cohort in config["enrichment"]["cohorts"]
            and population == config["enrichment"]["population"]
        )
        records.extend(run_one_population(raw, cohort, population, spec, config, output_dir / "analyses", enable_enrichment))
        del raw
        gc.collect()
    manifest = pd.DataFrame(records)
    manifest.to_csv(output_dir / "analysis_manifest.csv", index=False)
    return manifest


def run_strict_seaad_myeloid(config: dict, output_dir: Path, skip_network: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    myeloid = config["strict_seaad_myeloid"]
    cohort = myeloid["cohort"]
    spec = config["cohorts"][cohort]
    obj = ad.read_h5ad(ROOT / spec["path"], backed="r")
    source = obj.obs[myeloid["source_subpopulation_column"]].astype(str)
    include = np.zeros(obj.n_obs, dtype=bool)
    for prefix in myeloid["include_prefixes"]:
        include |= source.str.startswith(prefix).to_numpy()
    include |= source.isin(myeloid["include_exact"]).to_numpy()
    include &= ~source.isin(myeloid["exclude_exact"]).to_numpy()
    obj.file.close()
    raw = load_backed_subset(ROOT / spec["path"], include)
    print(f"Strict SEA-AD myeloid: {raw.n_obs:,} cells", flush=True)
    composition = (
        raw.obs.groupby([spec["donor_column"], myeloid["source_subpopulation_column"]], observed=True)
        .size().rename("n_cells").reset_index()
    )
    totals = composition.groupby(spec["donor_column"], observed=True)["n_cells"].transform("sum")
    composition["fraction"] = composition["n_cells"] / totals
    composition.to_csv(output_dir / "strict_myeloid_composition_by_donor.csv", index=False)
    records = run_one_population(
        raw, cohort, "Strict pooled myeloid", spec, config, output_dir / "analyses", enable_enrichment=False
    )
    labels = raw.obs[myeloid["source_subpopulation_column"]].astype(str).value_counts()
    sensitivity_rows: list[pd.DataFrame] = []
    for label, n_cells in labels.items():
        if int(n_cells) < int(myeloid["leave_one_subpopulation_out_minimum_cells"]):
            continue
        subset = raw[~raw.obs[myeloid["source_subpopulation_column"]].astype(str).eq(str(label))].copy()
        for anchor in config["anchors"]:
            sensitivity_dir = output_dir / "myeloid_sensitivity" / f"exclude_{key(label)}" / key(anchor)
            result = run_continuous_donor_pseudobulk(
                adata=subset,
                output_dir=sensitivity_dir,
                candidate_genes=config["candidate_genes"],
                donor_column=spec["donor_column"],
                pathology_column=spec["pathology_column"],
                minimum_donors=int(config["thresholds"]["minimum_donors"]),
                anchor_gene=anchor,
            )
            if result.status == "completed":
                frame = result.candidate_results.copy()
                frame.insert(0, "excluded_subpopulation", str(label))
                frame.insert(1, "cells_retained", int(subset.n_obs))
                frame["anchor_gene"] = anchor
                sensitivity_rows.append(frame)
        del subset
        gc.collect()
    sensitivity = pd.concat(sensitivity_rows, ignore_index=True) if sensitivity_rows else pd.DataFrame()
    sensitivity.to_csv(output_dir / "strict_myeloid_leave_one_subpopulation_out_candidates.csv", index=False)
    del raw
    gc.collect()
    return pd.DataFrame(records), sensitivity


def evaluate_inferred_myeloid(config: dict, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort in ["GSE157827", "GSE237718"]:
        spec = config["cohorts"][cohort]
        obj = ad.read_h5ad(ROOT / spec["path"], backed="r")
        obs = obj.obs
        unassigned = obs[spec["cell_type_column"]].astype(str).eq("Unassigned")
        score_columns = [column for column in obs if column.startswith("marker_score_") and column != "marker_score_microglia"]
        stronger = obs["marker_score_microglia"].to_numpy(dtype=float) > obs[score_columns].max(axis=1).to_numpy(dtype=float)
        multiple_markers = obs["cell_type_detected_markers"].to_numpy(dtype=float) >= 2
        eligible = unassigned.to_numpy() & stronger & multiple_markers
        rows.append({
            "cohort": cohort,
            "unassigned_cells": int(unassigned.sum()),
            "cells_with_microglia_score_stronger_than_other_lineages": int((unassigned.to_numpy() & stronger).sum()),
            "cells_with_at_least_two_detected_identity_markers": int((unassigned.to_numpy() & multiple_markers).sum()),
            "cells_meeting_strict_inferred_myeloid_rule": int(eligible.sum()),
            "status": "eligible" if eligible.sum() >= int(config["thresholds"]["minimum_cells"]) else "skipped",
            "reason": "requires strongest lineage score and at least two detected identity markers",
        })
        obj.file.close()
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "inferred_myeloid_sensitivity_eligibility.csv", index=False)
    return result


def metadata_source(cohort: str) -> pd.DataFrame:
    base = ROOT / "results/committee_extensions/donor_metadata_audit"
    mapping = {
        "GSE243292": base / "GSE243292_embedded_donor_metadata.csv",
        "GSE157827": base / "GSE157827_embedded_donor_metadata.csv",
        "GSE237718": base / "GSE237718_embedded_donor_metadata.csv",
        "SEAAD_MTG": base / "SEAAD_analysis_ready_donor_metadata.csv",
    }
    return pd.read_csv(mapping[cohort])


def resolve_metadata_column(frame: pd.DataFrame, requested: str) -> str | None:
    candidates = [requested, f"{requested}_public", f"{requested}_embedded", f"{requested}_meta"]
    return next((column for column in candidates if column in frame.columns), None)


def numeric_metadata(variable: str, series: pd.Series) -> bool:
    if variable in {"age_at_death", "pmi", "rin", "brain_ph", "years_of_education"}:
        return True
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna().mean() >= 0.9 and numeric.nunique() > 6


def nested_f_test(y: np.ndarray, reduced: np.ndarray, full: np.ndarray) -> dict[str, float]:
    reduced_fit = np.linalg.lstsq(reduced, y, rcond=None)[0]
    full_fit = np.linalg.lstsq(full, y, rcond=None)[0]
    reduced_sse = float(np.square(y - reduced @ reduced_fit).sum())
    full_residual = y - full @ full_fit
    full_sse = float(np.square(full_residual).sum())
    numerator_df = int(np.linalg.matrix_rank(full) - np.linalg.matrix_rank(reduced))
    denominator_df = int(len(y) - np.linalg.matrix_rank(full))
    if numerator_df <= 0 or denominator_df <= 0 or full_sse <= 0:
        return {"f_statistic": np.nan, "pval": np.nan, "numerator_df": numerator_df, "residual_df": denominator_df, "r_squared": np.nan}
    statistic = max(0.0, ((reduced_sse - full_sse) / numerator_df) / (full_sse / denominator_df))
    return {
        "f_statistic": statistic,
        "pval": float(f_distribution.sf(statistic, numerator_df, denominator_df)),
        "numerator_df": numerator_df,
        "residual_df": denominator_df,
        "r_squared": 1.0 - full_sse / float(np.square(y - y.mean()).sum()) if np.var(y) > 0 else np.nan,
    }


def prepare_metadata_design(series: pd.Series, variable: str, minimum_level: int) -> tuple[np.ndarray, pd.DataFrame, str, dict[str, int]] | None:
    if numeric_metadata(variable, series):
        values = pd.to_numeric(series, errors="coerce")
        mask = values.notna().to_numpy()
        values = values.loc[mask].to_numpy(dtype=float)
        if values.size < 10 or np.isclose(values.std(), 0):
            return None
        values = (values - values.mean()) / values.std(ddof=1)
        return mask, pd.DataFrame({variable: values}), "continuous_zscore", {}
    clean = series.astype("string")
    counts = clean.value_counts(dropna=True)
    retained = counts[counts >= minimum_level].index
    mask = clean.isin(retained).to_numpy()
    selected = clean.loc[mask]
    if selected.nunique() < 2 or mask.sum() < 10:
        return None
    dummies = pd.get_dummies(selected, prefix=key(variable), drop_first=True, dtype=float).reset_index(drop=True)
    return mask, dummies, "categorical_omnibus", {str(k): int(v) for k, v in selected.value_counts().items()}


def run_metadata_models(config: dict, manifest: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coverage_rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []
    interaction_rows: list[dict[str, object]] = []
    slope_rows: list[dict[str, object]] = []
    minimum_level = int(config["thresholds"]["minimum_donors_per_metadata_level"])
    minimum_df = int(config["thresholds"]["minimum_residual_degrees_freedom"])
    for record in manifest.itertuples(index=False):
        matrix_path = Path(record.donor_analysis_matrix)
        if record.donor_status != "completed" or not matrix_path.exists():
            continue
        frame = pd.read_csv(matrix_path)
        extra = metadata_source(record.cohort)
        extra = extra.drop(columns=[column for column in ["cohort", "n_observations"] if column in extra], errors="ignore")
        frame["donor_id"] = frame["donor_id"].astype(str)
        extra["donor_id"] = extra["donor_id"].astype(str)
        frame = frame.merge(extra, on="donor_id", how="left", suffixes=("", "_meta"), validate="one_to_one")
        anchor_key = key(record.anchor_gene)
        anchor_column = f"{anchor_key}_detection_rate_log1p_per_1000"
        if anchor_column not in frame:
            continue
        expression_columns = [
            column for column in frame
            if column.startswith("expr__") and column != f"expr__{record.anchor_gene}"
        ]
        for program, genes in config["programs"].items():
            present = [f"expr__{gene}" for gene in genes if f"expr__{gene}" in frame and gene != record.anchor_gene]
            if len(present) >= 2:
                values = frame[present].to_numpy(dtype=float)
                std = values.std(axis=0, ddof=1)
                std[std == 0] = 1
                frame[f"module__{program}"] = ((values - values.mean(axis=0)) / std).mean(axis=1)
        target_columns = expression_columns + [column for column in frame if column.startswith("module__")]
        seen_metadata: dict[tuple[str, ...], str] = {}
        for requested in config["cohorts"][record.cohort].get("metadata_variables", []):
            metadata_column = resolve_metadata_column(frame, requested)
            if metadata_column is None:
                coverage_rows.append({"cohort": record.cohort, "population": record.population, "anchor_gene": record.anchor_gene, "metadata_variable": requested, "status": "missing", "n_complete": 0, "level_counts": "{}"})
                continue
            signature = tuple(frame[metadata_column].astype("string").fillna("<NA>").tolist())
            if signature in seen_metadata:
                coverage_rows.append({
                    "cohort": record.cohort,
                    "population": record.population,
                    "anchor_gene": record.anchor_gene,
                    "metadata_variable": requested,
                    "status": "duplicate_representation",
                    "n_complete": int(frame[metadata_column].notna().sum()),
                    "level_counts": "{}",
                    "duplicate_of": seen_metadata[signature],
                })
                continue
            seen_metadata[signature] = requested
            design_info = prepare_metadata_design(frame[metadata_column], requested, minimum_level)
            if design_info is None:
                counts = frame[metadata_column].astype("string").value_counts(dropna=True).to_dict()
                coverage_rows.append({"cohort": record.cohort, "population": record.population, "anchor_gene": record.anchor_gene, "metadata_variable": requested, "status": "not_eligible", "n_complete": int(frame[metadata_column].notna().sum()), "level_counts": json.dumps({str(k): int(v) for k, v in counts.items()})})
                continue
            mask, metadata_design, model_type, level_counts = design_info
            work = frame.loc[mask].reset_index(drop=True)
            x = work[anchor_column].to_numpy(dtype=float)
            z = metadata_design.to_numpy(dtype=float)
            coverage_rows.append({"cohort": record.cohort, "population": record.population, "anchor_gene": record.anchor_gene, "metadata_variable": requested, "status": "eligible", "n_complete": int(mask.sum()), "model_type": model_type, "level_counts": json.dumps(level_counts, sort_keys=True)})
            anchor_test = nested_f_test(x, np.ones((len(x), 1)), np.column_stack([np.ones(len(x)), z]))
            anchor_rows.append({"cohort": record.cohort, "population": record.population, "anchor_gene": record.anchor_gene, "metadata_variable": requested, "model_type": model_type, "n_donors": len(x), **anchor_test})
            for target_column in target_columns:
                y = work[target_column].to_numpy(dtype=float)
                if np.isclose(np.std(y), 0) or np.isclose(np.std(x), 0):
                    continue
                reduced = np.column_stack([np.ones(len(x)), x, z])
                interactions = z * x.reshape(-1, 1)
                full = np.column_stack([reduced, interactions])
                test = nested_f_test(y, reduced, full)
                if test["residual_df"] < minimum_df:
                    continue
                target_type, target = target_column.split("__", 1)
                interaction_rows.append({"cohort": record.cohort, "population": record.population, "anchor_gene": record.anchor_gene, "metadata_variable": requested, "model_type": model_type, "target_type": target_type, "target": target, "n_donors": len(x), **test})
                if model_type == "categorical_omnibus":
                    selected_levels = work[metadata_column].astype(str)
                    for level, level_frame in work.assign(_level=selected_levels).groupby("_level", observed=True):
                        indices = level_frame.index.to_numpy()
                        if len(indices) < minimum_level or np.unique(x[indices]).size < 3:
                            continue
                        fit = sm.OLS(y[indices], sm.add_constant(x[indices])).fit()
                        slope_rows.append({"cohort": record.cohort, "population": record.population, "anchor_gene": record.anchor_gene, "metadata_variable": requested, "metadata_level": str(level), "target_type": target_type, "target": target, "n_donors": len(indices), "anchor_slope": float(fit.params[1]), "slope_se": float(fit.bse[1]), "slope_pval": float(fit.pvalues[1])})
    coverage = pd.DataFrame(coverage_rows)
    anchor_associations = pd.DataFrame(anchor_rows)
    interactions = pd.DataFrame(interaction_rows)
    slopes = pd.DataFrame(slope_rows)
    if not anchor_associations.empty:
        anchor_associations["pvals_adj"] = multipletests(anchor_associations["pval"].fillna(1), method="fdr_bh")[1]
        anchor_associations = anchor_associations.sort_values(["pvals_adj", "pval"])
    if not interactions.empty:
        interactions["pvals_adj"] = np.nan
        for _, indices in interactions.groupby(["cohort", "population", "anchor_gene", "metadata_variable", "target_type"], observed=True).groups.items():
            interactions.loc[indices, "pvals_adj"] = multipletests(interactions.loc[indices, "pval"].fillna(1), method="fdr_bh")[1]
        interactions = interactions.sort_values(["pvals_adj", "pval"])
    if not slopes.empty:
        slopes["slope_pvals_adj"] = multipletests(slopes["slope_pval"].fillna(1), method="fdr_bh")[1]
    coverage.to_csv(output_dir / "metadata_model_coverage.csv", index=False)
    anchor_associations.to_csv(output_dir / "anchor_metadata_associations.csv", index=False)
    interactions.to_csv(output_dir / "anchor_metadata_interactions.csv", index=False)
    slopes.to_csv(output_dir / "metadata_stratified_anchor_slopes.csv", index=False)
    return coverage, anchor_associations, interactions, slopes


def synthesize(config: dict, manifest: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    concordance_rows: list[dict[str, object]] = []
    gene_rows: list[pd.DataFrame] = []
    donor_rows: list[dict[str, object]] = []
    for (cohort, population), records in manifest.groupby(["cohort", "population"], observed=True):
        by_anchor = {row.anchor_gene: row for row in records.itertuples(index=False)}
        if not all(anchor in by_anchor for anchor in config["anchors"]):
            continue
        left = pd.read_csv(by_anchor["RHBDF2"].cell_results)
        right = pd.read_csv(by_anchor["ADAM17"].cell_results)
        merged = left[["names", "slope_per_log1p_rhbdf2", "pvals_adj"]].merge(
            right[["names", "slope_per_log1p_adam17", "pvals_adj"]], on="names", suffixes=("_rhbdf2", "_adam17")
        )
        merged = merged[~merged["names"].isin(config["anchors"])].copy()
        rho, pvalue = spearmanr(merged["slope_per_log1p_rhbdf2"], merged["slope_per_log1p_adam17"])
        r_support = merged["pvals_adj_rhbdf2"] < 0.05
        a_support = merged["pvals_adj_adam17"] < 0.05
        same = np.sign(merged["slope_per_log1p_rhbdf2"]) == np.sign(merged["slope_per_log1p_adam17"])
        merged["classification"] = np.select(
            [r_support & a_support & same, r_support & a_support & ~same, r_support & ~a_support, ~r_support & a_support],
            ["shared_same_direction", "shared_opposite_direction", "rhbdf2_preferential", "adam17_preferential"],
            default="unsupported_at_fdr",
        )
        merged.insert(0, "population", population)
        merged.insert(0, "cohort", cohort)
        gene_rows.append(merged)
        top_set_size = min(200, max(10, int(math.ceil(0.05 * len(merged)))))
        top_r = set(merged.nlargest(top_set_size, "slope_per_log1p_rhbdf2")["names"])
        top_a = set(merged.nlargest(top_set_size, "slope_per_log1p_adam17")["names"])
        concordance_rows.append({
            "cohort": cohort,
            "population": population,
            "n_genes": len(merged),
            "effect_spearman_rho": float(rho),
            "effect_spearman_pval": float(pvalue),
            "top_set_size": top_set_size,
            "top_positive_jaccard": len(top_r & top_a) / len(top_r | top_a) if top_r | top_a else np.nan,
            "shared_same_direction_fdr": int((r_support & a_support & same).sum()),
            "shared_opposite_direction_fdr": int((r_support & a_support & ~same).sum()),
            "rhbdf2_preferential_fdr": int((r_support & ~a_support).sum()),
            "adam17_preferential_fdr": int((~r_support & a_support).sum()),
        })
        r_matrix = pd.read_csv(by_anchor["RHBDF2"].donor_analysis_matrix)
        a_matrix = pd.read_csv(by_anchor["ADAM17"].donor_analysis_matrix)
        donor = r_matrix[["donor_id", "rhbdf2_detection_rate_log1p_per_1000"]].merge(
            a_matrix[["donor_id", "adam17_detection_rate_log1p_per_1000"]], on="donor_id"
        )
        rho, pvalue = spearmanr(donor["rhbdf2_detection_rate_log1p_per_1000"], donor["adam17_detection_rate_log1p_per_1000"])
        donor_rows.append({"cohort": cohort, "population": population, "n_donors": len(donor), "donor_anchor_spearman_rho": float(rho), "donor_anchor_spearman_pval": float(pvalue)})
    concordance = pd.DataFrame(concordance_rows)
    genes = pd.concat(gene_rows, ignore_index=True) if gene_rows else pd.DataFrame()
    donor = pd.DataFrame(donor_rows)
    concordance.to_csv(output_dir / "dual_anchor_concordance_summary.csv", index=False)
    genes.to_csv(output_dir / "dual_anchor_gene_classification.csv", index=False)
    donor.to_csv(output_dir / "donor_anchor_correlation.csv", index=False)
    return concordance, genes, donor


def cross_cohort_candidates(config: dict, manifest: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for record in manifest.itertuples(index=False):
        path = Path(record.donor_candidate_results)
        if record.donor_status != "completed" or not path.exists():
            continue
        frame = pd.read_csv(path)
        effect = f"effect_per_log1p_{key(record.anchor_gene)}_detection_per_1000_cells"
        frame = frame[["names", effect, "pvals", "pvals_adj", "self_association"]].copy()
        frame = frame.rename(columns={effect: "effect"})
        frame.insert(0, "anchor_gene", record.anchor_gene)
        frame.insert(0, "population", record.population)
        frame.insert(0, "cohort", record.cohort)
        rows.append(frame)
    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if long.empty:
        long.to_csv(output_dir / "cross_cohort_candidate_replication.csv", index=False)
        return long
    summary_rows: list[dict[str, object]] = []
    for (population, anchor, gene), frame in long.groupby(["population", "anchor_gene", "names"], observed=True):
        nonself = frame[~frame["self_association"].astype(bool)]
        if nonself.empty:
            continue
        positive = int((nonself.effect > 0).sum())
        negative = int((nonself.effect < 0).sum())
        summary_rows.append({
            "population": population,
            "anchor_gene": anchor,
            "gene": gene,
            "cohorts_tested": int(nonself.cohort.nunique()),
            "cohorts_fdr_supported": int((nonself.pvals_adj < 0.05).sum()),
            "positive_cohorts": positive,
            "negative_cohorts": negative,
            "direction_consistent": positive == 0 or negative == 0,
            "cohort_effects": json.dumps({row.cohort: float(row.effect) for row in nonself.itertuples()}, sort_keys=True),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["cohorts_fdr_supported", "cohorts_tested", "population", "anchor_gene", "gene"], ascending=[False, False, True, True, True])
    long.to_csv(output_dir / "cross_cohort_candidate_results_long.csv", index=False)
    summary.to_csv(output_dir / "cross_cohort_candidate_replication.csv", index=False)
    return summary


def adam17_output_gene_table(config: dict) -> pd.DataFrame:
    """Return the prespecified ADAM17-output panel with explicit evidence roles."""
    roles: dict[str, list[str]] = {}
    evidence: dict[str, list[str]] = {}
    order: list[str] = []
    for family, spec in config.get("adam17_output_families", {}).items():
        for gene in spec.get("genes", []):
            if gene not in roles:
                order.append(gene)
            roles.setdefault(gene, []).append(family)
            evidence.setdefault(gene, []).append(str(spec.get("evidence_class", "unspecified")))
    return pd.DataFrame([
        {
            "gene": gene,
            "primary_family": roles[gene][0],
            "families": ";".join(roles[gene]),
            "evidence_class": ";".join(dict.fromkeys(evidence[gene])),
            "panel_order": index,
        }
        for index, gene in enumerate(order)
    ])


def summarize_adam17_outputs(
    config: dict,
    manifest: pd.DataFrame,
    sensitivity: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize transcript associations for ADAM17 substrates and output readouts.

    The summary deliberately preserves the distinction between substrate-gene RNA
    abundance and biochemical ectodomain shedding.
    """
    genes = adam17_output_gene_table(config)
    if genes.empty:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    panel = set(genes["gene"])
    coverage_rows: list[dict[str, object]] = []
    for record in manifest.itertuples(index=False):
        result_path = Path(record.donor_candidate_results)
        if not result_path.is_absolute():
            result_path = ROOT / result_path
        available: set[str] = set()
        if record.donor_status == "completed" and result_path.exists():
            available = set(pd.read_csv(result_path, usecols=["names"])["names"].astype(str))
        for gene in genes.itertuples(index=False):
            coverage_rows.append({
                "cohort": record.cohort,
                "population": record.population,
                "anchor_gene": record.anchor_gene,
                "gene": gene.gene,
                "primary_family": gene.primary_family,
                "matrix_scope": record.matrix_scope,
                "donor_status": record.donor_status,
                "availability": "present" if gene.gene in available else "absent_from_working_matrix",
            })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output_dir / "adam17_output_matrix_coverage.csv", index=False)

    long_path = output_dir / "cross_cohort_candidate_results_long.csv"
    all_results = pd.read_csv(long_path) if long_path.exists() else pd.DataFrame()
    if all_results.empty:
        panel_long = pd.DataFrame()
    else:
        panel_long = all_results[all_results["names"].astype(str).isin(panel)].copy()
        panel_long = panel_long.merge(genes, left_on="names", right_on="gene", how="left", validate="many_to_one")
        panel_long["direction"] = np.select(
            [panel_long["effect"] > 0, panel_long["effect"] < 0],
            ["positive", "negative"],
            default="zero",
        )
        panel_long["donor_fdr_supported"] = panel_long["pvals_adj"].fillna(1.0) < 0.05
        panel_long["measurement_boundary"] = (
            "Transcript association only; no protein abundance, maturation, surface localization, or cleavage measured."
        )
        panel_long = panel_long.sort_values(["panel_order", "anchor_gene", "population", "cohort"])
    panel_long.to_csv(output_dir / "adam17_output_donor_results_long.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    if not panel_long.empty:
        for (family, population, anchor, gene), frame in panel_long.groupby(
            ["primary_family", "population", "anchor_gene", "gene"], observed=True
        ):
            positive = int((frame["effect"] > 0).sum())
            negative = int((frame["effect"] < 0).sum())
            fdr_positive = int(((frame["effect"] > 0) & frame["donor_fdr_supported"]).sum())
            fdr_negative = int(((frame["effect"] < 0) & frame["donor_fdr_supported"]).sum())
            best = frame.loc[frame["pvals_adj"].fillna(np.inf).idxmin()]
            replicated = max(fdr_positive, fdr_negative) >= 2
            if replicated:
                evidence_level = "replicated_same_direction_donor_fdr"
            elif fdr_positive + fdr_negative:
                evidence_level = "single_cohort_donor_fdr"
            elif positive == 0 or negative == 0:
                evidence_level = "direction_consistent_without_donor_fdr"
            else:
                evidence_level = "direction_inconsistent_without_donor_fdr"
            summary_rows.append({
                "primary_family": family,
                "population": population,
                "anchor_gene": anchor,
                "gene": gene,
                "cohorts_tested": int(frame["cohort"].nunique()),
                "positive_cohorts": positive,
                "negative_cohorts": negative,
                "fdr_positive_cohorts": fdr_positive,
                "fdr_negative_cohorts": fdr_negative,
                "direction_consistent_across_tested_cohorts": positive == 0 or negative == 0,
                "replicated_same_direction_donor_fdr": replicated,
                "best_cohort": best["cohort"],
                "best_effect": float(best["effect"]),
                "best_nominal_p": float(best["pvals"]),
                "best_adjusted_p": float(best["pvals_adj"]),
                "strongest_transcript_evidence": evidence_level,
            })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["replicated_same_direction_donor_fdr", "best_adjusted_p", "primary_family", "population", "anchor_gene", "gene"],
            ascending=[False, True, True, True, True, True],
        )
    summary.to_csv(output_dir / "adam17_output_gene_summary.csv", index=False)

    strict = pd.DataFrame()
    if not sensitivity.empty:
        strict = sensitivity[sensitivity["names"].astype(str).isin(panel)].copy()
        if not strict.empty:
            strict["effect"] = np.where(
                strict["anchor_gene"].eq("RHBDF2"),
                strict["effect_per_log1p_rhbdf2_detection_per_1000_cells"],
                strict["effect_per_log1p_adam17_detection_per_1000_cells"],
            )
            strict = strict.merge(genes, left_on="names", right_on="gene", how="left", validate="many_to_one")
            strict["donor_fdr_supported"] = strict["pvals_adj"].fillna(1.0) < 0.05
            strict = strict.sort_values(["panel_order", "anchor_gene", "excluded_subpopulation"])
    strict.to_csv(output_dir / "adam17_output_strict_myeloid_leave_one_out.csv", index=False)

    stability_rows: list[dict[str, object]] = []
    if not strict.empty:
        for (family, anchor, gene), frame in strict.groupby(
            ["primary_family", "anchor_gene", "gene"], observed=True
        ):
            effects = frame["effect"].dropna().to_numpy(dtype=float)
            if effects.size == 0:
                continue
            stability_rows.append({
                "primary_family": family,
                "anchor_gene": anchor,
                "gene": gene,
                "n_leave_one_out_variants": int(effects.size),
                "minimum_effect": float(effects.min()),
                "maximum_effect": float(effects.max()),
                "median_effect": float(np.median(effects)),
                "all_same_nonzero_direction": bool(np.all(effects > 0) or np.all(effects < 0)),
                "variants_fdr_supported": int(frame["donor_fdr_supported"].sum()),
                "excluded_subpopulations": ";".join(sorted(frame["excluded_subpopulation"].astype(str).unique())),
            })
    stability = pd.DataFrame(stability_rows)
    if not stability.empty:
        stability = stability.sort_values(
            ["variants_fdr_supported", "all_same_nonzero_direction", "primary_family", "anchor_gene", "gene"],
            ascending=[False, False, True, True, True],
        )
    stability.to_csv(output_dir / "adam17_output_strict_myeloid_stability.csv", index=False)
    return panel_long, summary, coverage, stability


def make_adam17_output_figure(
    config: dict,
    panel_long: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot signed donor-level evidence strength for the prespecified output panel."""
    if panel_long.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    genes = adam17_output_gene_table(config)
    gene_order = genes.sort_values("panel_order")["gene"].tolist()
    contexts = (
        panel_long[["cohort", "population"]]
        .drop_duplicates()
        .assign(context=lambda x: x["cohort"] + "\n" + x["population"])
    )
    context_order = contexts["context"].tolist()
    figure, axes = plt.subplots(
        1,
        len(config["anchors"]),
        figsize=(20, max(11, 0.43 * len(gene_order))),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    max_strength = 6.0
    for ax, anchor in zip(axes, config["anchors"], strict=True):
        frame = panel_long[panel_long["anchor_gene"].eq(anchor)].copy()
        frame["context"] = frame["cohort"] + "\n" + frame["population"]
        frame["signed_strength"] = (
            np.sign(frame["effect"])
            * -np.log10(frame["pvals_adj"].clip(lower=1e-300, upper=1.0))
        ).clip(lower=-max_strength, upper=max_strength)
        matrix = frame.pivot(index="gene", columns="context", values="signed_strength").reindex(
            index=gene_order, columns=context_order
        )
        sns.heatmap(
            matrix,
            mask=matrix.isna(),
            cmap="vlag",
            center=0,
            vmin=-max_strength,
            vmax=max_strength,
            linewidths=0.45,
            linecolor="white",
            cbar=ax is axes[-1],
            cbar_kws={"label": "signed -log10 donor FDR"} if ax is axes[-1] else None,
            ax=ax,
        )
        significant = frame[frame["donor_fdr_supported"]]
        row_lookup = {gene: index for index, gene in enumerate(gene_order)}
        column_lookup = {context: index for index, context in enumerate(context_order)}
        for row in significant.itertuples(index=False):
            if row.gene in row_lookup and row.context in column_lookup:
                ax.text(
                    column_lookup[row.context] + 0.5,
                    row_lookup[row.gene] + 0.5,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color="black",
                )
        ax.set_title(f"{anchor} anchor")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=45, labelsize=9)
        ax.tick_params(axis="y", labelsize=10)
    figure.suptitle(
        "ADAM17 substrate/output transcript associations across donor-aware analyses\n"
        "Stars mark donor FDR < 0.05; color does not measure ectodomain shedding",
        fontsize=20,
        fontweight="bold",
    )
    dpi = int(config.get("figures", {}).get("dpi", 600))
    figure.savefig(output_dir / "figure7_adam17_substrate_output_panel.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / "figure7_adam17_substrate_output_panel.pdf", bbox_inches="tight")
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "No rows met this criterion."
    display = frame[columns].copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.4g}" if pd.notna(value) else "")
    display = display.fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def write_adam17_output_report(
    config: dict,
    panel_long: pd.DataFrame,
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    stability: pd.DataFrame,
    output_dir: Path,
) -> None:
    if not config.get("adam17_output_families"):
        return
    role_table = adam17_output_gene_table(config)
    supported = panel_long[panel_long.get("donor_fdr_supported", False)].copy() if not panel_long.empty else pd.DataFrame()
    if not supported.empty:
        supported = supported.sort_values(["pvals_adj", "pvals"]).head(30)
    replicated = summary[summary.get("replicated_same_direction_donor_fdr", False)].copy() if not summary.empty else pd.DataFrame()
    stable = stability[stability.get("all_same_nonzero_direction", False)].copy() if not stability.empty else pd.DataFrame()
    present = int(coverage["availability"].eq("present").sum()) if not coverage.empty else 0
    total = int(len(coverage))
    family_lines = "\n".join(
        f"- **{family.replace('_', ' ')}:** {', '.join(spec['genes'])} ({spec['evidence_class']})."
        for family, spec in config["adam17_output_families"].items()
    )
    report = f"""# ADAM17 substrate and output extension: methods and data results

## Analytical question

This focused extension asked whether transcripts encoding known or literature-supported ADAM17 substrates, microglial receptor axes, EGFR ligands, and downstream signaling readouts covary with either continuous RHBDF2 or ADAM17 detection at donor level. It does **not** infer cleavage from substrate-gene RNA abundance. ADAM17 maturation, surface localization, catalytic activity, soluble ectodomain abundance, and receptor occupancy were not measured.

## Prespecified panel

{family_lines}

The panel contained {len(role_table)} unique genes. The primary analysis used pathology-adjusted donor pseudobulk separately within each cohort and population. Same-direction FDR support in at least two independent cohorts was required for transcript-level replication. Strict SEA-AD myeloid leave-one-subpopulation-out analyses tested sensitivity to pooled-myeloid composition.

## Matrix coverage

{present} of {total} planned cohort-population-anchor-gene cells were available in the working matrices. The expanded SEA-AD targeted object was rebuilt from the author `layers/UMIs` raw-count layer so the new panel was not evaluated against the earlier restricted 61-gene object. SEA-AD donor pseudobulk used summed source full-library UMI totals as its CPM denominator; both retained-panel counts and full-library normalization totals are recorded in the donor analysis matrices.

## Donor-level FDR-supported rows

{_markdown_table(supported, ['cohort', 'population', 'anchor_gene', 'gene', 'primary_family', 'effect', 'pvals_adj']) if not supported.empty else 'No panel row reached donor-level FDR < 0.05.'}

## Cross-cohort replication

{_markdown_table(replicated, ['population', 'anchor_gene', 'gene', 'cohorts_tested', 'fdr_positive_cohorts', 'fdr_negative_cohorts', 'best_adjusted_p']) if not replicated.empty else 'No non-self ADAM17-output gene achieved same-direction donor-level FDR support in two independent cohorts.'}

## Strict-myeloid composition stability

{_markdown_table(stable.head(30), ['anchor_gene', 'gene', 'n_leave_one_out_variants', 'minimum_effect', 'maximum_effect', 'variants_fdr_supported']) if not stable.empty else 'No output-panel row had a stable nonzero direction across all eligible leave-one-subpopulation-out variants.'}

## Interpretation boundary

Positive transcript covariance can nominate a substrate axis for matched protein or soluble-ectodomain measurement, but it cannot establish increased ADAM17 cleavage. A substrate transcript can be unchanged while shedding changes, and it can covary even when cleavage does not change. The decisive follow-up assays are mature versus pro-ADAM17 protein, cell-surface ADAM17 and substrate abundance, soluble TNF/TNFR/IL6R/TREM2/CSF1R ectodomains, substrate-resolved cleavage kinetics, and perturbation with iRhom2- or ADAM17-selective interventions.
"""
    (output_dir / "ADAM17_SUBSTRATE_OUTPUT_METHODS_AND_RESULTS.md").write_text(report, encoding="utf-8")


def make_figures(
    config: dict,
    eligibility: pd.DataFrame,
    concordance: pd.DataFrame,
    genes: pd.DataFrame,
    donor: pd.DataFrame,
    sensitivity: pd.DataFrame,
    anchor_metadata: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(config["figures"]["dpi"])
    sns.set_theme(style="whitegrid", context="talk")
    plot = eligibility.copy()
    plot["label"] = plot["cohort"] + " | " + plot["population"]
    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)
    sns.scatterplot(data=plot, x="detection_fraction", y="label", hue="anchor_gene", size="n_cells", sizes=(70, 420), palette={"RHBDF2": "#006d77", "ADAM17": "#e76f51"}, ax=ax)
    ax.set_xlabel("Fraction of cells with detected transcript")
    ax.set_ylabel("")
    ax.set_title("Dual-anchor eligibility across AD cohorts and populations")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, title="Anchor / cell count")
    fig.savefig(output_dir / "figure1_dual_anchor_detection.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if not concordance.empty:
        fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
        sns.barplot(data=concordance, x="effect_spearman_rho", y="population", hue="cohort", ax=ax)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Spearman correlation of transcriptome-wide anchor effects")
        ax.set_ylabel("")
        ax.set_title("RHBDF2 versus ADAM17 association concordance")
        fig.savefig(output_dir / "figure2_anchor_effect_concordance.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    if not genes.empty:
        full = genes.groupby(["cohort", "population"], observed=True).size().sort_values(ascending=False).index[0]
        frame = genes[(genes.cohort == full[0]) & (genes.population == full[1])].copy()
        fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        sns.scatterplot(data=frame, x="slope_per_log1p_rhbdf2", y="slope_per_log1p_adam17", hue="classification", alpha=0.65, s=24, ax=ax)
        ax.axhline(0, color="grey", linewidth=0.8); ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_title(f"Dual-anchor gene effects: {full[0]} {full[1]}")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        fig.savefig(output_dir / "figure3_dual_anchor_effect_scatter.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    if not donor.empty:
        matrix = donor.pivot(index="population", columns="cohort", values="donor_anchor_spearman_rho")
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        sns.heatmap(matrix, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax)
        ax.set_title("Donor-level RHBDF2-ADAM17 detection correlation")
        ax.set_xlabel(""); ax.set_ylabel("")
        fig.savefig(output_dir / "figure4_donor_anchor_correlation.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    if not sensitivity.empty:
        genes_to_plot = [gene for gene in ["RHBDF2", "ADAM17", "TREM2", "TYROBP", "SYK", "GPNMB"] if gene in sensitivity.names.values]
        fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), sharey=True, constrained_layout=True)
        palette = sns.color_palette("colorblind", sensitivity.excluded_subpopulation.nunique())
        labels = sorted(sensitivity.excluded_subpopulation.astype(str).unique())
        gene_order = genes_to_plot
        for ax, anchor in zip(axes, config["anchors"], strict=True):
            effect_column = f"effect_per_log1p_{key(anchor)}_detection_per_1000_cells"
            frame = sensitivity[
                sensitivity.names.isin(genes_to_plot)
                & sensitivity.anchor_gene.eq(anchor)
                & ~sensitivity.self_association.astype(bool)
            ].copy()
            y_lookup = {gene: index for index, gene in enumerate(gene_order)}
            for label_index, label in enumerate(labels):
                group = frame[frame.excluded_subpopulation.astype(str).eq(label)]
                y = group.names.map(y_lookup).to_numpy(dtype=float)
                offset = (label_index - (len(labels) - 1) / 2) * 0.045
                ax.scatter(group[effect_column], y + offset, color=palette[label_index], s=70, alpha=0.9, label=label)
            ax.axvline(0, color="black", linewidth=1)
            ax.set_yticks(range(len(gene_order)), gene_order)
            ax.invert_yaxis()
            ax.set_title(f"{anchor} anchor")
            ax.set_xlabel("Donor-level anchor-associated effect")
            ax.set_ylabel("")
        axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, title="Excluded subpopulation")
        fig.suptitle("SEA-AD strict-myeloid leave-one-subpopulation-out sensitivity", fontsize=22)
        fig.savefig(output_dir / "figure5_myeloid_composition_sensitivity.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    if not anchor_metadata.empty:
        top = anchor_metadata.nsmallest(25, "pval").copy()
        top["minus_log10_p"] = -np.log10(top["pval"].clip(lower=1e-300))
        top["label"] = top["cohort"] + " | " + top["population"] + " | " + top["metadata_variable"]
        fig, ax = plt.subplots(figsize=(12, max(7, 0.35 * len(top))), constrained_layout=True)
        sns.barplot(data=top, x="minus_log10_p", y="label", hue="anchor_gene", palette={"RHBDF2": "#006d77", "ADAM17": "#e76f51"}, ax=ax)
        ax.axvline(-math.log10(0.05), color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("-log10 nominal P value"); ax.set_ylabel("")
        ax.set_title("Nominal donor-level anchor-metadata ranking\nNo direct association passed global FDR < 0.05")
        ax.legend(title="Anchor")
        fig.savefig(output_dir / "figure6_anchor_metadata_associations.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def write_report(
    config: dict,
    eligibility: pd.DataFrame,
    manifest: pd.DataFrame,
    inferred: pd.DataFrame,
    concordance: pd.DataFrame,
    replication: pd.DataFrame,
    metadata_coverage: pd.DataFrame,
    anchor_metadata: pd.DataFrame,
    interactions: pd.DataFrame,
    output_dir: Path,
) -> None:
    eligible = eligibility[eligibility.status == "eligible"]
    enrichment_failures = []
    for value in manifest.enrichment_status.dropna():
        for item in json.loads(value):
            if item.get("status") != "completed":
                enrichment_failures.append(item)
    text = f"""# Committee-requested dual-anchor extension: methods and data results

## Scope

This run evaluated RHBDF2 and ADAM17 as independent transcript anchors in microglial/myeloid and endothelial populations. It used continuous cell-level association, candidate hurdle models, donor-level pseudobulk, strict SEA-AD myeloid pooling, leave-one-subpopulation-out sensitivity, donor metadata association, anchor-by-metadata interaction testing, enrichment, and cross-cohort concordance. All RNA findings remain association-level and do not measure ADAM17 maturation, surface localization, catalytic activity, or substrate shedding.

## Data eligibility

The eligibility table contains {len(eligibility)} cohort-population-anchor combinations; {len(eligible)} met the preregistered detection and donor thresholds. The primary analysis manifest contains {len(manifest)} completed anchor runs. SEA-AD results use a targeted {int(eligibility.loc[eligibility.cohort.eq('SEAAD_MTG'), 'n_genes'].max())}-gene matrix and are candidate/module validation rather than transcriptome-wide discovery.

## Methods

Raw-count population subsets were normalized to 10,000 counts per cell and log1p transformed for cell-level continuous and hurdle analyses. Transcriptome-wide linear association regressed each gene on continuous anchor expression. Candidate hurdle models separated transcript detection from positive-expression abundance and adjusted for donor using fixed effects or cluster-robust covariance according to dataset size. Donor pseudobulk summed source counts by donor, converted values to log1p counts per million, and regressed each transcript on log1p anchor detection per 1,000 cells with pathology adjustment.

The strict SEA-AD myeloid population included `Micro-PVM*` and `Monocyte` labels and excluded `Lymphocyte`. Composition sensitivity removed each sufficiently represented Micro-PVM subpopulation in turn. GSE157827 and GSE237718 unassigned cells were considered eligible for inferred-myeloid sensitivity only if microglial score exceeded all other lineage scores and at least two identity markers were detected.

Metadata models used donor-level values. Anchor-metadata tests evaluated whether donor anchor detection varied by each eligible cohort-native variable. Candidate and module effect-modification models compared `target ~ anchor + metadata` with `target ~ anchor + metadata + anchor:metadata`. Categorical levels required at least {config['thresholds']['minimum_donors_per_metadata_level']} donors; models required at least {config['thresholds']['minimum_residual_degrees_freedom']} residual degrees of freedom. FDR was controlled within each cohort, population, anchor, metadata variable, and target family. No universal disease-stage score was created.

## Data results

- Eligible anchor-population combinations: {len(eligible)} of {len(eligibility)}.
- Completed donor-level runs: {int((manifest.donor_status == 'completed').sum())} of {len(manifest)}.
- Inferred-myeloid sensitivity statuses: {json.dumps(dict(zip(inferred.cohort, inferred.status)), sort_keys=True)}.
- Dual-anchor concordance comparisons: {len(concordance)}.
- Candidate replication rows: {len(replication)}.
- Eligible metadata model combinations: {int((metadata_coverage.status == 'eligible').sum()) if not metadata_coverage.empty else 0}.
- FDR-supported anchor-metadata associations: {int((anchor_metadata.pvals_adj < 0.05).sum()) if not anchor_metadata.empty else 0}.
- FDR-supported anchor-by-metadata target interactions: {int((interactions.pvals_adj < 0.05).sum()) if not interactions.empty else 0}.
- Network enrichment failures recorded without terminating the run: {len(enrichment_failures)}.

The CSV outputs provide exact effect estimates, adjusted P values, donor counts, model degrees of freedom, stage-specific slopes, and concordance classifications. These data should be interpreted from the tables rather than from significance counts alone.

## Output map

- `eligibility/dual_anchor_population_eligibility.csv`: cell/donor detection eligibility.
- `analyses/`: independent continuous, hurdle, pseudobulk, ranked, and enrichment outputs by cohort/population/anchor.
- `myeloid/`: strict SEA-AD composition and leave-one-subpopulation-out results.
- `metadata/`: coverage, anchor-metadata associations, interaction tests, and stage-specific slopes.
- `synthesis/`: gene classification, donor-anchor correlation, and cross-cohort candidate replication.
- `figures/`: high-resolution summary figures.
"""
    (output_dir / "COMMITTEE_EXTENSION_METHODS_AND_DATA_RESULTS.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    output_dir = (ROOT / config["output_dir"]).resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "effective_config.yaml").write_text(yaml.safe_dump({k: v for k, v in config.items() if not k.startswith("_")}, sort_keys=False), encoding="utf-8")

    eligibility_dir = output_dir / "eligibility"; eligibility_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "analysis_manifest.csv"
    eligibility_path = eligibility_dir / "dual_anchor_population_eligibility.csv"
    if args.resume and manifest_path.exists() and eligibility_path.exists():
        print("[1/7] Reusing completed dual-anchor eligibility", flush=True)
        eligibility = pd.read_csv(eligibility_path)
        print("[2/7] Reusing completed primary analysis manifest", flush=True)
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest["population"] != "Strict pooled myeloid"].copy()
    else:
        print("[1/7] Dual-anchor eligibility", flush=True)
        eligibility, _ = build_eligibility(config, eligibility_dir)
        print("[2/7] Primary microglial and endothelial analyses", flush=True)
        manifest = run_primary_analyses(config, output_dir, args.skip_network_enrichment)

    print("[3/7] Strict SEA-AD myeloid pooling and composition sensitivity", flush=True)
    myeloid_dir = output_dir / "myeloid"; myeloid_dir.mkdir(exist_ok=True)
    myeloid_manifest, sensitivity = run_strict_seaad_myeloid(config, myeloid_dir, args.skip_network_enrichment)
    manifest = pd.concat([manifest, myeloid_manifest], ignore_index=True)
    manifest.to_csv(output_dir / "analysis_manifest.csv", index=False)

    print("[4/7] Optional inferred-myeloid eligibility", flush=True)
    inferred = evaluate_inferred_myeloid(config, myeloid_dir)

    print("[5/7] Donor metadata, stage, genotype, region, and condition models", flush=True)
    metadata_dir = output_dir / "metadata"; metadata_dir.mkdir(exist_ok=True)
    coverage, anchor_metadata, interactions, slopes = run_metadata_models(config, manifest, metadata_dir)

    print("[6/7] Dual-anchor and cross-cohort synthesis", flush=True)
    synthesis_dir = output_dir / "synthesis"; synthesis_dir.mkdir(exist_ok=True)
    concordance, genes, donor = synthesize(config, manifest, synthesis_dir)
    replication = cross_cohort_candidates(config, manifest, synthesis_dir)
    panel_long, panel_summary, panel_coverage, panel_stability = summarize_adam17_outputs(
        config,
        manifest,
        sensitivity,
        synthesis_dir,
    )

    print("[7/7] Figures and methods/data report", flush=True)
    figures_dir = output_dir / "figures"
    make_figures(config, eligibility, concordance, genes, donor, sensitivity, anchor_metadata, figures_dir)
    make_adam17_output_figure(config, panel_long, figures_dir)
    write_report(config, eligibility, manifest, inferred, concordance, replication, coverage, anchor_metadata, interactions, output_dir)
    write_adam17_output_report(
        config,
        panel_long,
        panel_summary,
        panel_coverage,
        panel_stability,
        output_dir,
    )
    pd.DataFrame([{"status": "completed", "config": str(config_path), "output_dir": str(output_dir)}]).to_csv(output_dir / "run_status.csv", index=False)
    print(f"Completed: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
