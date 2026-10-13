"""Conservative cross-cohort synthesis for independent Alzheimer analyses."""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.errors import EmptyDataError


CELL_TYPE_ALIASES = {
    "astrocyte": "Astrocytes",
    "astrocytes": "Astrocytes",
    "microglia": "Microglia",
    "oligodendrocyte": "Oligodendrocytes",
    "oligodendrocytes": "Oligodendrocytes",
    "opc": "OPCs",
    "opcs": "OPCs",
    "excitatory neurons": "Excitatory neurons",
    "inhibitory neurons": "Inhibitory neurons",
    "endothelial cells": "Endothelial cells",
    "pericytes": "Pericytes",
}


def canonical_cell_type(value: object) -> str:
    """Normalize only known spelling/plural aliases; preserve unknown labels."""
    text = str(value).strip()
    return CELL_TYPE_ALIASES.get(text.lower(), text)


def load_donor_candidate_results(
    cohort_directories: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load donor-aware results and report every included or unavailable cohort."""
    frames: list[pd.DataFrame] = []
    status_rows = []
    relative = Path("cross_cell_type/donor_candidate_associations_all_cell_types.csv")
    for cohort, directory in cohort_directories.items():
        path = directory / relative
        if not path.exists():
            status_rows.append({"cohort": cohort, "result_type": "candidate", "status": "unavailable", "path": str(path)})
            continue
        try:
            frame = pd.read_csv(path)
        except EmptyDataError:
            status_rows.append({"cohort": cohort, "result_type": "candidate", "status": "empty_no_donor_results", "path": str(path)})
            continue
        required = {"names", "cell_type", "pvals", "pvals_adj", "n_donors"}
        missing = required.difference(frame.columns)
        if missing:
            status_rows.append({"cohort": cohort, "result_type": "candidate", "status": "invalid_schema", "path": str(path), "reason": ";".join(sorted(missing))})
            continue
        frame["cohort"] = cohort
        frame["cell_type"] = frame["cell_type"].map(canonical_cell_type)
        frame["analysis_stratum"] = frame.get("analysis_stratum", "all")
        frame["analysis_subset"] = frame.get("analysis_subset", "primary")
        frames.append(frame)
        status_rows.append({"cohort": cohort, "result_type": "candidate", "status": "included", "path": str(path), "rows": int(frame.shape[0])})
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return combined, pd.DataFrame(status_rows)


def load_donor_program_results(
    cohort_directories: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load donor-aware module results with the same provenance contract."""
    frames: list[pd.DataFrame] = []
    status_rows = []
    relative = Path("programs/donor_program_associations.csv")
    for cohort, directory in cohort_directories.items():
        path = directory / relative
        if not path.exists():
            status_rows.append({"cohort": cohort, "result_type": "program", "status": "unavailable", "path": str(path)})
            continue
        try:
            frame = pd.read_csv(path)
        except EmptyDataError:
            status_rows.append({"cohort": cohort, "result_type": "program", "status": "empty_no_donor_results", "path": str(path)})
            continue
        required = {"program", "cell_type", "pval", "pvals_adj_global", "n_donors", "effect"}
        missing = required.difference(frame.columns)
        if missing:
            status_rows.append({"cohort": cohort, "result_type": "program", "status": "invalid_schema", "path": str(path), "reason": ";".join(sorted(missing))})
            continue
        frame["cohort"] = cohort
        frame["cell_type"] = frame["cell_type"].map(canonical_cell_type)
        frame["analysis_stratum"] = frame.get("analysis_stratum", "all")
        frame["analysis_subset"] = frame.get("analysis_subset", "primary")
        frames.append(frame)
        status_rows.append({"cohort": cohort, "result_type": "program", "status": "included", "path": str(path), "rows": int(frame.shape[0])})
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return combined, pd.DataFrame(status_rows)


def summarize_replication(
    frame: pd.DataFrame,
    feature_column: str,
    effect_column: str,
    p_column: str,
    adjusted_p_column: str,
) -> pd.DataFrame:
    """Count cohort-level directional support without pooling cells or effect sizes."""
    output_columns = [
        "cell_type", feature_column, "cohorts_tested", "contexts_tested",
        "positive_nominal_cohorts", "negative_nominal_cohorts",
        "positive_fdr_cohorts", "negative_fdr_cohorts", "best_pval",
        "best_adjusted_pval", "best_context", "replication_tier",
    ]
    if frame.empty:
        return pd.DataFrame(columns=output_columns)
    work = frame.copy()
    work[effect_column] = pd.to_numeric(work[effect_column], errors="coerce")
    work[p_column] = pd.to_numeric(work[p_column], errors="coerce")
    work[adjusted_p_column] = pd.to_numeric(work[adjusted_p_column], errors="coerce")
    rows = []
    for (cell_type, feature), group in work.groupby(["cell_type", feature_column], dropna=False):
        # A cohort can contribute at most once to each direction, even if several regions were tested.
        cohort_support = []
        for cohort, cohort_group in group.groupby("cohort"):
            nominal_pos = bool(((cohort_group[p_column] < 0.05) & (cohort_group[effect_column] > 0)).any())
            nominal_neg = bool(((cohort_group[p_column] < 0.05) & (cohort_group[effect_column] < 0)).any())
            fdr_pos = bool(((cohort_group[adjusted_p_column] < 0.05) & (cohort_group[effect_column] > 0)).any())
            fdr_neg = bool(((cohort_group[adjusted_p_column] < 0.05) & (cohort_group[effect_column] < 0)).any())
            cohort_support.append((str(cohort), nominal_pos, nominal_neg, fdr_pos, fdr_neg))
        positive_nominal = sorted(x[0] for x in cohort_support if x[1])
        negative_nominal = sorted(x[0] for x in cohort_support if x[2])
        positive_fdr = sorted(x[0] for x in cohort_support if x[3])
        negative_fdr = sorted(x[0] for x in cohort_support if x[4])
        if max(len(positive_fdr), len(negative_fdr)) >= 2:
            tier = "replicated_fdr"
        elif max(len(positive_nominal), len(negative_nominal)) >= 2:
            tier = "replicated_nominal"
        elif max(len(positive_fdr), len(negative_fdr)) == 1:
            tier = "single_cohort_fdr"
        elif max(len(positive_nominal), len(negative_nominal)) == 1:
            tier = "single_cohort_nominal"
        else:
            tier = "no_directional_support"
        best_index = group[p_column].fillna(np.inf).idxmin()
        best = group.loc[best_index]
        context = f"{best['cohort']}|{best.get('analysis_stratum', 'all')}"
        rows.append({
            "cell_type": cell_type,
            feature_column: feature,
            "cohorts_tested": int(group["cohort"].nunique()),
            "contexts_tested": int(group[["cohort", "analysis_stratum"]].drop_duplicates().shape[0]),
            "positive_nominal_cohorts": ";".join(positive_nominal),
            "negative_nominal_cohorts": ";".join(negative_nominal),
            "positive_fdr_cohorts": ";".join(positive_fdr),
            "negative_fdr_cohorts": ";".join(negative_fdr),
            "best_pval": best[p_column],
            "best_adjusted_pval": best[adjusted_p_column],
            "best_context": context,
            "replication_tier": tier,
        })
    return pd.DataFrame(rows).sort_values(
        ["replication_tier", "best_pval"],
        key=lambda s: s.map({"replicated_fdr": 0, "replicated_nominal": 1, "single_cohort_fdr": 2, "single_cohort_nominal": 3, "no_directional_support": 4}) if s.name == "replication_tier" else s,
    )


def build_ligand_receptor_gate(
    candidate_summary: pd.DataFrame,
    program_summary: pd.DataFrame,
    included_donor_cohorts: int,
) -> pd.DataFrame:
    """Permit cross-cell communication only after stable independent replication."""
    candidate_replicated = int(candidate_summary["replication_tier"].eq("replicated_fdr").sum()) if not candidate_summary.empty else 0
    program_replicated = int(program_summary["replication_tier"].eq("replicated_fdr").sum()) if not program_summary.empty else 0
    passed = included_donor_cohorts >= 2 and (candidate_replicated + program_replicated) > 0
    reason = (
        "At least two donor-valid cohorts and at least one same-cell-type FDR-replicated feature are available."
        if passed
        else "Requires at least two donor-valid cohorts plus a same-cell-type, same-direction FDR-replicated candidate or module."
    )
    return pd.DataFrame([{
        "gate": "cross_cell_type_ligand_receptor",
        "status": "eligible_not_run" if passed else "withheld",
        "included_donor_cohorts": included_donor_cohorts,
        "replicated_fdr_candidates": candidate_replicated,
        "replicated_fdr_programs": program_replicated,
        "reason": reason,
    }])


def plot_replication_heatmap(
    frame: pd.DataFrame,
    feature_column: str,
    effect_column: str,
    p_column: str,
    output_path: Path,
    max_features: int = 24,
) -> None:
    """Plot signed -log10(p) values with compact labels and fixed dimensions."""
    if frame.empty:
        return
    work = frame.copy()
    work["context"] = work["cohort"].astype(str) + " | " + work["cell_type"].astype(str)
    if "analysis_stratum" in work and work["analysis_stratum"].astype(str).ne("all").any():
        work["context"] += " | " + work["analysis_stratum"].astype(str)
    work["signed_log10_p"] = np.sign(pd.to_numeric(work[effect_column], errors="coerce")) * -np.log10(
        pd.to_numeric(work[p_column], errors="coerce").clip(lower=np.finfo(float).tiny)
    )
    ranking = work.groupby(feature_column)["signed_log10_p"].apply(lambda x: x.abs().max()).nlargest(max_features).index
    matrix = work[work[feature_column].isin(ranking)].pivot_table(
        index=feature_column, columns="context", values="signed_log10_p", aggfunc="max"
    ).reindex(ranking)
    if matrix.empty:
        return
    width = max(8.5, 0.55 * matrix.shape[1] + 3)
    height = max(5.5, 0.33 * matrix.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    limit = max(2.0, float(np.nanmax(np.abs(matrix.to_numpy()))))
    sns.heatmap(matrix, cmap="vlag", center=0, vmin=-limit, vmax=limit, ax=ax, cbar_kws={"label": "signed -log10(nominal p)"})
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=55, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
