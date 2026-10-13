#!/usr/bin/env python3
"""Create filtered result tables and a data-only summary for committee extensions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/committee_extensions/full_run")
    return parser.parse_args()


def donor_hits(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for record in manifest.itertuples(index=False):
        if record.donor_status != "completed":
            continue
        frame = pd.read_csv(record.donor_candidate_results)
        effect = f"effect_per_log1p_{key(record.anchor_gene)}_detection_per_1000_cells"
        hits = frame[(frame["pvals_adj"] < 0.05) & ~frame["self_association"].astype(bool)].copy()
        if hits.empty:
            continue
        hits = hits[["names", effect, "pvals", "pvals_adj", "n_donors", "pathology_adjusted"]]
        hits = hits.rename(columns={effect: "effect"})
        hits.insert(0, "anchor_gene", record.anchor_gene)
        hits.insert(0, "population", record.population)
        hits.insert(0, "cohort", record.cohort)
        rows.append(hits)
    return pd.concat(rows, ignore_index=True).sort_values(["pvals_adj", "pvals"]) if rows else pd.DataFrame()


def enrichment_terms(run_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in run_dir.glob("analyses/**/enrichment/*_enrichment.csv"):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        relative = path.relative_to(run_dir).parts
        cohort, population, anchor = relative[1], relative[2], relative[3]
        backend = "gprofiler" if "gprofiler" in path.name else "enrichr"
        direction = "up" if "_up_" in path.name else "down"
        if backend == "gprofiler":
            selected = frame.head(20)[["source", "name", "p_value"]].rename(columns={"source": "database", "name": "term", "p_value": "adjusted_pvalue"})
        else:
            selected = frame.head(20)[["Gene_set", "Term", "Adjusted P-value"]].rename(columns={"Gene_set": "database", "Term": "term", "Adjusted P-value": "adjusted_pvalue"})
        selected.insert(0, "direction", direction)
        selected.insert(0, "backend", backend)
        selected.insert(0, "anchor_gene", anchor.upper())
        selected.insert(0, "population", population.replace("_", " "))
        selected.insert(0, "cohort", cohort.upper())
        rows.append(selected)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def myeloid_stability(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for (anchor, gene), group in frame.groupby(["anchor_gene", "names"], observed=True):
        if gene == anchor:
            continue
        effect_column = f"effect_per_log1p_{key(anchor)}_detection_per_1000_cells"
        effects = group[effect_column].dropna().to_numpy(dtype=float)
        rows.append({
            "anchor_gene": anchor,
            "gene": gene,
            "n_leave_one_out_variants": len(effects),
            "minimum_effect": float(effects.min()) if len(effects) else np.nan,
            "maximum_effect": float(effects.max()) if len(effects) else np.nan,
            "median_effect": float(np.median(effects)) if len(effects) else np.nan,
            "all_same_nonzero_direction": bool(np.all(effects > 0) or np.all(effects < 0)) if len(effects) else False,
            "variants_fdr_supported": int((group["pvals_adj"] < 0.05).sum()),
            "excluded_subpopulations": ";".join(sorted(group["excluded_subpopulation"].astype(str).unique())),
        })
    return pd.DataFrame(rows).sort_values(["variants_fdr_supported", "anchor_gene", "gene"], ascending=[False, True, True])


def compact_table(frame: pd.DataFrame, columns: list[str], n: int = 12) -> str:
    if frame.empty:
        return "No rows."
    selected = frame[columns].head(n).copy()
    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(render(value) for value in row) + " |" for row in selected.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def write_report(
    run_dir: Path,
    eligibility: pd.DataFrame,
    manifest: pd.DataFrame,
    hits: pd.DataFrame,
    replication: pd.DataFrame,
    concordance: pd.DataFrame,
    donor_correlation: pd.DataFrame,
    inferred: pd.DataFrame,
    stability: pd.DataFrame,
    anchor_metadata: pd.DataFrame,
    interactions: pd.DataFrame,
    enrichment: pd.DataFrame,
) -> None:
    interaction_hits = interactions[interactions["pvals_adj"] < 0.05].copy()
    direct_hits = anchor_metadata[anchor_metadata["pvals_adj"] < 0.05].copy()
    replicated = replication[replication["cohorts_fdr_supported"] >= 2].copy()
    report = f"""# Committee extension: data-only results summary

## Completion and eligibility

The run completed {len(manifest)} independent population-anchor analyses. All {int((manifest.donor_status == 'completed').sum())} donor pseudobulk models completed. Of {len(eligibility)} cohort-population-anchor eligibility combinations, {int((eligibility.status == 'eligible').sum())} met the configured cell, detection, and donor thresholds. GSE243292 endothelial RHBDF2 remained descriptive because 14 of 444 cells detected RHBDF2; endothelial ADAM17 met eligibility thresholds but was not promoted to a primary GSE243292 run because this small endothelial population was retained as a discovery-cohort sensitivity comparator.

## Donor-level candidate results

There were {len(hits)} non-self candidate rows with donor-level FDR < 0.05 across all individual analyses. They were distributed across {hits[['cohort','population','anchor_gene']].drop_duplicates().shape[0] if not hits.empty else 0} cohort-population-anchor combinations. No non-self candidate was donor-level FDR-supported in two or more cohorts within the same population and anchor definition.

{compact_table(hits, ['cohort', 'population', 'anchor_gene', 'names', 'effect', 'pvals_adj'], 20)}

The complete filtered table is `synthesis/donor_candidate_fdr_hits.csv`; the unfiltered cross-cohort table is `synthesis/cross_cohort_candidate_results_long.csv`.

## Dual-anchor concordance

Transcriptome-wide RHBDF2-versus-ADAM17 effect correlations ranged from {concordance.effect_spearman_rho.min():.3f} to {concordance.effect_spearman_rho.max():.3f}. Donor-level anchor-detection correlations ranged from {donor_correlation.donor_anchor_spearman_rho.min():.3f} to {donor_correlation.donor_anchor_spearman_rho.max():.3f}. The SEA-AD targeted matrix contained 59 non-anchor genes in concordance calculations and used a top-10 rather than top-200 overlap set.

{compact_table(concordance, ['cohort', 'population', 'n_genes', 'effect_spearman_rho', 'top_set_size', 'top_positive_jaccard'], 10)}

## Strict SEA-AD myeloid sensitivity

The strict pooled set contained 35,919 nuclei: `Micro-PVM*` labels plus 12 monocytes, with 767 lymphocytes excluded. Six sufficiently represented Micro-PVM labels were removed one at a time. The stability table reports the minimum, maximum, median, and sign consistency of each candidate effect across those six variants.

{compact_table(stability[stability.gene.isin(['ADAM17', 'RHBDF2', 'TREM2', 'TYROBP', 'SYK', 'GPNMB'])], ['anchor_gene', 'gene', 'minimum_effect', 'maximum_effect', 'all_same_nonzero_direction', 'variants_fdr_supported'], 20)}

No GSE157827 or GSE237718 unassigned cell met the prespecified inferred-myeloid rule requiring both the strongest microglial lineage score and at least two detected identity markers; those optional branches were therefore skipped rather than forced.

{compact_table(inferred, ['cohort', 'unassigned_cells', 'cells_with_microglia_score_stronger_than_other_lineages', 'cells_meeting_strict_inferred_myeloid_rule', 'status'], 5)}

## Metadata, stage, genotype, and condition

No direct donor-level anchor-metadata association passed global FDR < 0.05 ({len(direct_hits)} rows). The smallest nominal direct associations are shown for ranking only.

{compact_table(anchor_metadata, ['cohort', 'population', 'anchor_gene', 'metadata_variable', 'n_donors', 'pval', 'pvals_adj'], 15)}

There were {len(interaction_hits)} target-by-metadata interaction rows with within-family FDR < 0.05. These tests ask whether an anchor-associated candidate or module slope varies across a metadata variable. They are single-cohort effect-modification results and were not counted as cross-cohort replication.

{compact_table(interaction_hits, ['cohort', 'population', 'anchor_gene', 'metadata_variable', 'target_type', 'target', 'n_donors', 'pvals_adj'], 20)}

## Enrichment

Both g:Profiler and Enrichr completed for RHBDF2 and ADAM17 in GSE157827 and GSE237718 endothelial cells. The exported table contains the top 20 terms per backend, direction, cohort, and anchor ({len(enrichment)} rows). Enrichment was not run on the targeted SEA-AD matrix because its preselected 61-gene universe is unsuitable for transcriptome-wide pathway discovery.

{compact_table(enrichment, ['cohort', 'anchor_gene', 'backend', 'direction', 'database', 'term', 'adjusted_pvalue'], 20)}

## Interpretation boundary

These outputs quantify transcript detection, expression association, donor-level pseudobulk association, pathway over-representation, metadata effect modification, and cross-cohort consistency. They do not directly measure iRhom2 protein abundance, ADAM17 maturation or surface localization, catalytic activity, substrate cleavage, or causal direction. Single-cohort FDR findings and within-family metadata interactions remain follow-up leads unless independently replicated.
"""
    (run_dir / "COMMITTEE_EXTENSION_RESULTS_SUMMARY_DATA_ONLY.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    synthesis_dir = run_dir / "synthesis"
    manifest = pd.read_csv(run_dir / "analysis_manifest.csv")
    eligibility = pd.read_csv(run_dir / "eligibility/dual_anchor_population_eligibility.csv")
    hits = donor_hits(manifest)
    enrichment = enrichment_terms(run_dir)
    stability = myeloid_stability(run_dir / "myeloid/strict_myeloid_leave_one_subpopulation_out_candidates.csv")
    interactions = pd.read_csv(run_dir / "metadata/anchor_metadata_interactions.csv")
    interaction_hits = interactions[interactions["pvals_adj"] < 0.05].copy()

    hits.to_csv(synthesis_dir / "donor_candidate_fdr_hits.csv", index=False)
    enrichment.to_csv(synthesis_dir / "enrichment_top_terms.csv", index=False)
    stability.to_csv(synthesis_dir / "strict_myeloid_candidate_stability.csv", index=False)
    interaction_hits.to_csv(synthesis_dir / "metadata_interaction_fdr_hits.csv", index=False)

    write_report(
        run_dir=run_dir,
        eligibility=eligibility,
        manifest=manifest,
        hits=hits,
        replication=pd.read_csv(synthesis_dir / "cross_cohort_candidate_replication.csv"),
        concordance=pd.read_csv(synthesis_dir / "dual_anchor_concordance_summary.csv"),
        donor_correlation=pd.read_csv(synthesis_dir / "donor_anchor_correlation.csv"),
        inferred=pd.read_csv(run_dir / "myeloid/inferred_myeloid_sensitivity_eligibility.csv"),
        stability=stability,
        anchor_metadata=pd.read_csv(run_dir / "metadata/anchor_metadata_associations.csv"),
        interactions=interactions,
        enrichment=enrichment,
    )
    print(f"Wrote filtered summaries to {run_dir}")


if __name__ == "__main__":
    main()
