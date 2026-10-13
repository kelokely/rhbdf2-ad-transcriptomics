#!/usr/bin/env python3
"""Run expanded human inflammatory modules and animal-aware mouse comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.tnf_cross_species_utils import (
    aggregate_backed_population,
    compare_species,
    exact_two_group_test,
    fit_module_associations,
    module_scores,
    prepare_gse148405,
    read_one_to_one_orthologs,
)


ROOT = Path(__file__).resolve().parent


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tnf_cross_species.yaml")
    parser.add_argument("--stage", choices=["all", "human", "mouse", "compare"], default="all")
    parser.add_argument("--download", action="store_true", help="Fetch missing public mouse counts, metadata, and MGI orthology.")
    parser.add_argument("--overwrite", action="store_true", help="Replace this extension's output directory.")
    return parser.parse_args()


def fetch_if_missing(url: str, path: Path) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(path)


def source_manifest(paths: list[Path], output: Path) -> None:
    rows = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    pd.DataFrame(rows).to_csv(output / "mouse_source_manifest.csv", index=False)


def save_population(
    cohort: str,
    population: str,
    matrix_path: Path,
    cell_column: str,
    cell_value: str,
    donor_column: str,
    condition_column: str,
    modules: dict[str, list[str]],
    anchors: list[str],
    direct_genes: list[str],
    output: Path,
    config: dict,
    region_column: str | None = None,
    region_value: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    genes = sorted(set().union(*map(set, modules.values())) | set(anchors) | set(direct_genes))
    metadata, log_cpm = aggregate_backed_population(
        path=matrix_path,
        cell_type_column=cell_column,
        cell_type_value=cell_value,
        donor_column=donor_column,
        condition_column=condition_column,
        genes=genes,
        region_column=region_column,
        region_value=region_value,
    )
    output.mkdir(parents=True, exist_ok=True)
    keep = metadata["n_cells"].ge(int(config["minimum_cells_per_sample"]))
    metadata = metadata.loc[keep].reset_index(drop=True)
    log_cpm = log_cpm.loc[keep].reset_index(drop=True)
    metadata.to_csv(output / "sample_metadata.csv", index=False)
    log_cpm.to_csv(output / "sample_panel_log1p_cpm.csv", index=False)
    associations = []
    gene_associations = []
    gene_coverage = []
    coverages = []
    score_tables = []
    for anchor in anchors:
        direct_present = [g for g in direct_genes if g in log_cpm and g != anchor]
        direct_eligible = [g for g in direct_present if (log_cpm[g] > 0).mean() >= float(config["minimum_donor_detection_fraction"])
                           and log_cpm[g].std(ddof=1) > 0]
        for gene in direct_genes:
            gene_coverage.append({"cohort": cohort, "population": population, "anchor": anchor, "gene": gene,
                                  "status": "anchor_excluded" if gene == anchor else
                                  "eligible" if gene in direct_eligible else
                                  "not_in_matrix" if gene not in log_cpm else "insufficient_donor_detection_or_variance"})
        if direct_eligible:
            values = log_cpm[direct_eligible]
            standardized = (values - values.mean()) / values.std(ddof=1)
            direct_result = fit_module_associations(metadata, standardized, anchor, cohort, population,
                                                    int(config["minimum_samples_per_analysis"]))
            if not direct_result.empty:
                gene_associations.append(direct_result.rename(columns={"module": "gene"}))
        scores, coverage = module_scores(
            log_cpm, modules, anchor,
            int(config["minimum_genes_per_module"]),
            float(config["minimum_module_fraction"]),
            float(config["minimum_donor_detection_fraction"]),
        )
        coverage.insert(0, "cohort", cohort)
        coverage.insert(1, "population", population)
        coverages.append(coverage)
        if not scores.empty:
            association = fit_module_associations(metadata, scores, anchor, cohort, population,
                                                  int(config["minimum_samples_per_analysis"]))
            if not association.empty:
                association = association.merge(coverage[["module", "n_scored", "scored_genes"]], on="module", how="left")
                associations.append(association)
            score_table = pd.concat([metadata[["sample_id", "condition"]], scores], axis=1)
            score_table.insert(2, "anchor", anchor)
            score_tables.append(score_table)
    association_frame = pd.concat(associations, ignore_index=True) if associations else pd.DataFrame()
    coverage_frame = pd.concat(coverages, ignore_index=True)
    score_frame = pd.concat(score_tables, ignore_index=True) if score_tables else pd.DataFrame()
    association_frame.to_csv(output / "module_associations.csv", index=False)
    direct_frame = pd.concat(gene_associations, ignore_index=True) if gene_associations else pd.DataFrame()
    direct_frame.to_csv(output / "direct_gene_associations.csv", index=False)
    pd.DataFrame(gene_coverage).to_csv(output / "direct_gene_coverage.csv", index=False)
    coverage_frame.to_csv(output / "module_coverage.csv", index=False)
    score_frame.to_csv(output / "sample_module_scores.csv", index=False)
    return association_frame, coverage_frame, score_frame, metadata, direct_frame


def make_figure(human: pd.DataFrame, mouse: pd.DataFrame, output: Path) -> None:
    if human.empty and mouse.empty:
        return
    combined = pd.concat([human, mouse], ignore_index=True)
    combined = combined[combined["population"].isin(["Microglia", "Microglia cortex"])]
    combined = combined[combined["anchor"].isin(["RHBDF2", "Rhbdf2"])]
    if combined.empty:
        return
    combined = combined.copy()
    combined["context"] = combined["cohort"].astype(str) + "\n" + combined["population"].astype(str)
    modules = sorted(combined["module"].unique())
    contexts = list(dict.fromkeys(combined["context"]))
    effect = combined.pivot(index="module", columns="context", values="effect").reindex(index=modules, columns=contexts)
    qval = combined.pivot(index="module", columns="context", values="qval_within_cohort_population_anchor").reindex(index=modules, columns=contexts)
    fig, ax = plt.subplots(figsize=(max(8, len(contexts) * 1.6), max(5, len(modules) * 0.65)))
    vals = effect.to_numpy(dtype=float)
    limit = max(0.3, float(np.nanquantile(np.abs(vals), 0.95)))
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for y in range(len(modules)):
        for x in range(len(contexts)):
            if pd.notna(qval.iloc[y, x]) and qval.iloc[y, x] < 0.05:
                ax.text(x, y, "*", ha="center", va="center", color="black", fontsize=15)
    ax.set_xticks(range(len(contexts)), contexts, rotation=25, ha="right")
    ax.set_yticks(range(len(modules)), [m.replace("_", " ") for m in modules])
    ax.set_title("RHBDF2/Rhbdf2-linked RNA module associations by biological replicate")
    fig.colorbar(im, ax=ax, label="Donor- or mouse-level association coefficient")
    fig.text(0.02, 0.01, "* within-cohort FDR < 0.05. RNA scores are not measured TNF secretion or protein activity.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output / "human_mouse_inflammatory_modules.png", dpi=350)
    fig.savefig(output / "human_mouse_inflammatory_modules.pdf")
    plt.close(fig)


def write_report(config: dict, human: pd.DataFrame, mouse: pd.DataFrame, contrast: pd.DataFrame,
                 comparison: pd.DataFrame, coverage: pd.DataFrame, human_direct: pd.DataFrame,
                 mouse_direct: pd.DataFrame, direct_comparison: pd.DataFrame, output: Path) -> None:
    def table(frame: pd.DataFrame, fields: list[str]) -> list[str]:
        if frame.empty:
            return ["No eligible rows."]
        rows = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
        for _, row in frame.iterrows():
            values = []
            for field in fields:
                value = row[field]
                if isinstance(value, (float, np.floating)):
                    values.append(f"{value:.3g}" if pd.notna(value) else "NA")
                else:
                    values.append(str(value))
            rows.append("| " + " | ".join(values) + " |")
        return rows

    microglia = human[human["population"].eq("Microglia")]
    summaries = []
    for anchor in ["RHBDF2", "ADAM17"]:
        subset = microglia[microglia["anchor"].eq(anchor)]
        for module, group in subset.groupby("module"):
            positive = group[(group["effect"] > 0) & group["qval_within_cohort_population_anchor"].lt(0.05)]
            negative = group[(group["effect"] < 0) & group["qval_within_cohort_population_anchor"].lt(0.05)]
            summaries.append({"anchor": anchor, "module": module, "positive_FDR_cohorts": len(positive),
                              "negative_FDR_cohorts": len(negative), "eligible_cohorts": len(group),
                              "positive_cohort_names": ", ".join(positive["cohort"]),
                              "negative_cohort_names": ", ".join(negative["cohort"])})
    summary = pd.DataFrame(summaries).sort_values(["anchor", "positive_FDR_cohorts", "module"], ascending=[True, False, True])
    summary.to_csv(output / "human_microglia_module_replication.csv", index=False)
    direct_summaries = []
    for anchor in ["RHBDF2", "ADAM17"]:
        subset = human_direct[human_direct["population"].eq("Microglia") & human_direct["anchor"].eq(anchor)]
        for gene, group in subset.groupby("gene"):
            positive = group[(group["effect"] > 0) & group["qval_within_cohort_population_anchor"].lt(0.05)]
            negative = group[(group["effect"] < 0) & group["qval_within_cohort_population_anchor"].lt(0.05)]
            direct_summaries.append({"anchor": anchor, "gene": gene, "positive_FDR_cohorts": len(positive),
                                     "negative_FDR_cohorts": len(negative), "eligible_cohorts": len(group),
                                     "positive_cohort_names": ", ".join(positive["cohort"]),
                                     "negative_cohort_names": ", ".join(negative["cohort"])})
    direct_summary = pd.DataFrame(direct_summaries).sort_values(
        ["anchor", "positive_FDR_cohorts", "gene"], ascending=[True, False, True]
    )
    direct_summary.to_csv(output / "human_microglia_direct_gene_replication.csv", index=False)
    rhbdf2_tnfaip3 = direct_summary[direct_summary["anchor"].eq("RHBDF2") & direct_summary["gene"].eq("TNFAIP3")]
    adam17_il6r = direct_summary[direct_summary["anchor"].eq("ADAM17") & direct_summary["gene"].eq("IL6R")]
    human_sig = human[human["qval_within_cohort_population_anchor"].lt(0.05)].copy()
    human_sig = human_sig.sort_values(["cohort", "population", "anchor", "qval_within_cohort_population_anchor"])
    direct_focus = human_direct[human_direct["gene"].isin(["TNF", "TNFRSF1A", "TNFRSF1B", "IL1B", "IL6", "IL6R"])]
    direct_focus = direct_focus[direct_focus["population"].isin(["Microglia", "Microglia mapped immune"])]
    direct_focus = direct_focus.sort_values(["cohort", "anchor", "gene"])
    mouse_direct_focus = mouse_direct[mouse_direct["gene"].isin(["Tnf", "Tnfrsf1a", "Tnfrsf1b", "Il1b", "Il6", "Il6ra"])]
    human_tnf = direct_focus[direct_focus["anchor"].eq("RHBDF2") & direct_focus["gene"].eq("TNF")]
    human_tnf_supported = human_tnf[human_tnf["qval_within_cohort_population_anchor"].lt(0.05)]
    mouse_tnf = mouse_direct_focus[mouse_direct_focus["anchor"].eq("Rhbdf2") & mouse_direct_focus["gene"].eq("Tnf")]
    mouse_tnf_q = float(mouse_tnf.iloc[0]["qval_within_cohort_population_anchor"]) if not mouse_tnf.empty else np.nan
    mouse_samples = pd.read_csv(output / "mouse" / "GSE148405" / "Microglia_cortex" / "sample_metadata.csv")
    secondary = config["secondary_mouse_candidate"]
    secondary_status = "downloaded locally but not analyzed" if (ROOT / secondary["source_path"]).is_file() else "identified but not downloaded locally"
    text = [
        "# TNF and Inflammatory RNA Module Extension",
        "",
        "## Study Boundary",
        "The human and mouse analyses were run independently. A module score is a transcript-level proxy, not TNF release, receptor activation, NF-kappaB nuclear translocation, or ADAM17 cleavage. The human unit is the donor; the mouse unit is the animal. Mouse brain regions from one animal are never counted as independent animals. This is a new extension, not a change to the earlier manuscript findings.",
        "",
        "## Methods",
        "Eight investigator-defined inflammatory/context modules were frozen in the effective YAML. They separate ligand/receptor context, canonical TNF/NF-kappaB response, TNF feedback/survival, IL-1/inflammasome, IL-6/JAK-STAT, interferon, chemokine, and phagolysosomal transcriptional patterns. These overlap biologically and are not independent pathway measurements; TNF-feedback genes can respond to stimuli other than TNF. A separate direct-gene analysis tests configured inflammatory transcripts, with its own within-cohort/anchor FDR family. RHBDF2 or ADAM17 was removed from a module or direct-gene test when used as its anchor. For each eligible population, raw counts were summed within biological sample. Gene CPM used the original full-library denominator, including when an extracted targeted matrix was used. Log1p CPM for each scored gene was standardized across samples within a cohort/population, then averaged for modules or used individually for direct-gene tests. A module required at least four expressed genes and coverage of at least half its requested genes. Associations were fitted by OLS against log1p anchor-detection rate per 1,000 cells, adjusting for full-library depth per cell and cohort-native pathology or mouse genotype when residual degrees of freedom permitted. The association table records which adjustments were actually used, plus leave-one-sample-out effect-sign concordance. Benjamini-Hochberg correction was applied across eligible modules within each cohort, population, and anchor, not globally across all analyses. The mouse genotype contrast used an exact animal-label permutation test; it is a different estimand from the anchor association.",
        "",
        "Mouse cells are FACS-isolated microglia from 30-week-old WT and 5xFAD mice in GSE148405. Per-cell GEO titles supply genotype, region, and replicate; replicate numbers identify the same animals across regions. The primary analysis uses cortex only, four animals per genotype. The source count CSV was streamed to a panel AnnData object without changing the full-library denominator. Human-to-mouse genes were mapped through the MGI protein-coding one-to-one orthology report. Human input H5AD matrices were checked to contain integer counts. GSE157827's author-labeled endothelial group contains other vascular cell types; endothelial results therefore represent a broader vascular-labeled compartment and are not purified endothelium. A microglial module scored in vascular-labeled cells must not be read as proof of intrinsic endothelial activity. SEA-AD uses a 61-gene targeted working matrix, so most modules are ineligible there and cannot be treated as negative results.",
        "",
        "## Human Results",
        f"Eligible association rows: {len(human)}; within-cohort FDR < 0.05: {int((human.get('qval_within_cohort_population_anchor', pd.Series(dtype=float)) < 0.05).sum())}.",
        "See `human_module_associations.csv` for all coefficients, P values, FDR values, and sample counts. `module_coverage.csv` distinguishes unsupported gene sets from negative association results.",
        "",
        "### Microglia replication counts",
        *table(summary, ["anchor", "module", "positive_FDR_cohorts", "negative_FDR_cohorts", "eligible_cohorts"]),
        "",
        "### Direct transcript replication across full human microglia cohorts",
        *table(direct_summary, ["anchor", "gene", "positive_FDR_cohorts", "negative_FDR_cohorts", "eligible_cohorts"]),
        "",
        f"RHBDF2-TNFAIP3 positive RNA association: {int(rhbdf2_tnfaip3.iloc[0]['positive_FDR_cohorts']) if not rhbdf2_tnfaip3.empty else 0} full microglia cohorts. ADAM17-IL6R positive RNA association: {int(adam17_il6r.iloc[0]['positive_FDR_cohorts']) if not adam17_il6r.empty else 0} full microglia cohorts. Neither result measures TNF induction, receptor cleavage, or ADAM17 enzymatic activity.",
        "TNFAIP3 is a negative-feedback regulator associated with several inflammatory stimuli; its replicated RNA association does not establish TNF-specific activation or causal signaling from iRhom2.",
        "",
        "### FDR-supported human association rows",
        *table(human_sig, ["cohort", "population", "anchor", "module", "effect", "qval_within_cohort_population_anchor", "loo_sign_concordance"]),
        "",
        "### Direct TNF, receptor, and cytokine transcript checks in microglia",
        *table(direct_focus, ["cohort", "population", "anchor", "gene", "effect", "qval_within_cohort_population_anchor"]),
        "",
        f"For RHBDF2 versus TNF RNA specifically, FDR-supported positive cohorts are: {', '.join(human_tnf_supported['cohort']) if not human_tnf_supported.empty else 'none'}. SEA-AD is a targeted 61-gene analysis, so its association is supportive but not equivalent to full-transcriptome validation.",
        "A direct-gene association is donor-level co-variation of RNA, not evidence of soluble TNF release or receptor activation. Missing rows must be checked against `direct_gene_coverage.csv` before interpreting them as null.",
        "",
        "## Mouse Results",
        f"Cortical microglia association rows: {len(mouse)}; within-analysis FDR < 0.05: {int((mouse.get('qval_within_cohort_population_anchor', pd.Series(dtype=float)) < 0.05).sum())}.",
        f"Exact 5xFAD-versus-WT module contrasts: {len(contrast)}; FDR < 0.05: {int((contrast.get('qval_exact_bh', pd.Series(dtype=float)) < 0.05).sum())}.",
        f"The Rhbdf2 transcript was detected in {int(mouse_samples['Rhbdf2_detected_cells'].min())}-{int(mouse_samples['Rhbdf2_detected_cells'].max())} cortical microglia per animal. The direct Rhbdf2-Tnf RNA association has within-mouse-analysis FDR q={mouse_tnf_q:.3g}.",
        "The 30-week GSE148405 dataset has only four mice per genotype in the primary cortex comparison. With 70 possible label allocations, the smallest possible two-sided exact P value is 2/70 = 0.0286, making FDR-supported detection of multiple modules difficult. Non-significance cannot exclude a smaller effect, and positive nominal associations require independent validation. One WT mouse has much greater sequencing depth than the others; the anchor model adjusts for full-library depth, and the leave-one-out column reports directional sensitivity.",
        "",
        "### Mouse Rhbdf2 association and genotype contrast",
        *table(mouse[mouse["anchor"].eq("Rhbdf2")].merge(contrast, on="module", how="left"),
               ["module", "effect", "qval_within_cohort_population_anchor", "loo_sign_concordance", "disease_minus_control", "pval_exact", "qval_exact_bh"]),
        "",
        "### Direct mouse inflammatory transcript checks",
        *table(mouse_direct_focus, ["anchor", "gene", "effect", "qval_within_cohort_population_anchor"]),
        "",
        "## Cross-Species Comparison",
        f"Comparable anchor-module rows: {len(comparison)}; strict cross-species FDR-supported rows: {int(comparison['cross_species_fdr_support'].sum()) if not comparison.empty else 0}.",
        f"Comparable direct-transcript rows: {len(direct_comparison)}; strict cross-species FDR-supported rows: {int(direct_comparison['cross_species_fdr_support'].sum()) if not direct_comparison.empty else 0}.",
        "The comparison uses mapped one-to-one protein-coding orthologs and evaluates within-species effect directions, not absolute human-versus-mouse RNA levels. A 5xFAD amyloid model is not equivalent to sporadic human AD. Disease-genotype contrasts must not be conflated with within-species Rhbdf2 association. No strict cross-species result should be inferred from same-sign coefficients alone.",
        "Age, sex, region, cell-state composition, and other cohort-specific factors were not uniformly modeled in this extension. Pathology/genotype and library-depth adjustment reduce some confounding but do not make the RNA associations causal or prove a TNF-ADAM17-iRhom2 mechanism.",
        "",
        "## Second Mouse Dataset Readiness",
        f"GSE140510 (7-month mouse snRNA-seq; three individually processed WT and three WT-5xFAD animals, plus Trem2 genotypes) is {secondary_status}. Its public raw 10x matrices do not carry ready-to-use cell-type labels. It was not used for the result tables here; an auditable microglia annotation is required before animal-aware pseudobulk comparison.",
        "",
        "## Source and Reproducibility",
        "Mouse counts and per-cell genotype/region/replicate labels: [GSE148405](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE148405). Second mouse source: [GSE140510](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140510). Human-mouse one-to-one orthology: [MGI report](https://www.informatics.jax.org/downloads/reports/index.html). Module design was informed by [MSigDB TNF-via-NF-kappaB](https://www.gsea-msigdb.org/gsea/msigdb/cards/HALLMARK_TNFA_SIGNALING_VIA_NFKB.html) and [Reactome TNFR1 signaling](https://reactome.org/content/detail/R-HSA-5357956); the configured panels are investigator-defined subsets, not exact exports of those resources. `mouse_source_manifest.csv` records source-file sizes and SHA-256 hashes.",
        "",
        "## Files",
        "- `human_module_associations.csv`: human donor-level tests.",
        "- `human_direct_gene_associations.csv` and `mouse_direct_gene_associations.csv`: direct transcript tests.",
        "- `mouse_module_associations.csv`: mouse-level Rhbdf2/Adam17 tests.",
        "- `mouse_genotype_module_contrasts.csv`: exact 5xFAD-WT tests.",
        "- `cross_species_module_comparison.csv`: matched-association comparison.",
        "- `cross_species_direct_gene_comparison.csv`: mapped direct-transcript comparison.",
        "- `module_coverage.csv`: detected genes and eligibility for every context.",
        "- `human_microglia_module_replication.csv`: counts of FDR-supported human microglia cohorts per module.",
        "- `human_microglia_direct_gene_replication.csv`: counts of FDR-supported human microglia cohorts per direct transcript.",
        "- `mouse_source_manifest.csv`: frozen source checksums.",
        "- `human_mouse_inflammatory_modules.png` and `.pdf`: descriptive effect map.",
        "- `effective_config.yaml`: frozen analysis settings.",
    ]
    output.joinpath("METHODS_AND_DATA_RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    a = args()
    config_path = (ROOT / a.config).resolve()
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    output = ROOT / config["output_dir"]
    if a.overwrite and output.exists():
        import shutil
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "effective_config.yaml"
    if snapshot.exists() and snapshot.read_text(encoding="utf-8") != config_text:
        raise ValueError("The existing results were produced with a different config; choose a new output_dir or use --overwrite.")
    snapshot.write_text(config_text, encoding="utf-8")

    mouse_spec = config["mouse"]
    ortholog_spec = config["orthology"]
    for url, path in [
        (mouse_spec["matrix_url"], ROOT / mouse_spec["matrix_path"]),
        (mouse_spec["metadata_url"], ROOT / mouse_spec["metadata_path"]),
        (ortholog_spec["source_url"], ROOT / ortholog_spec["path"]),
    ]:
        if a.download:
            fetch_if_missing(url, path)
    if a.stage in {"all", "mouse"}:
        for field in ["matrix_path", "metadata_path"]:
            if not (ROOT / mouse_spec[field]).is_file():
                raise FileNotFoundError(f"Missing {field}; rerun with --download.")
        if not (ROOT / ortholog_spec["path"]).is_file():
            raise FileNotFoundError("Missing MGI orthology report; rerun with --download.")
        source_manifest([ROOT / mouse_spec["matrix_path"], ROOT / mouse_spec["metadata_path"],
                         ROOT / ortholog_spec["path"]], output)
        prepared = ROOT / mouse_spec["prepared_path"]
        prepared_summary = prepared.with_suffix(".preparation.json")
        requested = sorted(set().union(*map(set, config["modules"].values())) |
                           set(config["direct_gene_checks"]) | {"RHBDF2", "ADAM17"})
        existing_summary = json.loads(prepared_summary.read_text(encoding="utf-8")) if prepared_summary.is_file() else {}
        if not prepared.is_file() or existing_summary.get("panel_human_genes_requested") != requested:
            print("Preparing GSE148405 count and animal metadata object", flush=True)
            prepare_gse148405(ROOT / mouse_spec["matrix_path"], ROOT / mouse_spec["metadata_path"],
                              ROOT / ortholog_spec["path"], config["modules"], prepared,
                              config["direct_gene_checks"])

    human_results = []
    human_gene_results = []
    all_coverage = []
    if a.stage in {"all", "human"}:
        for cohort, spec in config["human_cohorts"].items():
            for population, label in spec["populations"].items():
                print(f"Human: {cohort} / {population}", flush=True)
                associations, coverage, _, metadata, direct = save_population(
                    cohort, population, ROOT / spec["path"], spec["cell_type_column"], label,
                    spec["donor_column"], spec["pathology_column"], config["modules"], ["RHBDF2", "ADAM17"],
                    config["direct_gene_checks"],
                    output / "human" / cohort / population.replace(" ", "_"), config,
                )
                human_results.append(associations)
                human_gene_results.append(direct)
                all_coverage.append(coverage)
                print(f"  {len(metadata)} donors; {len(associations)} eligible module tests", flush=True)
        pd.concat(human_results, ignore_index=True).to_csv(output / "human_module_associations.csv", index=False)
        pd.concat(human_gene_results, ignore_index=True).to_csv(output / "human_direct_gene_associations.csv", index=False)

    if a.stage in {"all", "mouse"}:
        orthologs = read_one_to_one_orthologs(ROOT / ortholog_spec["path"])
        human_to_mouse = dict(zip(orthologs["human_gene"], orthologs["mouse_gene"]))
        mouse_modules = {name: [human_to_mouse[g] for g in genes if g in human_to_mouse]
                         for name, genes in config["modules"].items()}
        mouse_direct_genes = [human_to_mouse[g] for g in config["direct_gene_checks"] if g in human_to_mouse]
        (output / "mouse_module_orthologs.json").write_text(json.dumps(mouse_modules, indent=2) + "\n", encoding="utf-8")
        population = "Microglia cortex"
        print("Mouse: GSE148405 / cortical microglia", flush=True)
        mouse_result, coverage, scores, metadata, mouse_direct = save_population(
            "GSE148405", population, ROOT / mouse_spec["prepared_path"], "cell_type", "Microglia",
            "mouse_id", "genotype", mouse_modules, ["Rhbdf2", "Adam17"],
            mouse_direct_genes,
            output / "mouse" / "GSE148405" / "Microglia_cortex", config,
            region_column="region", region_value=mouse_spec["primary_region"],
        )
        all_coverage.append(coverage)
        mouse_result.to_csv(output / "mouse_module_associations.csv", index=False)
        mouse_direct.to_csv(output / "mouse_direct_gene_associations.csv", index=False)
        if not scores.empty:
            selected = scores[scores["anchor"].eq("Rhbdf2")].drop(columns=["anchor"])
            contrast = exact_two_group_test(metadata, selected.drop(columns=["sample_id", "condition"]),
                                            mouse_spec["genotype_control"], mouse_spec["genotype_disease"])
        else:
            contrast = pd.DataFrame()
        contrast.to_csv(output / "mouse_genotype_module_contrasts.csv", index=False)
        print(f"  {len(metadata)} mice; {len(mouse_result)} eligible module tests", flush=True)

    if a.stage in {"all", "compare"}:
        human = pd.read_csv(output / "human_module_associations.csv")
        mouse = pd.read_csv(output / "mouse_module_associations.csv")
        contrast = pd.read_csv(output / "mouse_genotype_module_contrasts.csv")
        human_direct = pd.read_csv(output / "human_direct_gene_associations.csv")
        mouse_direct = pd.read_csv(output / "mouse_direct_gene_associations.csv")
        comparison = compare_species(human, mouse)
        comparison.to_csv(output / "cross_species_module_comparison.csv", index=False)
        orthologs = read_one_to_one_orthologs(ROOT / ortholog_spec["path"])
        mouse_to_human = dict(zip(orthologs["mouse_gene"], orthologs["human_gene"]))
        direct_mouse_mapped = mouse_direct.copy()
        direct_mouse_mapped["module"] = direct_mouse_mapped["gene"].map(mouse_to_human)
        direct_human_mapped = human_direct.rename(columns={"gene": "module"})
        direct_comparison = compare_species(direct_human_mapped, direct_mouse_mapped)
        if not direct_comparison.empty:
            direct_comparison = direct_comparison.rename(columns={"module": "gene"})
        direct_comparison.to_csv(output / "cross_species_direct_gene_comparison.csv", index=False)
        coverage_paths = sorted(output.glob("**/module_coverage.csv"))
        coverage = pd.concat([pd.read_csv(p) for p in coverage_paths], ignore_index=True)
        coverage.to_csv(output / "module_coverage.csv", index=False)
        make_figure(human, mouse, output)
        write_report(config, human, mouse, contrast, comparison, coverage, human_direct, mouse_direct,
                     direct_comparison, output)
        print(f"Comparison: {len(comparison)} rows, {int(comparison['cross_species_fdr_support'].sum()) if not comparison.empty else 0} strict supported", flush=True)


if __name__ == "__main__":
    main()
