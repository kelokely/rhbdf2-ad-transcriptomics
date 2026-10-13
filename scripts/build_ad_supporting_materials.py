#!/usr/bin/env python3
"""Build focused-microglia figures and supporting tables for the AD manuscript."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = Path(os.getenv("RHBDF2_FOCUSED_FIGURE_DIR", str(ROOT / "report/focused_microglia")))
FIGURE_DIR = MANUSCRIPT / "figures"
TABLE_DIR = MANUSCRIPT / "supporting_tables"
EXTENSION_TABLES = ROOT / "report" / "independent_ad_extension" / "tables"
ARCHIVE_RESULTS = ROOT / "Archives" / "results"

INK = "#19333a"
MUTED = "#6f7f83"
TEAL = "#198a8a"
BLUE = "#39739d"
RUST = "#c65f3d"
GOLD = "#dda51e"
PALE = "#eef2f1"

CANDIDATES = [
    "RHBDF2",
    "ADAM17",
    "TREM2",
    "TYROBP",
    "SYK",
    "TMEM173",
    "TBK1",
    "IRF3",
    "RELA",
    "TNF",
    "IL6R",
    "ITGAX",
    "CLEC7A",
    "CST7",
    "LPL",
]


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def dense_vector(adata, gene: str) -> np.ndarray:
    values = adata[:, gene].X
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values).ravel()


def load_normalized(path: Path):
    adata = sc.read_h5ad(path)
    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)
    return adata


def group_summary(adata, group_col: str, order: list[str]) -> pd.DataFrame:
    rows = []
    for group in order:
        mask = adata.obs[group_col].astype(str).to_numpy() == group
        for gene in CANDIDATES:
            if gene not in adata.var_names or not mask.any():
                continue
            values = dense_vector(adata, gene)[mask]
            rows.append(
                {
                    "group": group,
                    "gene": gene,
                    "mean": float(values.mean()),
                    "fraction": float((values > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def draw_dotplot(
    ax: plt.Axes,
    summary: pd.DataFrame,
    groups: list[str],
    group_labels: list[str],
    title: str,
    color_norm: Normalize,
) -> None:
    gene_pos = {gene: i for i, gene in enumerate(CANDIDATES)}
    group_pos = {group: i for i, group in enumerate(groups)}
    x = summary["gene"].map(gene_pos)
    y = summary["group"].map(group_pos)
    sizes = 12 + summary["fraction"] * 560
    ax.scatter(
        x,
        y,
        s=sizes,
        c=summary["mean"],
        cmap="YlOrRd",
        norm=color_norm,
        edgecolor="#5f6262",
        linewidth=0.35,
    )
    ax.set_xticks(range(len(CANDIDATES)), CANDIDATES, rotation=55, ha="right")
    ax.set_yticks(range(len(groups)), group_labels)
    ax.set_xlim(-0.65, len(CANDIDATES) - 0.35)
    ax.set_ylim(len(groups) - 0.5, -0.5)
    ax.set_title(title, loc="left", color=INK, fontweight="bold")
    ax.grid(axis="x", color="#e7e9e8", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#879496")


def add_dotplot_legends(fig: plt.Figure, color_norm: Normalize) -> None:
    cax = fig.add_axes([0.84, 0.13, 0.018, 0.18])
    sm = plt.cm.ScalarMappable(norm=color_norm, cmap="YlOrRd")
    colorbar = fig.colorbar(sm, cax=cax)
    colorbar.set_label("Mean log1p expression", fontsize=9)
    lax = fig.add_axes([0.865, 0.13, 0.12, 0.18])
    lax.axis("off")
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    for i, fraction in enumerate((0.1, 0.3, 0.5, 0.8)):
        lax.scatter(0.18, 0.82 - i * 0.22, s=12 + fraction * 560, color="#9ba7a8")
        lax.text(0.45, 0.82 - i * 0.22, f"{fraction:.0%}", va="center", fontsize=8)
    lax.text(0.02, 1.02, "Detected cells", fontsize=9, fontweight="bold", color=INK)


def build_focused_figures() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
        }
    )

    gse243292 = load_normalized(ROOT / "data" / "raw" / "GSE243292_Microglia_GEO.h5ad")
    gse146639 = load_normalized(
        ROOT / "data" / "raw" / "GSE146639_barcoded_smartseq2_microglia.h5ad"
    )
    gse243292.obs["RHBDF2_state"] = np.where(
        dense_vector(gse243292, "RHBDF2") > 0, "Detected", "Undetected"
    )
    gse146639.obs["RHBDF2_state"] = np.where(
        dense_vector(gse146639, "RHBDF2") > 0, "Detected", "Undetected"
    )

    sub_order = [
        "Homeostatic",
        "Motile",
        "ARM",
        "Dystrophic",
        "Oli_microglia",
        "Ast_microglia",
        "Ex_microglia",
    ]
    sub_labels = [
        "Homeostatic",
        "Motile",
        "ARM",
        "Dystrophic",
        "Oli-like",
        "Ast-like",
        "Ex-like",
    ]
    at_order = ["A+T+", "A+T-", "A-T-"]

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.5), constrained_layout=True)
    plot_data = pd.DataFrame(
        {
            "RHBDF2": dense_vector(gse243292, "RHBDF2"),
            "Subpopulation": gse243292.obs["microglia_subpopulations"].astype(str).to_numpy(),
            "AT status": gse243292.obs["atscore"].astype(str).to_numpy(),
        }
    )
    sns.violinplot(
        data=plot_data,
        x="Subpopulation",
        y="RHBDF2",
        order=sub_order,
        palette=sns.color_palette("Set2", len(sub_order)),
        cut=0,
        inner="quartile",
        linewidth=0.8,
        ax=axes[0],
    )
    axes[0].set_xticks(range(len(sub_labels)), sub_labels)
    axes[0].set_title("A  RHBDF2 expression across microglial subpopulations", loc="left", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Normalized log1p expression")
    for idx, group in enumerate(sub_order):
        mask = plot_data["Subpopulation"] == group
        detected = (plot_data.loc[mask, "RHBDF2"] > 0).mean() * 100
        axes[0].text(idx, axes[0].get_ylim()[1] * 0.96, f"{detected:.1f}%", ha="center", fontsize=8)

    sns.violinplot(
        data=plot_data,
        x="AT status",
        y="RHBDF2",
        order=at_order,
        palette=[TEAL, GOLD, BLUE],
        cut=0,
        inner="quartile",
        linewidth=0.8,
        ax=axes[1],
    )
    axes[1].set_title("B  RHBDF2 expression across amyloid/tau categories", loc="left", fontweight="bold")
    axes[1].set_xlabel("Amyloid/tau category")
    axes[1].set_ylabel("Normalized log1p expression")
    for idx, group in enumerate(at_order):
        mask = plot_data["AT status"] == group
        detected = (plot_data.loc[mask, "RHBDF2"] > 0).mean() * 100
        axes[1].text(idx, axes[1].get_ylim()[1] * 0.96, f"{detected:.1f}%", ha="center", fontsize=8)
    fig.suptitle("Focused GSE243292 microglial RHBDF2 expression context", fontsize=15, fontweight="bold")
    save_figure(fig, "Figure_S5_GSE243292_rhbdf2_context")

    sub_summary = group_summary(gse243292, "microglia_subpopulations", sub_order)
    state_summary = group_summary(gse243292, "RHBDF2_state", ["Detected", "Undetected"])
    color_max = max(sub_summary["mean"].quantile(0.98), state_summary["mean"].quantile(0.98))
    norm = Normalize(0, color_max)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.3), constrained_layout=False)
    fig.subplots_adjust(left=0.13, right=0.81, top=0.91, bottom=0.10, hspace=0.56)
    draw_dotplot(axes[0], sub_summary, sub_order, sub_labels, "A  Candidate panel by microglial subpopulation", norm)
    draw_dotplot(axes[1], state_summary, ["Detected", "Undetected"], ["RHBDF2 detected", "RHBDF2 undetected"], "B  Candidate panel by operational RHBDF2 state", norm)
    add_dotplot_legends(fig, norm)
    fig.suptitle("GSE243292 candidate-panel expression", fontsize=15, fontweight="bold")
    save_figure(fig, "Figure_S6_GSE243292_candidate_expression")

    make_correlation_figure(
        ARCHIVE_RESULTS / "GSE243292_run" / "tables" / "candidate_gene_correlations.csv",
        "GSE243292 non-self candidate correlations with RHBDF2",
        "Figure_S7_GSE243292_candidate_correlations",
    )

    diagnosis_order = ["CTR", "AD"]
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    diagnosis_data = pd.DataFrame(
        {
            "RHBDF2": dense_vector(gse146639, "RHBDF2"),
            "Diagnosis": gse146639.obs["diagnosis"].astype(str).to_numpy(),
        }
    )
    sns.violinplot(
        data=diagnosis_data,
        x="Diagnosis",
        y="RHBDF2",
        order=diagnosis_order,
        palette=[BLUE, RUST],
        cut=0,
        inner="quartile",
        linewidth=0.9,
        ax=ax,
    )
    for idx, group in enumerate(diagnosis_order):
        values = diagnosis_data.loc[diagnosis_data["Diagnosis"] == group, "RHBDF2"]
        ax.text(idx, ax.get_ylim()[1] * 0.95, f"Detected: {(values > 0).mean():.1%}", ha="center", fontsize=9)
    ax.set_title("GSE146639 RHBDF2 expression by diagnosis", fontweight="bold")
    ax.set_ylabel("Normalized log1p expression")
    ax.set_xlabel("")
    ax.set_xticks([0, 1], ["Control", "Alzheimer disease"])
    save_figure(fig, "Figure_S8_GSE146639_rhbdf2_diagnosis")

    diagnosis_summary = group_summary(gse146639, "diagnosis", diagnosis_order)
    state_summary = group_summary(gse146639, "RHBDF2_state", ["Detected", "Undetected"])
    color_max = max(diagnosis_summary["mean"].quantile(0.98), state_summary["mean"].quantile(0.98))
    norm = Normalize(0, color_max)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.1), constrained_layout=False)
    fig.subplots_adjust(left=0.13, right=0.81, top=0.90, bottom=0.11, hspace=0.62)
    draw_dotplot(axes[0], diagnosis_summary, diagnosis_order, ["Control", "Alzheimer disease"], "A  Candidate panel by diagnosis", norm)
    draw_dotplot(axes[1], state_summary, ["Detected", "Undetected"], ["RHBDF2 detected", "RHBDF2 undetected"], "B  Candidate panel by operational RHBDF2 state", norm)
    add_dotplot_legends(fig, norm)
    fig.suptitle("GSE146639 candidate-panel expression", fontsize=15, fontweight="bold")
    save_figure(fig, "Figure_S9_GSE146639_candidate_expression")

    make_correlation_figure(
        ARCHIVE_RESULTS / "GSE146639_run" / "tables" / "candidate_gene_correlations.csv",
        "GSE146639 non-self candidate correlations with RHBDF2",
        "Figure_S10_GSE146639_candidate_correlations",
    )


def make_correlation_figure(path: Path, title: str, stem: str) -> None:
    data = pd.read_csv(path)
    gene_col = "gene" if "gene" in data.columns else "names"
    pearson_col = next(col for col in data.columns if "pearson" in col and col.endswith("r"))
    spearman_col = next(col for col in data.columns if "spearman" in col and col.endswith("r"))
    data = data[data[gene_col] != "RHBDF2"].copy()
    data = data.set_index(gene_col)[[pearson_col, spearman_col]]
    data.columns = ["Pearson r", "Spearman r"]
    data = data.loc[data["Spearman r"].abs().sort_values(ascending=False).index]
    limit = max(0.12, float(np.ceil(data.abs().to_numpy().max() * 20) / 20))
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    fig.subplots_adjust(left=0.20, right=0.86, top=0.91, bottom=0.12)
    sns.heatmap(
        data,
        cmap="vlag",
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=True,
        fmt=".3f",
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Correlation coefficient"},
        ax=ax,
    )
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.text(0.5, 0.025, "RHBDF2 self-correlation (=1.0) is omitted to preserve the non-self color scale.", ha="center", fontsize=8, color=MUTED)
    save_figure(fig, stem)


def fmt_p(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 1e-99:
        return "<1e-99"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def render_table(
    data: pd.DataFrame,
    stem: str,
    title: str,
    subtitle: str,
    wrap: dict[str, int] | None = None,
    font_size: float = 7.2,
    column_widths: list[float] | None = None,
) -> None:
    wrap = wrap or {}
    rendered = data.copy().astype(str)
    for column, width in wrap.items():
        if column in rendered.columns:
            rendered[column] = rendered[column].map(lambda value: textwrap.fill(value, width=width))
    rendered.columns = [textwrap.fill(str(column), width=18) for column in rendered.columns]

    nrows, ncols = rendered.shape
    height = max(3.0, 1.75 + nrows * 0.42)
    width = 11.0 if ncols >= 5 else 9.5
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    fig.text(0.04, 0.965, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.925, subtitle, fontsize=8.5, color=MUTED, va="top")
    table = ax.table(
        cellText=rendered.values,
        colLabels=rendered.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        bbox=[0.03, 0.02, 0.94, 0.84],
        colWidths=column_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0.8)
        cell.PAD = 0.035
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f0f3f2" if row % 2 == 0 else "#ffffff")
            cell.get_text().set_color("#25383d")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(TABLE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(TABLE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    data.to_csv(TABLE_DIR / f"{stem}.csv", index=False)


def build_supporting_tables() -> None:
    cohort = pd.DataFrame(
        [
            ["GSE243292", "DLPFC snRNA-seq", "15", "122,606; 3,982 microglia", "Focused microglia + multicell reference"],
            ["GSE146639", "Cortical microglia scRNA-seq", "14", "11,775", "Focused microglia validation"],
            ["GSE138852", "Entorhinal cortex snRNA-seq", "Pooled", "Processed multicell object", "Descriptive sensitivity only"],
            ["GSE157827", "Prefrontal cortex snRNA-seq", "21", "Prepared multicell object", "Independent donor replication"],
            ["GSE237718", "Temporal cortex snRNA-seq", "56", "461,113", "Replication + APOE sensitivity"],
            ["GSE268599", "Five-region astrocyte snRNA-seq", "32", "629,749", "Powered astrocyte replication"],
            ["SEA-AD MTG", "Middle temporal gyrus snRNA-seq", "84", "1,178,693", "Large multicell replication"],
            ["GSE269906", "BA10 Stereo-seq", "12", "298,399", "Independent spatial context"],
        ],
        columns=["Cohort", "Tissue / assay", "Donors", "Analyzed cells or nuclei", "Analytical role"],
    )
    render_table(
        cohort,
        "Table_S1_cohort_inventory",
        "Supporting Table S1. Cohort inventory and analytical roles",
        "Counts refer to analyzed objects; pooled libraries and within-cohort strata were not treated as independent donor replication.",
        wrap={"Tissue / assay": 24, "Analyzed cells or nuclei": 24, "Analytical role": 28},
        column_widths=[0.12, 0.22, 0.08, 0.23, 0.29],
    )

    selected = ["RHBDF2", "ADAM17", "TREM2", "TYROBP", "SYK", "IL6R", "ITGAX", "CLEC7A"]
    focused_rows = []
    for dataset in ("GSE243292", "GSE146639"):
        data = pd.read_csv(ARCHIVE_RESULTS / f"{dataset}_run" / "tables" / "candidate_evidence_summary.csv")
        data = data.set_index("gene").loc[selected].reset_index()
        for row in data.itertuples(index=False):
            focused_rows.append(
                {
                    "Dataset": dataset,
                    "Gene": row.gene,
                    "Cell logFC (q)": f"{row.cell_level_logfoldchange:+.3f} ({fmt_p(row.cell_level_padj)})",
                    "Continuous slope (q)": f"{row.continuous_slope_per_log1p_rhbdf2:+.3f} ({fmt_p(row.continuous_padj)})",
                    "Pseudobulk effect (q)": f"{row.pseudobulk_effect_log1p_cpm:+.3f} ({fmt_p(row.pseudobulk_padj)})",
                    "Pearson / Spearman": f"{row.candidate_pearson_r_with_RHBDF2:+.3f} / {row.candidate_spearman_r_with_RHBDF2:+.3f}",
                }
            )
    focused = pd.DataFrame(focused_rows)
    render_table(
        focused,
        "Table_S2_focused_microglia_evidence",
        "Supporting Table S2. Focused microglia candidate evidence",
        "Parentheses contain FDR-adjusted P values. Pseudobulk is the primary transcript-level inference; cell and continuous models remain supportive.",
        font_size=6.8,
        column_widths=[0.12, 0.09, 0.18, 0.19, 0.20, 0.16],
    )

    candidate = pd.read_csv(EXTENSION_TABLES / "synthesis__candidate_cross_cohort_replication.csv")
    candidate = candidate[(candidate["replication_tier"] == "replicated_fdr") & (candidate["names"] != "RHBDF2")].copy()
    candidate["Direction"] = np.where(candidate["positive_fdr_cohorts"].notna(), "Positive", "Negative")
    candidate["FDR-supporting cohorts"] = candidate["positive_fdr_cohorts"].fillna(candidate["negative_fdr_cohorts"])
    candidate["Best q"] = candidate["best_adjusted_pval"].map(fmt_p)
    candidate = candidate[["cell_type", "names", "Direction", "FDR-supporting cohorts", "Best q"]]
    candidate.columns = ["Cell type", "Gene", "Direction", "FDR-supporting cohorts", "Best q"]
    render_table(
        candidate,
        "Table_S3_replicated_candidates",
        "Supporting Table S3. Non-self candidate associations replicated at donor-level FDR",
        "Replication required the same cell type and direction in at least two independent donor-valid cohorts. RHBDF2 self-association anchors are excluded.",
        wrap={"Cell type": 22, "FDR-supporting cohorts": 26},
        column_widths=[0.20, 0.10, 0.12, 0.42, 0.10],
    )

    program = pd.read_csv(EXTENSION_TABLES / "synthesis__program_cross_cohort_replication.csv")
    program = program[program["replication_tier"] == "replicated_fdr"].copy()
    program["Direction"] = np.where(program["positive_fdr_cohorts"].notna(), "Positive", "Negative")
    program["FDR-supporting cohorts"] = program["positive_fdr_cohorts"].fillna(program["negative_fdr_cohorts"])
    program["Program"] = program["program"].str.replace("_", " ", regex=False)
    program["Best q"] = program["best_adjusted_pval"].map(fmt_p)
    program = program[["cell_type", "Program", "Direction", "FDR-supporting cohorts", "Best q"]]
    program.columns = ["Cell type", "Program", "Direction", "FDR-supporting cohorts", "Best q"]
    render_table(
        program,
        "Table_S4_replicated_programs",
        "Supporting Table S4. Transcript-derived programs replicated at donor-level FDR",
        "Program scores are mean-expression RNA proxies and do not directly measure pathway activation, phosphorylation, proteolysis, or signaling flux.",
        wrap={"Cell type": 22, "Program": 24, "FDR-supporting cohorts": 20},
        column_widths=[0.18, 0.22, 0.11, 0.40, 0.09],
    )

    spatial = pd.read_csv(EXTENSION_TABLES / "GSE269906_spatial__rhbdf2_spatial_pathology_comparison.csv")
    spatial_table = pd.DataFrame(
        {
            "Cell type": spatial["cell_type"],
            "AD donors": spatial["ad_donors"],
            "Control donors": spatial["control_donors"],
            "AD detection (%)": (100 * spatial["ad_mean_detection_fraction"]).map(lambda x: f"{x:.3f}"),
            "Control detection (%)": (100 * spatial["control_mean_detection_fraction"]).map(lambda x: f"{x:.3f}"),
            "AD-control (pp)": (100 * spatial["difference_ad_minus_control"]).map(lambda x: f"{x:+.3f}"),
            "Adjusted P": spatial["pvals_adj"].map(fmt_p),
        }
    )
    render_table(
        spatial_table,
        "Table_S5_spatial_pathology_comparison",
        "Supporting Table S5. Spatial RHBDF2 detection by pathology",
        "Donor means were compared with Welch tests and corrected across marker-derived cell types. No comparison passed FDR < 0.05.",
        font_size=6.7,
        column_widths=[0.17, 0.09, 0.11, 0.15, 0.17, 0.13, 0.12],
    )

    status = pd.read_csv(EXTENSION_TABLES / "expanded_database__expanded_database_status.csv")
    roles = {
        "agora": "AD multi-omics and target context",
        "signor": "Directed signaling and regulation",
        "iptmnet": "PTM and enzyme-substrate context",
        "merops": "Protease and cleavage relationships",
        "chembl": "Chemical bioactivity",
        "dgidb": "Drug-gene interactions",
        "biogrid": "Protein interaction network",
        "niagads": "AD genetics resource availability",
    }
    interpretations = {
        "agora": "Complete for 43 genes",
        "signor": "826 records; 8 completed-empty",
        "iptmnet": "Service failure; not biological absence",
        "merops": "ADAM17 targeted; others not applicable",
        "chembl": "30 mapped; 12 not found; 1 failed",
        "dgidb": "Complete for 43 genes",
        "biogrid": "API key unavailable",
        "niagads": "Candidate pages available for 43 genes",
    }
    status_table = pd.DataFrame(
        {
            "Resource": status["resource"].str.upper(),
            "Evidence role": status["resource"].map(roles),
            "Acquisition summary": status["resource"].map(interpretations),
            "Rows": status["rows"],
        }
    )
    render_table(
        status_table,
        "Table_S6_external_resource_status",
        "Supporting Table S6. Expanded external-resource acquisition status",
        "Database coverage is contextual evidence. Technical failure or missing credentials were retained as unavailable and never converted to biological absence.",
        wrap={"Evidence role": 30, "Acquisition summary": 34},
        column_widths=[0.14, 0.27, 0.43, 0.10],
    )


def main() -> None:
    build_focused_figures()
    build_supporting_tables()
    manifest = {
        "figures": sorted(path.name for path in FIGURE_DIR.glob("Figure_S*.png")),
        "tables": sorted(path.name for path in TABLE_DIR.glob("Table_S*.csv")),
    }
    (TABLE_DIR / "supporting_materials_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
