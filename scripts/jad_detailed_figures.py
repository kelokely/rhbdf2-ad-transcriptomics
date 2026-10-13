"""Data-rich figures from frozen donor aggregates, without changing inference."""

from pathlib import Path
import json
import string

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import seaborn as sns

from src.jad_revision_utils import design_matrix, fit_many


COHORTS = ["GSE243292", "GSE157827", "GSE237718"]
PALETTE = {"GSE243292": "#277c82", "GSE157827": "#bb753a", "GSE237718": "#355e83",
           "GSE268599": "#718854", "SEAAD_MTG": "#9c4d42"}
MODULES = {"tnf_ligand_receptor_context": "TNF ligand/receptor", "tnf_nfkb_response": "TNF/NF-kB response",
           "tnf_feedback_survival": "Inflammatory feedback", "il1_inflammasome_context": "IL-1/inflammasome",
           "il6_jak_stat_response": "IL-6/JAK/STAT", "interferon_response": "Interferon response",
           "chemokine_recruitment": "Chemokine recruitment", "microglial_phagolysosomal_context": "Phagolysosomal context"}


def donor_abundance(metadata, counts, selected):
    """Align selected donor identifiers, never normalize by a candidate-panel sum."""
    if len(metadata) != len(counts) or metadata.donor_id.astype(str).duplicated().any():
        raise ValueError("Donor count rows must match unique aggregate metadata")
    if selected.donor_id.astype(str).duplicated().any():
        raise ValueError("Selected donor identifiers must be unique")
    lookup = pd.Series(np.arange(len(metadata)), index=metadata.donor_id.astype(str))
    positions = lookup.reindex(selected.donor_id.astype(str))
    if positions.isna().any():
        raise ValueError("Selected donor absent from aggregate metadata")
    positions = positions.to_numpy(int)
    total = metadata.iloc[positions].library_counts.to_numpy(float)
    if not np.isfinite(total).all() or not np.allclose(total, selected.library_counts.to_numpy(float)) or (total <= 0).any():
        raise ValueError("Full-library denominators do not match selected donors")
    values = counts.iloc[positions].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid donor counts")
    return pd.DataFrame(np.log1p(values / total[:, None] * 1e6), columns=counts.columns)


def load_donors(directory, genes):
    metadata = pd.read_csv(directory / "donor_aggregate_metadata.csv", dtype={"donor_id": str})
    selected = pd.read_csv(directory / "donors_primary.csv", dtype={"donor_id": str})
    counts = pd.read_csv(directory / "donor_counts.csv.gz", usecols=lambda gene: gene in genes)
    missing = set(genes) - set(counts.columns)
    if missing:
        raise ValueError(f"Plot genes absent from counts: {sorted(missing)}")
    return selected, donor_abundance(metadata, counts, selected)


def partial_coordinates(design, response):
    """Added-variable coordinates have exactly the original adjusted slope."""
    nuisance = design.drop(columns="anchor").to_numpy(float)
    anchor = design.anchor.to_numpy(float)
    response = np.asarray(response, float)
    if response.shape != anchor.shape or not np.isfinite(response).all():
        raise ValueError("Response must be a finite vector with one value per donor")
    x = anchor - nuisance @ np.linalg.lstsq(nuisance, anchor, rcond=None)[0]
    y = response - nuisance @ np.linalg.lstsq(nuisance, response, rcond=None)[0]
    if x @ x <= 1e-12:
        raise ValueError("Anchor has no independent variation after adjustment")
    slope = float(x @ y / (x @ x))
    return x, y, slope


def checked_coordinates(metadata, response, expected, anchor="RHBDF2"):
    design = design_matrix(metadata, anchor)
    x, y, slope = partial_coordinates(design, response)
    fitted = fit_many(pd.DataFrame({"response": response}), design, minimum_df=3).iloc[0]
    for name in ["effect", "ci95_low", "ci95_high"]:
        if not np.isclose(fitted[name], expected[name], rtol=1e-7, atol=1e-8):
            raise ValueError(f"Plotted {name} differs from saved model")
    if not np.isclose(slope, expected.effect, rtol=1e-7, atol=1e-8):
        raise ValueError("Added-variable slope differs from saved adjusted effect")
    return x, y


def save(fig, out, name):
    fig.savefig(out / "figures" / f"{name}.png", dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "figures" / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(labelsize=11)


def primary(frame, anchor="RHBDF2"):
    return frame[frame.anchor.eq(anchor) & frame.variant.eq("primary") & frame.subset.eq("primary")]


def detection_figure(profiles, out):
    profiles = profiles[profiles.cohort.ne("SEAAD_outputs")].copy()
    regional = profiles.groupby(["population", "cohort", "region"]).rh_detection_fraction.median()
    matrix = regional.groupby(["population", "cohort"]).median().unstack() * 100
    collapsed = profiles.groupby(["cohort", "population", "donor_id"], as_index=False).agg(
        detection_percent=("rh_detection_fraction", lambda x: x.median() * 100),
        minimum_region_cells=("n_cells", "min"), regions=("region", "nunique"))
    collapsed.to_csv(out / "tables/Figure_2_donor_detection_points.csv", index=False)
    fig = plt.figure(figsize=(12, 9), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
    ax = fig.add_subplot(grid[0, :])
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax,
                annot_kws={"fontsize": 12}, cbar_kws={"label": "Median detection (%)"})
    panel(ax, "A", "Cross-cell-type detection overview")
    ax.set_xlabel(""); ax.set_ylabel(""); ax.tick_params(axis="x", rotation=0, labelsize=11)
    rng = np.random.default_rng(20261012)
    for k, population in enumerate(["Microglia", "Astrocytes", "Endothelial cells"]):
        ax = fig.add_subplot(grid[1, k])
        subset = collapsed[collapsed.population.eq(population)]
        cohorts = [c for c in PALETTE if c in subset.cohort.unique()]
        for i, cohort in enumerate(cohorts):
            frame = subset[subset.cohort.eq(cohort)]
            values = frame.detection_percent.to_numpy()
            ax.boxplot(values, positions=[i], widths=.55, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": to_rgba(PALETTE[cohort], .15), "edgecolor": PALETTE[cohort]},
                       medianprops={"color": "#222222"}, whiskerprops={"color": PALETTE[cohort]},
                       capprops={"color": PALETTE[cohort]})
            x = i + rng.uniform(-.18, .18, len(frame))
            good = frame.minimum_region_cells.ge(20).to_numpy()
            ax.scatter(x[good], values[good], s=25, color=PALETTE[cohort], alpha=.65, zorder=3)
            ax.scatter(x[~good], values[~good], s=25, facecolors="none", edgecolors=PALETTE[cohort], zorder=3)
            ax.text(i, 1.01, f"n={len(frame)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=10)
        ax.set_xticks(range(len(cohorts)), [c.replace("GSE", "").replace("SEAAD_MTG", "SEA-AD") for c in cohorts], rotation=35, ha="right")
        ax.set_ylabel("RHBDF2-detected nuclei (%)", fontsize=12); ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e6e6e6", linewidth=.6)
        panel(ax, string.ascii_uppercase[k+1], population.replace("Microglia", "Microglia / myeloid"))
    save(fig, out, "Figure_2_detection")


def correlation_figure(data, out, correlations):
    fig = plt.figure(figsize=(12, 8.8), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    ax = fig.add_subplot(grid[0, :2])
    labels = correlations.cohort.str.replace("SEAAD_MTG", "SEA-AD", regex=False) + " | " + correlations.population.str.replace("Endothelial cells", "Endothelium", regex=False)
    values = correlations[["unadjusted_spearman_r", "partial_rank_r"]].copy()
    values.index = labels; values.columns = ["Unadjusted", "Depth/pathology adjusted"]
    sns.heatmap(values, annot=True, fmt=".2f", vmin=-1, vmax=1, center=0, cmap="BrBG_r", ax=ax,
                annot_kws={"fontsize": 13}, cbar_kws={"label": "Rank correlation", "shrink": .75})
    panel(ax, "A", "The seven primary correlation comparisons")
    ax.set_xlabel(""); ax.set_ylabel(""); ax.tick_params(axis="y", labelsize=11)
    ax = fig.add_subplot(grid[0, 2]); ax.axis("off")
    panel(ax, "B", "Adjusted-model summary")
    table = ax.table(cellText=[[str(row.n_donors), f"{row.partial_rank_r:.2f}", f"{row.qval_focused_seven:.3f}"] for row in correlations.itertuples()],
                     colLabels=["Donors", "Partial r", "FDR"], cellLoc="center", bbox=[0, .05, 1, .87])
    style_table(table, 12)
    points = []
    for i, cohort in enumerate(COHORTS):
        ax = fig.add_subplot(grid[1, i])
        m, _ = load_donors(data / "human" / cohort / "Microglia/all/primary", ["ADAM17"])
        ax.scatter(m.RHBDF2_detection_fraction*100, m.ADAM17_detection_fraction*100,
                   s=42, color=PALETTE[cohort], alpha=.85, edgecolors="white", linewidths=.5)
        ax.set_xlabel("RHBDF2 detection (%)", fontsize=12); ax.set_ylabel("ADAM17 detection (%)", fontsize=12)
        row = correlations[correlations.cohort.eq(cohort) & correlations.population.eq("Microglia")].iloc[0]
        panel(ax, string.ascii_uppercase[i+2], cohort)
        ax.text(.02, .98, f"n={len(m)}  |  raw r={row.unadjusted_spearman_r:.2f}\nadjusted r={row.partial_rank_r:.2f}",
                transform=ax.transAxes, va="top", fontsize=11,
                bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        points.append(m.assign(cohort=cohort))
    pd.concat(points).to_csv(out / "tables/Figure_5_donor_detection_points.csv", index=False)
    save(fig, out, "Figure_5_adjusted_anchor_correlations")


def style_table(table, fontsize=12):
    table.auto_set_font_size(False); table.set_fontsize(fontsize)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#ffffff"); cell.set_linewidth(2)
        cell.set_facecolor("#e8edef" if row == 0 else "#f4f5f5" if row % 2 else "white")
        cell.PAD = .10
        if row == 0: cell.get_text().set_fontweight("bold")


def tnf_figure(data, out, estimates):
    fig = plt.figure(figsize=(12, 8), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.3, .9])
    coordinates = []
    ordered = estimates.set_index("cohort").loc[COHORTS].reset_index()
    for i, row in ordered.iterrows():
        m, values = load_donors(data / "human" / row.cohort / "Microglia/all/primary", ["TNFAIP3"])
        x, y = checked_coordinates(m, values.TNFAIP3, row)
        ax = fig.add_subplot(grid[0, i])
        ax.scatter(x, y, s=42, color=PALETTE[row.cohort], edgecolors="white", linewidths=.5, alpha=.85)
        line = np.linspace(x.min(), x.max(), 100)
        ax.plot(line, row.effect * line, color="#333333", lw=1.5)
        ax.axhline(0, color="#cccccc", lw=.7); ax.axvline(0, color="#cccccc", lw=.7)
        ax.set_xlabel("Adjusted RHBDF2 detection\n(log1p detection residual)", fontsize=12)
        ax.set_ylabel("Adjusted TNFAIP3 abundance\n(log1p CPM residual)", fontsize=12)
        panel(ax, string.ascii_uppercase[i], row.cohort)
        ax.text(.03, .97, f"{len(m)} donors\nFDR={row.qval:.4g}", transform=ax.transAxes, va="top",
                fontsize=12, bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        coordinates.append(m.assign(cohort=row.cohort, TNFAIP3_log1p_CPM=values.TNFAIP3,
                                    anchor_residual=x, response_residual=y))
    lower = grid[1, :].subgridspec(1, 2, width_ratios=[1.25, 1.15])
    ax = fig.add_subplot(lower[0, 0])
    for i, row in ordered.iterrows():
        color = PALETTE[row.cohort]
        ax.plot([row.ci95_low, row.ci95_high], [i, i], color=color, lw=2)
        ax.scatter(row.effect, i, s=55, color=color, zorder=3)
    ax.set_yticks(range(len(ordered)), ordered.cohort, fontsize=12); ax.invert_yaxis()
    ax.set_ylim(2.55, -.55); ax.axvline(0, color="#888888", ls="--", lw=1)
    ax.set_xlabel("Adjusted coefficient and HC3 95% CI", fontsize=12)
    panel(ax, "D", "Effect size and uncertainty")
    ax = fig.add_subplot(lower[0, 1]); ax.axis("off"); panel(ax, "E", "Saved primary model estimates")
    table = ax.table(cellText=[[f"{r.effect:.3f}", f"{r.ci95_low:.3f} to {r.ci95_high:.3f}", f"{r.qval:.4g}"] for r in ordered.itertuples()],
                     colLabels=["Coefficient", "95% CI", "FDR"], cellLoc="center", bbox=[0, .05, 1, .88], colWidths=[.25, .50, .25])
    style_table(table, 11)
    pd.concat(coordinates).to_csv(out / "tables/Figure_6_TNFAIP3_donor_coordinates.csv", index=False)
    save(fig, out, "Figure_6_TNFAIP3_HC3")


def mouse_figure(data, out, estimates):
    directory = data / "mouse/cortex/primary"
    m = pd.read_csv(directory / "donors_primary.csv", dtype={"donor_id": str})
    scores = pd.read_csv(directory / "scores_inflammatory_modules_Rhbdf2_primary.csv")
    if len(m) != len(scores): raise ValueError("Mouse module scores and donor rows differ")
    for row in estimates.itertuples():
        checked_coordinates(m, scores[row.module], pd.Series(row._asdict()), "Rhbdf2")
    order = sorted(range(len(m)), key=lambda i: (m.iloc[i].condition != "WT", str(m.iloc[i].donor_id)))
    m = m.iloc[order].reset_index(drop=True); scores = scores.iloc[order].reset_index(drop=True)
    labels = [str(x).replace("_rep", " ") for x in m.donor_id]
    fig = plt.figure(figsize=(12, 8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[.75, 1.45], width_ratios=[1.55, 1])
    ax = fig.add_subplot(grid[0, 0]); colors = m.condition.map({"WT": "#355e83", "5xFAD": "#a14d36"})
    ax.bar(range(len(m)), m.Rhbdf2_detection_fraction*100, color=colors, width=.7)
    ax.set_xticks(range(len(m)), labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Rhbdf2 detection (%)", fontsize=12); panel(ax, "A", "Detection in each independent animal")
    for i, row in m.iterrows():
        ax.text(i, row.Rhbdf2_detection_fraction*100+.35, f"{row.n_cells} cells", ha="center", fontsize=9)
    ax.set_ylim(0, m.Rhbdf2_detection_fraction.max()*100+3)
    ax = fig.add_subplot(grid[0, 1])
    ax.bar(range(len(m)), np.log10(m.library_counts/m.n_cells), color=colors, width=.7)
    ax.set_xticks(range(len(m)), labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("log10 assigned counts / cell", fontsize=12); panel(ax, "B", "Sequencing-depth context")
    ax = fig.add_subplot(grid[1, 0])
    heat = scores.T; heat.index = [MODULES[g] for g in heat.index]; heat.columns = labels
    sns.heatmap(heat, cmap="BrBG_r", center=0, annot=True, fmt=".1f", annot_kws={"fontsize": 10}, ax=ax,
                cbar_kws={"label": "Mean donor-standardized RNA score", "shrink": .65})
    ax.set_xlabel(""); ax.set_ylabel(""); ax.tick_params(axis="y", labelsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=10)
    panel(ax, "C", "Observed transcript-module scores")
    ax = fig.add_subplot(grid[1, 1]); ordered = estimates.set_index("module").loc[scores.columns].reset_index()
    for i, row in ordered.iterrows():
        ax.plot([row.ci95_low, row.ci95_high], [i, i], color="#616c71", lw=1.5)
        ax.scatter(row.effect, i, s=30, color="#277c82")
    ax.set_yticks(range(len(ordered)), [f"{i+1}" for i in range(len(ordered))]); ax.invert_yaxis()
    ax.axvline(0, color="#999999", ls="--", lw=1)
    ax.set_xlabel("Rhbdf2 coefficient and HC3 95% CI", fontsize=11)
    ax.set_ylabel("Module row in panel C", fontsize=11); panel(ax, "D", "Adjusted anchor association")
    ax.text(.98, -.18, f"n={len(m)} animals; all FDR={estimates.qval.iloc[0]:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11)
    pd.concat([m, scores], axis=1).to_csv(out / "tables/Figure_7_mouse_donor_scores.csv", index=False)
    save(fig, out, "Figure_7_mouse_cortex")


def selection_figures(core, out):
    selected = core[core.cohort.eq("GSE268599") & core.variant.eq("primary") & core.anchor.eq("RHBDF2")]
    subsets = ["primary", "quota_inclusive", "competing_lineage_sensitivity"]
    names = ["Strict markers", "Quota inclusive", "Lineage sensitivity"]
    colors = ["#277c82", "#bb753a", "#355e83"]
    for region in ["BA20", "BA46", "EC", "V1", "V2"]:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), layout="constrained")
        frame = selected[selected.region.eq(region)]
        for i, gene in enumerate(["FCER1G", "TREM2", "TYROBP", "SYK", "C3"]):
            ax = axes.flat[i]
            for j, (subset, color) in enumerate(zip(subsets, colors)):
                found = frame[frame.gene.eq(gene) & frame.subset.eq(subset)]
                if len(found) != 1: raise ValueError(f"Selection figure requires one {region}/{gene}/{subset} estimate")
                row = found.iloc[0]
                ax.plot([row.ci95_low, row.ci95_high], [j, j], color=color, lw=2)
                ax.scatter(row.effect, j, color=color, s=45)
            ax.set_yticks(range(3), names, fontsize=10); ax.set_ylim(2.6, -.6)
            ax.axvline(0, ls="--", color="#aaaaaa", lw=1); ax.set_xlabel("Coefficient and HC3 95% CI", fontsize=11)
            primary_row = frame[frame.gene.eq(gene) & frame.subset.eq("primary")].iloc[0]
            panel(ax, string.ascii_uppercase[i], f"{gene}  |  FDR {primary_row.qval:.3g}")
        ax = axes.flat[5]; ax.axis("off"); panel(ax, "F", f"{region} selection comparison")
        text = []
        for subset, name in zip(subsets, names):
            row = frame[frame.subset.eq(subset)].iloc[0]
            text.append(f"{name}\n{int(row.n_donors)} donors, {int(row.n_cells):,} nuclei")
        ax.text(0, .89, "\n\n".join(text)+"\n\nThe lineage sensitivity retained\nthe same strict-marker cells.\nThese are not new cohorts.",
                transform=ax.transAxes, va="top", fontsize=11, linespacing=1.35)
        save(fig, out, f"Figure_S3_astrocyte_selection_{region}")


def genotype_figure(data, out, estimates):
    genotype = pd.read_csv(out / "tables/microglia_genotype_adjusted_sensitivity.csv")
    genotype = genotype[genotype.anchor.eq("RHBDF2") & genotype.gene.eq("TNFAIP3")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), layout="constrained", gridspec_kw={"height_ratios": [1.35, .75]})
    for i, cohort in enumerate(["GSE243292", "GSE237718"]):
        m, v = load_donors(data / "human" / cohort / "Microglia/all/primary", ["TNFAIP3"])
        field = "trem2" if cohort == "GSE243292" else "apoe_genotype"
        groups = sorted(m[field].dropna().astype(str).unique())
        ax = axes[0, i]
        for j, group in enumerate(groups):
            mask = m[field].astype(str).eq(group)
            ax.scatter(m.loc[mask, "RHBDF2_detection_fraction"]*100, v.loc[mask, "TNFAIP3"],
                       s=48, color=["#277c82", "#bb753a", "#355e83", "#718854"][j % 4],
                       label=f"{group} (n={mask.sum()})", alpha=.85)
        ax.set_xlabel("RHBDF2 detection (%)", fontsize=12); ax.set_ylabel("TNFAIP3 log1p CPM", fontsize=12)
        panel(ax, string.ascii_uppercase[i], f"{cohort} donor observations")
        ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(.5, -.28), ncol=2, frameon=False)
        ax = axes[1, i]
        before = estimates[estimates.cohort.eq(cohort)].iloc[0]
        after = genotype[genotype.cohort.eq(cohort)].iloc[0]
        for j, (row, color) in enumerate([(before, "#9c4d42"), (after, "#277c82")]):
            ax.plot([row.ci95_low, row.ci95_high], [j, j], color=color, lw=2)
            ax.scatter(row.effect, j, s=50, color=color)
            ax.text(.02, .97-j*.77, f"FDR={row.qval:.4g}", transform=ax.transAxes, va="top", fontsize=11)
        ax.set_yticks([0, 1], ["Pathology + depth", "Plus genotype"], fontsize=11); ax.set_ylim(1.55, -.55)
        ax.axvline(0, ls="--", lw=1, color="#999999"); ax.set_xlabel("Coefficient and HC3 95% CI", fontsize=12)
        panel(ax, string.ascii_uppercase[i+2], f"{cohort} model sensitivity")
    save(fig, out, "Figure_S16_genotype_sensitivity")


def build_all(data, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    profiles = pd.read_csv(out / "tables/donor_profile_eligibility.csv", dtype={"donor_id": str})
    correlations = pd.read_csv(out / "tables/adjusted_anchor_correlations.csv")
    tnf = pd.read_csv(out / "tables/human_microglia_TNFAIP3.csv")
    modules = pd.read_csv(data / "inflammatory_modules.csv")
    mouse = primary(modules, "Rhbdf2")
    mouse = mouse[mouse.cohort.eq("GSE148405") & mouse.region.eq("cortex")]
    detection_figure(profiles, out)
    correlation_figure(data, out, correlations)
    tnf_figure(data, out, tnf)
    mouse_figure(data, out, mouse)
    selection_figures(pd.read_csv(data / "core_candidates.csv"), out)
    genotype_figure(data, out, tnf)
    (out / "figure_validation.json").write_text(json.dumps({
        "source": str(data), "inference_changed": False, "TNFAIP3_donors": dict(zip(tnf.cohort, tnf.n_donors)),
        "mouse_animals": len(pd.read_csv(data / "mouse/cortex/primary/donors_primary.csv")), "plotted_coefficients_and_intervals_match_saved_models": True,
        "horizontal_jitter": "Figure 2 only, display only, fixed seed 20261012",
        "donor_coordinate_tables_saved": True}, indent=2)+"\n")
