"""Donor-aware inflammatory RNA modules and a separate mouse validation branch."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
import csv
import gzip
import json
import re
import tarfile
import xml.etree.ElementTree as ET

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm


def read_one_to_one_orthologs(path: Path) -> pd.DataFrame:
    """Read MGI's protein-coding one-to-one mouse/human orthology report."""
    frame = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if frame.shape[1] != 6:
        raise ValueError("The MGI orthology report must have six columns.")
    frame.columns = ["mgi_id", "mouse_gene", "mouse_entrez", "hgnc_id", "human_gene", "human_entrez"]
    frame = frame.dropna(subset=["mouse_gene", "human_gene"])
    frame = frame.drop_duplicates(["human_gene", "mouse_gene"])
    if frame["human_gene"].duplicated().any() or frame["mouse_gene"].duplicated().any():
        raise ValueError("The MGI report contains non-unique gene-symbol mappings.")
    return frame


def read_gse148405_metadata(path: Path) -> pd.DataFrame:
    """Extract per-cell labels from GEO MINiML; never treat wells as mice."""
    with tarfile.open(path, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.name.endswith(".xml")]
        if len(members) != 1:
            raise ValueError("Expected one MINiML XML file in the GEO archive.")
        handle = archive.extractfile(members[0])
        if handle is None:
            raise ValueError("Could not read the MINiML XML file.")
        root = ET.parse(handle).getroot()
    rows = []
    pattern = re.compile(r"(?P<cell>P\d+_[A-Z]\d+)_(?P<genotype>5xFAD|WT)_C57BL/6J_(?P<region>.+)_rep(?P<rep>\d+)")
    for node in root.iter():
        if not node.tag.endswith("Sample"):
            continue
        title = next((child.text.strip() for child in node if child.tag.endswith("Title") and child.text), "")
        match = pattern.fullmatch(title)
        if not match:
            continue
        row = match.groupdict()
        row["mouse_id"] = f"{row['genotype']}_rep{row['rep']}"
        row["geo_accession"] = node.attrib.get("iid", "")
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("cell", verify_integrity=True)
    if frame.empty or frame["mouse_id"].nunique() < 2:
        raise ValueError("GSE148405 metadata did not resolve independent mice.")
    return frame


def prepare_gse148405(
    matrix_path: Path,
    metadata_path: Path,
    ortholog_path: Path,
    modules: dict[str, list[str]],
    output_path: Path,
    direct_genes: list[str] | None = None,
) -> dict[str, object]:
    """Stream the full CSV, retain panel counts, and preserve full-library sizes."""
    metadata = read_gse148405_metadata(metadata_path)
    orthologs = read_one_to_one_orthologs(ortholog_path)
    human_genes = set().union(*map(set, modules.values())) | {"RHBDF2", "ADAM17"} | set(direct_genes or [])
    mapping = orthologs[orthologs["human_gene"].isin(human_genes)].copy()
    mouse_genes = set(mapping["mouse_gene"])
    if not {"Rhbdf2", "Adam17"}.issubset(mouse_genes):
        raise ValueError("The orthology file does not map both required anchors.")

    retained: dict[str, np.ndarray] = {}
    duplicated = Counter()
    with gzip.open(matrix_path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if not header or header[0] != "GeneID":
            raise ValueError("Unexpected GSE148405 count-matrix header.")
        cell_ids = header[1:]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("Duplicate cell columns in the GSE148405 matrix.")
        selected_positions = [i for i, cell in enumerate(cell_ids) if cell in metadata.index]
        selected_ids = [cell_ids[i] for i in selected_positions]
        if len(selected_ids) != len(metadata):
            raise ValueError("GEO metadata and count-matrix cells do not match.")
        library_sizes = np.zeros(len(selected_ids), dtype=np.float64)
        n_genes_seen = 0
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"Malformed count row for {row[0] if row else 'unknown'}.")
            gene = row[0]
            if gene.startswith("ERCC-"):
                continue
            counts = np.fromiter((int(row[i + 1]) for i in selected_positions), dtype=np.int64, count=len(selected_positions))
            library_sizes += counts
            n_genes_seen += 1
            if gene in mouse_genes:
                if gene in retained:
                    retained[gene] += counts
                    duplicated[gene] += 1
                else:
                    retained[gene] = counts

    if np.any(library_sizes <= 0):
        raise ValueError("At least one retained mouse cell has no non-ERCC library counts.")
    gene_order = sorted(retained)
    obs = metadata.loc[selected_ids].copy()
    obs["source_library_counts"] = library_sizes
    obs["species"] = "Mus musculus"
    obs["cell_type"] = "Microglia"
    matrix = np.column_stack([retained[g] for g in gene_order])
    adata = ad.AnnData(X=sparse.csr_matrix(matrix), obs=obs, var=pd.DataFrame(index=gene_order))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path, compression="gzip")
    summary = {
        "accession": "GSE148405",
        "source_matrix": str(matrix_path),
        "source_metadata": str(metadata_path),
        "orthology_file": str(ortholog_path),
        "n_cells": int(adata.n_obs),
        "n_mice": int(obs["mouse_id"].nunique()),
        "n_regions": int(obs["region"].nunique()),
        "n_genes_scanned_excluding_ercc": n_genes_seen,
        "n_panel_genes": int(adata.n_vars),
        "panel_human_genes_requested": sorted(human_genes),
        "panel_mouse_genes_requested": sorted(mouse_genes),
        "panel_genes_missing": sorted(mouse_genes - set(gene_order)),
        "duplicated_panel_gene_rows": dict(duplicated),
        "library_denominator": "sum of all non-ERCC source count rows, before panel extraction",
    }
    output_path.with_suffix(".preparation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    obs.groupby(["genotype", "region", "mouse_id"], observed=True).size().rename("n_cells").reset_index().to_csv(
        output_path.with_suffix(".animal_region_counts.csv"), index=False
    )
    return summary


def aggregate_backed_population(
    path: Path,
    cell_type_column: str,
    cell_type_value: str,
    donor_column: str,
    condition_column: str,
    genes: list[str],
    region_column: str | None = None,
    region_value: str | None = None,
    chunk_size: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate raw panel counts by biological sample using full-library denominators."""
    obj = ad.read_h5ad(path, backed="r")
    try:
        obs = obj.obs
        mask = obs[cell_type_column].astype(str).eq(cell_type_value).to_numpy(copy=True)
        if region_column and region_value:
            mask &= obs[region_column].astype(str).eq(region_value).to_numpy()
        positions = np.flatnonzero(mask)
        present = [g for g in dict.fromkeys(genes) if g in obj.var_names]
        gene_positions = [int(obj.var_names.get_loc(g)) for g in present]
        sample_ids = obs.iloc[positions][donor_column].astype(str).to_numpy()
        sample_labels, sample_index = np.unique(sample_ids, return_inverse=True)
        aggregated = np.zeros((len(sample_labels), len(present)), dtype=np.float64)
        detected = np.zeros((len(sample_labels), len(present)), dtype=np.int64)
        libraries = np.zeros(len(sample_labels), dtype=np.float64)
        n_cells = np.bincount(sample_index, minlength=len(sample_labels))
        conditions = []
        for sample in sample_labels:
            vals = obs.iloc[positions][sample_ids == sample][condition_column].dropna().astype(str).unique()
            if len(vals) != 1:
                raise ValueError(f"Condition is not unique for biological sample {sample}: {vals.tolist()}")
            conditions.append(vals[0])

        for start in range(0, len(positions), chunk_size):
            end = min(start + chunk_size, len(positions))
            block = obj.X[positions[start:end]]
            if hasattr(block, "to_memory"):
                block = block.to_memory()
            if sparse.issparse(block):
                values = block[:, gene_positions].toarray()
                block_libraries = np.asarray(block.sum(axis=1)).reshape(-1)
            else:
                block = np.asarray(block)
                values = block[:, gene_positions]
                block_libraries = block.sum(axis=1)
            chunk_obs = obs.iloc[positions[start:end]]
            if "source_library_counts" in chunk_obs:
                block_libraries = chunk_obs["source_library_counts"].to_numpy(dtype=float)
            elif "source_total_umis" in chunk_obs:
                block_libraries = chunk_obs["source_total_umis"].to_numpy(dtype=float)
            elif "total_counts" in chunk_obs:
                block_libraries = chunk_obs["total_counts"].to_numpy(dtype=float)
            np.add.at(aggregated, sample_index[start:end], values)
            np.add.at(detected, sample_index[start:end], values > 0)
            np.add.at(libraries, sample_index[start:end], block_libraries)

        metadata = pd.DataFrame({
            "sample_id": sample_labels,
            "condition": conditions,
            "n_cells": n_cells,
            "source_library_counts": libraries,
        })
        if np.any(~np.isfinite(libraries)) or np.any(libraries <= 0):
            raise ValueError("Biological samples must have positive, finite library totals.")
        for anchor in ["RHBDF2", "ADAM17", "Rhbdf2", "Adam17"]:
            if anchor in present:
                j = present.index(anchor)
                metadata[f"{anchor}_detected_cells"] = detected[:, j]
                metadata[f"{anchor}_detection_fraction"] = detected[:, j] / n_cells
        log_cpm = pd.DataFrame(np.log1p(1e6 * aggregated / libraries[:, None]), columns=present)
        return metadata, log_cpm
    finally:
        obj.file.close()


def module_scores(
    log_cpm: pd.DataFrame,
    modules: dict[str, list[str]],
    anchor: str,
    minimum_genes: int,
    minimum_fraction: float,
    minimum_detection_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make donor-standardized RNA proxy scores and transparent coverage records."""
    scores = pd.DataFrame(index=log_cpm.index)
    coverage = []
    for name, requested in modules.items():
        unique = list(dict.fromkeys(g for g in requested if g != anchor))
        present = [g for g in unique if g in log_cpm.columns]
        eligible = [g for g in present if (log_cpm[g] > 0).mean() >= minimum_detection_fraction and log_cpm[g].std(ddof=1) > 0]
        complete = len(eligible) >= minimum_genes and len(eligible) / max(len(unique), 1) >= minimum_fraction
        coverage.append({
            "module": name,
            "anchor": anchor,
            "n_requested_after_anchor_exclusion": len(unique),
            "n_present": len(present),
            "n_scored": len(eligible),
            "present_genes": ";".join(present),
            "scored_genes": ";".join(eligible),
            "missing_genes": ";".join(g for g in unique if g not in present),
            "status": "eligible" if complete else "insufficient_gene_coverage",
        })
        if complete:
            values = log_cpm[eligible]
            scores[name] = ((values - values.mean()) / values.std(ddof=1)).mean(axis=1)
    return scores, pd.DataFrame(coverage)


def fit_module_associations(
    metadata: pd.DataFrame,
    scores: pd.DataFrame,
    anchor: str,
    cohort: str,
    population: str,
    minimum_samples: int,
) -> pd.DataFrame:
    """Regress each RNA score on anchor detection, condition, and library depth."""
    predictor_name = f"{anchor}_detection_fraction"
    if predictor_name not in metadata or len(metadata) < minimum_samples:
        return pd.DataFrame()
    predictor = np.log1p(1000 * metadata[predictor_name].to_numpy(dtype=float))
    if np.unique(predictor).size < 3 or np.isclose(np.std(predictor), 0):
        return pd.DataFrame()
    if "source_library_counts" not in metadata or "n_cells" not in metadata:
        raise ValueError("Library-depth adjustment requires source counts and cell counts per sample.")
    library_per_cell = metadata["source_library_counts"].to_numpy(dtype=float) / metadata["n_cells"].to_numpy(dtype=float)
    if np.any(~np.isfinite(library_per_cell)) or np.any(library_per_cell <= 0):
        raise ValueError("Library depth per cell must be positive and finite.")
    condition = pd.get_dummies(metadata["condition"].astype(str), drop_first=True, dtype=float).reset_index(drop=True)
    base = pd.DataFrame({"anchor_detection_log1p_per_1000": predictor})
    full = pd.concat([base, pd.DataFrame({"log_library_per_cell": np.log(library_per_cell)}), condition], axis=1)
    if len(metadata) - full.shape[1] - 1 >= 3:
        design = full
        condition_adjusted = not condition.empty
        library_adjusted = True
    else:
        with_library = full.drop(columns=list(condition.columns))
        if len(metadata) - with_library.shape[1] - 1 >= 3:
            design = with_library
            condition_adjusted = False
            library_adjusted = True
        else:
            design = base
            condition_adjusted = False
            library_adjusted = False
    design = sm.add_constant(design, has_constant="add")
    if np.linalg.matrix_rank(design.to_numpy(dtype=float)) < design.shape[1]:
        return pd.DataFrame()
    rows = []
    for module in scores:
        y = scores[module].to_numpy(dtype=float)
        fit = sm.OLS(y, design).fit()
        direction = np.sign(fit.params["anchor_detection_log1p_per_1000"])
        loo_effects = []
        for held_out in range(len(metadata)):
            included = np.arange(len(metadata)) != held_out
            loo_design = design.loc[included]
            if np.linalg.matrix_rank(loo_design.to_numpy(dtype=float)) < loo_design.shape[1]:
                continue
            loo_fit = sm.OLS(y[included], loo_design).fit()
            loo_effects.append(float(loo_fit.params["anchor_detection_log1p_per_1000"]))
        rows.append({
            "cohort": cohort,
            "population": population,
            "anchor": anchor,
            "module": module,
            "effect": float(fit.params["anchor_detection_log1p_per_1000"]),
            "pval": float(fit.pvalues["anchor_detection_log1p_per_1000"]),
            "n_samples": len(metadata),
            "condition_adjusted": condition_adjusted,
            "library_adjusted": library_adjusted,
            "r_squared": float(fit.rsquared),
            "loo_n_evaluable": len(loo_effects),
            "loo_sign_concordance": float(np.mean(np.sign(loo_effects) == direction)) if loo_effects else np.nan,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["qval_within_cohort_population_anchor"] = multipletests(result["pval"], method="fdr_bh")[1]
    return result


def exact_two_group_test(metadata: pd.DataFrame, scores: pd.DataFrame, control: str, disease: str) -> pd.DataFrame:
    """Animal-level exact permutation test for the disease-model contrast."""
    labels = metadata["condition"].astype(str).to_numpy()
    valid = np.isin(labels, [control, disease])
    labels = labels[valid]
    n_disease = int(np.sum(labels == disease))
    n_control = int(np.sum(labels == control))
    if min(n_disease, n_control) < 3:
        return pd.DataFrame()
    masks = []
    for indices in combinations(range(len(labels)), n_disease):
        mask = np.zeros(len(labels), dtype=bool)
        mask[list(indices)] = True
        masks.append(mask)
    rows = []
    for module in scores:
        values = scores.loc[valid, module].to_numpy(dtype=float)
        effect = float(values[labels == disease].mean() - values[labels == control].mean())
        null = np.array([values[m].mean() - values[~m].mean() for m in masks])
        pval = float(np.mean(np.abs(null) >= abs(effect) - 1e-12))
        rows.append({"module": module, "disease_minus_control": effect, "pval_exact": pval,
                     "n_disease_mice": n_disease, "n_control_mice": n_control, "n_permutations": len(masks)})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["qval_exact_bh"] = multipletests(result["pval_exact"], method="fdr_bh")[1]
    return result


def compare_species(human: pd.DataFrame, mouse: pd.DataFrame) -> pd.DataFrame:
    """Compare signs of like-for-like microglial anchor associations only."""
    rows = []
    if human.empty or mouse.empty:
        return pd.DataFrame()
    human = human[human["population"].eq("Microglia")]
    mouse = mouse[mouse["population"].eq("Microglia cortex")]
    for anchor_human, anchor_mouse in [("RHBDF2", "Rhbdf2"), ("ADAM17", "Adam17")]:
        for module in sorted(set(human["module"]) & set(mouse["module"])):
            h = human[human["anchor"].eq(anchor_human) & human["module"].eq(module)]
            m = mouse[mouse["anchor"].eq(anchor_mouse) & mouse["module"].eq(module)]
            if h.empty or m.empty:
                continue
            positive = h[(h["effect"] > 0) & (h["qval_within_cohort_population_anchor"] < 0.05)]
            negative = h[(h["effect"] < 0) & (h["qval_within_cohort_population_anchor"] < 0.05)]
            human_direction = 1 if len(positive) >= 2 else -1 if len(negative) >= 2 else 0
            mouse_effect = float(m.iloc[0]["effect"])
            mouse_q = float(m.iloc[0]["qval_within_cohort_population_anchor"])
            rows.append({
                "module": module,
                "human_anchor": anchor_human,
                "mouse_anchor": anchor_mouse,
                "n_human_microglia_cohorts": int(h["cohort"].nunique()),
                "human_positive_fdr_cohorts": int(len(positive)),
                "human_negative_fdr_cohorts": int(len(negative)),
                "human_replicated_direction": human_direction,
                "mouse_cortex_effect": mouse_effect,
                "mouse_cortex_qval": mouse_q,
                "same_direction": bool(human_direction != 0 and np.sign(mouse_effect) == human_direction),
                "cross_species_fdr_support": bool(human_direction != 0 and np.sign(mouse_effect) == human_direction and mouse_q < 0.05),
            })
    return pd.DataFrame(rows)
