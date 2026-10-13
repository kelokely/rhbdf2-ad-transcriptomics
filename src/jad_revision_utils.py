"""Independent donor-level reanalysis with explicit QC and robust uncertainty."""

from pathlib import Path
import json

import anndata as ad
from anndata.io import read_elem
import h5py
import numpy as np
import pandas as pd
from scipy import sparse, stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def full_libraries(obj, source_path=None):
    for field in ["source_library_counts", "source_total_umis", "total_counts"]:
        if field in obj.obs:
            values = obj.obs[field].to_numpy(dtype=float)
            break
    else:
        if source_path is None:
            return None, "sum of all genes in the full count matrix"
        with h5py.File(source_path, "r") as handle:
            group = handle["obs"]
            index = pd.Index(read_elem(group[group.attrs["_index"]]))
            positions = index.get_indexer(obj.obs_names)
            if (positions < 0).any():
                raise ValueError("Prepared SEA-AD nuclei do not match source identifiers")
            values = np.asarray(read_elem(group["Number of UMIs"]), dtype=float)[positions]
        field = "source obs Number of UMIs matched by nucleus identifier"
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Full-library counts must be positive and finite")
    return values, field


def aggregate(obj, mask, donor_column, condition_column, libraries=None, chunk_size=3000):
    positions = np.flatnonzero(mask)
    obs = obj.obs.iloc[positions].copy()
    labels, codes = np.unique(obs[donor_column].astype(str), return_inverse=True)
    genes = obj.var_names.astype(str).tolist()
    if len(set(genes)) != len(genes):
        raise ValueError("Duplicate gene names require resolution before aggregation")
    counts = np.zeros((len(labels), len(genes)))
    detections = np.zeros_like(counts)
    totals = np.zeros(len(labels))
    for start in range(0, len(positions), chunk_size):
        end = min(start + chunk_size, len(positions))
        block = obj.X[positions[start:end]]
        block = sparse.csr_matrix(block)
        if block.data.size and ((block.data < 0).any() or not np.isfinite(block.data).all()
                                or not np.allclose(block.data, np.rint(block.data))):
            raise ValueError("Expected finite, nonnegative integer counts in X")
        membership = sparse.csr_matrix((np.ones(end-start), (codes[start:end], np.arange(end-start))),
                                       shape=(len(labels), end-start))
        counts += (membership @ block).toarray()
        detected = block.copy()
        detected.data = (detected.data > 0).astype(float)
        detections += (membership @ detected).toarray()
        cell_totals = libraries[positions[start:end]] if libraries is not None else np.asarray(block.sum(1)).ravel()
        totals += np.bincount(codes[start:end], weights=cell_totals, minlength=len(labels))
    metadata = pd.DataFrame({"donor_id": labels, "n_cells": np.bincount(codes), "library_counts": totals})
    grouped = obs.assign(_donor=obs[donor_column].astype(str)).groupby("_donor", observed=True)
    for column in obs.columns:
        if column == donor_column or grouped[column].nunique(dropna=True).max() > 1:
            continue
        metadata[column] = grouped[column].first().reindex(labels).to_numpy()
    if condition_column not in metadata:
        raise ValueError("Pathology/genotype is not unique within donor")
    metadata["condition"] = metadata[condition_column].astype(str)
    for anchor in ["RHBDF2", "ADAM17", "Rhbdf2", "Adam17"]:
        if anchor in genes:
            j = genes.index(anchor)
            metadata[f"{anchor}_detection_fraction"] = detections[:, j] / metadata.n_cells
    if (totals <= 0).any():
        raise ValueError("Zero donor full-library denominator")
    return metadata, pd.DataFrame(counts, columns=genes), pd.DataFrame(detections, columns=genes)


def design_matrix(metadata, anchor, age_sex=False):
    x = np.log1p(1000 * metadata[f"{anchor}_detection_fraction"].to_numpy(float))
    if len(np.unique(x)) < 3:
        raise ValueError("Fewer than three unique anchor detection values")
    depth = np.log1p(metadata.library_counts.to_numpy(float) / metadata.n_cells.to_numpy(float))
    design = pd.DataFrame({"intercept": np.ones(len(metadata)), "anchor": x,
                           "log_library_per_cell": depth-depth.mean()}, index=metadata.index)
    dummies = pd.get_dummies(metadata.condition, prefix="condition", drop_first=True, dtype=float)
    design = pd.concat([design, dummies], axis=1)
    if age_sex:
        age_column = "age_at_death" if "age_at_death" in metadata else "age"
        age = pd.to_numeric(metadata[age_column], errors="coerce")
        if age.isna().any() or metadata.sex.isna().any():
            raise ValueError("Age/sex sensitivity requires complete cases")
        design["age"] = age-age.mean()
        design = pd.concat([design, pd.get_dummies(metadata.sex.astype(str), prefix="sex", drop_first=True, dtype=float)], axis=1)
    if np.linalg.matrix_rank(design.to_numpy(float)) != design.shape[1]:
        raise ValueError("Rank-deficient design")
    return design


def fit_many(values, design, covariance="HC3", minimum_df=5):
    x = np.asarray(design, dtype=float)
    y = np.asarray(values, dtype=float)
    df = len(x)-x.shape[1]
    if df < minimum_df:
        raise ValueError(f"Only {df} residual degrees of freedom; require {minimum_df}")
    inverse = np.linalg.pinv(x.T @ x)
    influence = x @ inverse
    beta = influence.T @ y
    residual = y-x @ beta
    leverage = np.sum(influence*x, axis=1)
    if (leverage >= 1-1e-10).any():
        raise ValueError("A donor has unit leverage; robust covariance undefined")
    if covariance == "HC3":
        se = np.sqrt((influence[:, 1]/(1-leverage))**2 @ (residual**2))
    elif covariance == "OLS":
        se = np.sqrt((residual**2).sum(0)/df * inverse[1, 1])
    else:
        raise ValueError("Unsupported covariance")
    varying = np.std(y, axis=0) > 1e-10
    t = np.divide(beta[1], se, out=np.zeros_like(se), where=se > 1e-12)
    p = 2*stats.t.sf(np.abs(t), df)
    p[~varying] = 1
    ci = stats.t.ppf(.975, df)*se
    return pd.DataFrame({"gene": values.columns, "effect": beta[1], "standard_error": se,
                         "statistic": t, "pval": p, "ci95_low": beta[1]-ci, "ci95_high": beta[1]+ci,
                         "residual_df": df, "covariance": covariance, "tested": varying,
                         "maximum_leverage": leverage.max()})


def adjust_fdr(frame, groups, column="qval"):
    if frame.empty:
        frame[column] = pd.Series(dtype=float)
        return frame
    for _, indexes in frame.groupby(groups, dropna=False).groups.items():
        valid=frame.loc[indexes,"pval"].notna() & np.isfinite(frame.loc[indexes,"pval"])
        tested=frame.loc[indexes].index[valid]
        frame.loc[tested, column] = multipletests(frame.loc[tested, "pval"], method="fdr_bh")[1] if len(tested) else np.nan
    return frame


def scores_for_modules(values, modules, anchor, minimum_genes=2, minimum_fraction=0):
    scores, coverage = {}, []
    for name, requested in modules.items():
        requested = list(dict.fromkeys(g for g in requested if g != anchor))
        available = [g for g in requested if g in values and values[g].std(ddof=1) > 1e-10]
        eligible = len(available) >= minimum_genes and len(available)/max(1,len(requested)) >= minimum_fraction
        coverage.append({"module": name, "n_requested": len(requested), "n_present_varying": len(available),
                         "genes": ";".join(available), "eligible": eligible,
                         "interpretation": "complete proxy" if len(available)==len(requested) else "partial gene-set proxy"})
        if eligible:
            v = values[available]
            scores[name] = ((v-v.mean())/v.std(ddof=1)).mean(axis=1)
    return pd.DataFrame(scores, index=values.index), pd.DataFrame(coverage)


def partial_detection_correlation(metadata):
    y = stats.rankdata(metadata.RHBDF2_detection_fraction)
    z = stats.rankdata(metadata.ADAM17_detection_fraction)
    nuisance = pd.DataFrame({"intercept": 1., "depth": stats.rankdata(metadata.library_counts/metadata.n_cells)}, index=metadata.index)
    nuisance = pd.concat([nuisance, pd.get_dummies(metadata.condition, drop_first=True, dtype=float)], axis=1)
    n = nuisance.to_numpy(float)
    y -= n @ np.linalg.lstsq(n,y,rcond=None)[0]
    z -= n @ np.linalg.lstsq(n,z,rcond=None)[0]
    r = np.corrcoef(y,z)[0,1]
    df = len(y)-np.linalg.matrix_rank(n)-1
    p = 2*stats.t.sf(abs(r)*np.sqrt(df/max(1e-15,1-r*r)), df) if df > 0 else np.nan
    return {"partial_rank_r": r, "pval": p, "residual_df": df,
            "unadjusted_spearman_r": stats.spearmanr(metadata.RHBDF2_detection_fraction, metadata.ADAM17_detection_fraction).statistic}


def metadata_tests(metadata, anchor, design, values, genes, fields):
    direct, interactions = [], []
    base = design[["intercept", "log_library_per_cell"]]
    outcome = design.anchor
    for field in fields:
        if field not in metadata:
            continue
        raw = metadata[field]
        numeric = pd.to_numeric(raw, errors="coerce")
        is_numeric = numeric.notna().sum() == raw.notna().sum() and raw.nunique() > 5
        complete = numeric.notna() if is_numeric else raw.notna() & ~raw.astype(str).isin(["Unknown","nan","Not available"])
        if not is_numeric:
            counts=raw[complete].astype(str).value_counts()
            complete &= raw.astype(str).isin(counts[counts>=4].index)
        if complete.sum() < 10:
            continue
        factor = pd.DataFrame({field: numeric[complete]-numeric[complete].mean()}) if is_numeric else pd.get_dummies(raw[complete].astype(str),prefix=field,drop_first=True,dtype=float)
        if factor.shape[1] == 0 or factor.shape[1] > 5:
            continue
        # Do not silently duplicate the pathology adjustment with an identical factor.
        for kind in ["direct", "interaction"]:
            x = pd.concat([base.loc[complete], factor],axis=1) if kind=="direct" else (
                design.loc[complete].copy() if field=="condition" else pd.concat([design.loc[complete],factor],axis=1))
            terms = list(factor.columns)
            if kind=="interaction":
                terms = [f"interaction_{c}" for c in factor]
                for c, term in zip(factor,terms):
                    x[term] = design.loc[complete,"anchor"]*factor[c]
            if len(x)-x.shape[1]<5 or np.linalg.matrix_rank(x)!=x.shape[1]:
                continue
            xv=x.to_numpy(float)
            leverage=np.sum((xv@np.linalg.pinv(xv.T@xv))*xv,axis=1)
            if (leverage>=1-1e-10).any():
                continue
            columns = ["anchor_detection"] if kind=="direct" else [g for g in genes if g in values and g!=anchor]
            for gene in columns:
                y = outcome.loc[complete] if kind=="direct" else values.loc[complete,gene]
                if np.std(y)<1e-10:
                    continue
                fitted = sm.OLS(y, x).fit(cov_type="HC3",use_t=True)
                restriction = np.zeros((len(terms),x.shape[1]))
                for i, term in enumerate(terms):
                    restriction[i,x.columns.get_loc(term)] = 1
                p = float(fitted.wald_test(restriction,use_f=True,scalar=True).pvalue)
                if not np.isfinite(p):
                    continue
                row = {"field":field,"gene":gene,"n_donors":len(x),"pval":p,"test":"HC3 joint F test", "factor_levels":raw[complete].nunique()}
                (direct if kind=="direct" else interactions).append(row)
    return pd.DataFrame(direct),pd.DataFrame(interactions)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str)+"\n")
