"""Reanalyse the frozen human and mouse data without overwriting earlier runs."""

import argparse
import hashlib
import json
from pathlib import Path
import traceback

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from src.jad_revision_utils import (aggregate, full_libraries, design_matrix, fit_many, adjust_fdr,
                                  scores_for_modules, partial_detection_correlation, metadata_tests, write_json)
from src.multicell_ad_utils import CELL_TYPE_LABELS
from src.tnf_cross_species_utils import read_one_to_one_orthologs

ROOT = Path(__file__).resolve().parent


def safe(value):
    return str(value).replace(" ","_").replace("/","_")


def contexts(cohort, obj):
    if cohort=="GSE243292":
        return [(CELL_TYPE_LABELS.get(v,v),"all","primary",obj.obs.final_celltype.astype(str).eq(v).to_numpy(copy=True))
                for v in obj.obs.final_celltype.unique() if v in CELL_TYPE_LABELS]
    if cohort=="GSE268599":
        result=[]
        marker_genes=[g for g in ["P2RY12","CX3CR1","CSF1R","AIF1","LST1"] if g in obj.var_names]
        positions=[int(obj.var_names.get_loc(g)) for g in marker_genes]
        # Independent identity markers avoid filtering on the candidate outcomes.
        block=sparse.csr_matrix(obj.X[:,positions])
        multiple=np.asarray((block>0).sum(1)).ravel()>=2
        competing=multiple & (obj.obs.marker_score_microglia.to_numpy()>obj.obs.marker_score_astrocytes.to_numpy())
        for region in sorted(obj.obs.brain_region.unique()):
            base=obj.obs.brain_region.astype(str).eq(region).to_numpy(copy=True)
            strict=base & obj.obs.astrocyte_selection_tier.astype(str).eq("strict_marker").to_numpy()
            result.extend([("Astrocytes",region,"primary",strict),("Astrocytes",region,"quota_inclusive",base),
                           ("Astrocytes",region,"competing_lineage_sensitivity",strict & ~competing)])
        return result
    result=[]
    for population in sorted(obj.obs.cell_type.astype(str).unique()):
        if population in ["Unassigned","Other"]:
            continue
        mask=obj.obs.cell_type.astype(str).eq(population).to_numpy(copy=True)
        if cohort.startswith("SEAAD") and population=="Microglia":
            mask &= obj.obs.subpopulation.astype(str).str.startswith("Micro-PVM").to_numpy()
        result.append((population,"all","primary",mask))
        if cohort=="SEAAD_outputs" and population=="Microglia":
            labels=obj.obs.subpopulation.astype(str)
            for label,n in labels[mask].value_counts().items():
                if n>=200:
                    result.append((population,"all",f"omit_{safe(label)}",mask & labels.ne(label).to_numpy()))
    return result


def models(metadata, counts, cfg, key, candidates, modules, inflammatory, out, full_scope, outputs):
    metadata.to_csv(out/"donor_qc_all.csv",index=False)
    variants=[("primary",cfg["minimum_cells"],False,"HC3"),("min50",cfg["sensitivity_minimum_cells"],False,"HC3"),
              ("OLS",cfg["minimum_cells"],False,"OLS")]
    if "sex" in metadata and ("age" in metadata or "age_at_death" in metadata):
        variants.append(("age_sex",cfg["minimum_cells"],True,"HC3"))
    for variant,mincells,agesex,covariance in variants:
        valid=metadata.n_cells.ge(mincells)
        if agesex:
            age="age_at_death" if "age_at_death" in metadata else "age"
            valid &= pd.to_numeric(metadata[age],errors="coerce").notna() & metadata.sex.notna()
        m=metadata.loc[valid].reset_index(drop=True)
        c=counts.loc[valid].reset_index(drop=True)
        v=np.log1p(c.div(m.library_counts,axis=0)*1e6)
        m.to_csv(out/f"donors_{variant}.csv",index=False)
        for anchor in cfg["anchors"]:
            descriptor={**key,"variant":variant,"anchor":anchor,"n_donors":len(m),"n_cells":int(m.n_cells.sum()),"minimum_cells":mincells}
            detected_total=float((m.n_cells*m[f"{anchor}_detection_fraction"]).sum()) if f"{anchor}_detection_fraction" in m else 0
            detected_donors=int((m[f"{anchor}_detection_fraction"]>0).sum()) if f"{anchor}_detection_fraction" in m else 0
            descriptor.update(anchor_detected_cells=int(round(detected_total)),donors_with_anchor_detection=detected_donors)
            if anchor not in c or len(m)<cfg["minimum_donors"] or descriptor["n_cells"]<cfg.get("minimum_population_cells",0) or detected_total<cfg.get("minimum_anchor_detected_cells",0) or detected_donors<cfg.get("minimum_donors_with_detection",0):
                outputs["status"].append({**descriptor,"status":"ineligible","reason":"anchor, donor, population-cell, or detection eligibility threshold not met"})
                clean_ineligible(out,anchor,variant)
                continue
            try:
                design=design_matrix(m,anchor,agesex)
                fitted=fit_many(v,design,covariance,cfg["minimum_residual_df"])
                fitted=fitted.assign(**descriptor)
                fitted["qval_transcriptome"] = np.nan
                if full_scope:
                    from statsmodels.stats.multitest import multipletests
                    eligible=fitted.tested & fitted.gene.ne(anchor)
                    fitted.loc[eligible,"qval_transcriptome"]=multipletests(fitted.loc[eligible,"pval"],method="fdr_bh")[1]
                    if variant=="primary":
                        fitted.to_csv(out/f"transcriptome_{anchor}.csv",index=False)
                        fitted.loc[eligible,["gene","statistic"]].sort_values(["statistic","gene"],ascending=[False,True]).to_csv(out/f"ranked_{anchor}.rnk",sep="\t",header=False,index=False)
                        outputs["rankings"].append({**descriptor,"file":str(out/f"ranked_{anchor}.rnk")})
                subset=fitted[fitted.gene.isin(candidates) & fitted.gene.ne(anchor)].copy()
                outputs["candidates"].append(subset)
                core=yaml.safe_load((ROOT/cfg["registry"]).read_text())["candidate_genes"]
                outputs["core_candidates"].append(subset[subset.gene.isin(core)].copy())
                direct=fitted[fitted.gene.isin(inflammatory["direct_gene_checks"]) & fitted.gene.ne(anchor)].copy()
                outputs["inflammatory_genes"].append(direct)
                for family,sets,ming,minfrac in [("programs",modules,2,0),("inflammatory_modules",inflammatory["modules"],4,.5)]:
                    scores,coverage=scores_for_modules(v,sets,anchor,ming,minfrac)
                    coverage.assign(**descriptor).to_csv(out/f"coverage_{family}_{anchor}_{variant}.csv",index=False)
                    scores.to_csv(out/f"scores_{family}_{anchor}_{variant}.csv",index=False)
                    if not scores.empty:
                        result=fit_many(scores,design,covariance,cfg["minimum_residual_df"]).rename(columns={"gene":"module"})
                        outputs[family].append(result.assign(**descriptor))
                if variant=="primary":
                    d,i=metadata_tests(m,anchor,design,v,candidates,cfg["metadata_fields"])
                    if not d.empty: outputs["metadata_direct"].append(d.assign(**descriptor))
                    if not i.empty: outputs["metadata_interactions"].append(i.assign(**descriptor))
                outputs["status"].append({**descriptor,"status":"completed","reason":"","model_terms":";".join(design.columns)})
            except ValueError as e:
                clean_ineligible(out,anchor,variant)
                outputs["status"].append({**descriptor,"status":"ineligible","reason":str(e)})
        if variant=="primary" and len(m)>=cfg["minimum_donors"] and all(f"{g}_detection_fraction" in m for g in ["RHBDF2","ADAM17"]):
            outputs["correlations"].append({**key,**partial_detection_correlation(m),"n_donors":len(m)})


def clean_ineligible(out,anchor,variant):
    names=[f"{prefix}_{family}_{anchor}_{variant}.csv" for prefix in ["scores","coverage"] for family in ["programs","inflammatory_modules"]]
    if variant=="primary": names += [f"transcriptome_{anchor}.csv",f"ranked_{anchor}.rnk"]
    for name in names:
        path=out/name
        if path.exists(): path.unlink()


def human(cfg,out,outputs):
    registry=yaml.safe_load((ROOT/cfg["registry"]).read_text())
    inflammatory=yaml.safe_load((ROOT/cfg["inflammation"]).read_text())
    expanded=yaml.safe_load((ROOT/cfg["expanded_registry"]).read_text())
    candidates=list(dict.fromkeys(registry["candidate_genes"]+expanded.get("candidate_genes",[])))
    inputs={"GSE243292": {"prepared_path":"data/raw/GSE243292_ADsnRNAseq_GEO.h5ad"}}
    inputs.update({k:registry["cohorts"][k] for k in ["GSE157827","GSE237718","GSE268599","SEAAD_MTG"]})
    inputs["SEAAD_outputs"]={"prepared_path":"data/external_ad_cohorts/SEAAD_MTG/processed/SEAAD_MTG_adam17_outputs_canonical.h5ad"}
    for cohort,entry in inputs.items():
        path=ROOT/entry["prepared_path"]
        obj=ad.read_h5ad(path,backed="r")
        try:
            source=ROOT/registry["cohorts"]["SEAAD_MTG"]["source_directory"]/"SEAAD_MTG_RNAseq_final-nuclei.2026-06-22.h5ad" if cohort=="SEAAD_MTG" else None
            libraries,denominator=full_libraries(obj,source)
            donor,condition=("sampleID","atscore") if cohort=="GSE243292" else ("donor_id","pathology")
            outputs["inputs"].append({"cohort":cohort,"path":str(path),"n_cells":obj.n_obs,"n_genes":obj.n_vars,"denominator":denominator,
                                      "file_size":path.stat().st_size,"mtime_ns":path.stat().st_mtime_ns})
            for population,region,subset,mask in contexts(cohort,obj):
                if cohort=="GSE243292" and population=="Astrocytes":
                    clean=ad.read_h5ad(ROOT/"data/raw/GSE243292_Astrocyte_GEO.h5ad",backed="r")
                    ids=clean.obs_names[~clean.obs.astrocyte_subpopulation.astype(str).str.contains("Doublets")]
                    mask &= obj.obs_names.isin(ids)
                    clean.file.close()
                key={"cohort":cohort,"population":population,"region":region,"subset":subset}
                directory=out/"human"/cohort/safe(population)/safe(region)/subset
                directory.mkdir(parents=True,exist_ok=True)
                fingerprint={"source_size":path.stat().st_size,"source_mtime_ns":path.stat().st_mtime_ns,
                             "selected_identifiers_sha256":hashlib.sha256(pd.util.hash_pandas_object(obj.obs_names[mask],index=False).to_numpy().tobytes()).hexdigest()}
                marker=directory/"aggregate_done.json"
                cached=json.loads(marker.read_text()) if marker.exists() else {}
                if cached.get("fingerprint")==fingerprint:
                    metadata=pd.read_csv(directory/"donor_aggregate_metadata.csv",dtype={"donor_id":str})
                    counts=pd.read_csv(directory/"donor_counts.csv.gz")
                else:
                    print(f"Aggregating {key}: {int(mask.sum())} cells",flush=True)
                    metadata,counts,detections=aggregate(obj,mask,donor,condition,libraries)
                    metadata.to_csv(directory/"donor_aggregate_metadata.csv",index=False)
                    counts.to_csv(directory/"donor_counts.csv.gz",index=False)
                    detections.to_csv(directory/"donor_detected_cells.csv.gz",index=False)
                    write_json(directory/"aggregate_done.json",{**key,"fingerprint":fingerprint})
                if cohort.startswith("SEAAD"):
                    rich=pd.read_csv(ROOT/"results/committee_extensions/donor_metadata_audit/SEAAD_analysis_ready_donor_metadata.csv",dtype={"donor_id":str})
                    missing=[g for g in rich if g not in metadata]
                    metadata=metadata.merge(rich[["donor_id",*missing]],on="donor_id",how="left",validate="one_to_one")
                    metadata.to_csv(directory/"donor_metadata_enriched.csv",index=False)
                models(metadata,counts,cfg,key,candidates,registry["programs"],inflammatory,directory,obj.n_vars>1000,outputs)
                print(f"Completed {key}",flush=True)
        finally:
            obj.file.close()


def mouse(cfg,out,outputs):
    inflammatory=yaml.safe_load((ROOT/cfg["inflammation"]).read_text())
    source=ROOT/inflammatory["mouse"]["prepared_path"]
    qc=pd.read_csv(ROOT/"report/jad_submission_audit_2026-10-12/mouse_source_cell_qc_flags.csv",index_col="cell_id")
    obj=ad.read_h5ad(source)
    qc=qc.loc[obj.obs_names]
    mappings=read_one_to_one_orthologs(ROOT/inflammatory["orthology"]["path"])
    mapping=dict(zip(mappings.human_gene,mappings.mouse_gene))
    modules={k:[mapping[g] for g in v if g in mapping] for k,v in inflammatory["modules"].items()}
    direct=[mapping[g] for g in inflammatory["direct_gene_checks"] if g in mapping]
    cfgmouse={**cfg,"anchors":["Rhbdf2","Adam17"],"minimum_donors":6,"minimum_residual_df":3,"metadata_fields":[],
              "minimum_population_cells":0,"minimum_anchor_detected_cells":0,"minimum_donors_with_detection":0}
    base=(qc.source_genes_detected.ge(cfg["mouse_qc"]["minimum_detected_genes"]) & qc.assigned_gene_counts.ge(cfg["mouse_qc"]["minimum_assigned_gene_counts"])).to_numpy()
    qc["primary_qc_pass"]=base
    (out/"mouse").mkdir(exist_ok=True)
    qc.to_csv(out/"mouse/cell_qc.csv")
    branches=[("cortex","primary",base),("cortex","upper_count_ceiling",base & qc.assigned_gene_counts.le(700000).to_numpy()),
              ("cortex","omit_high_depth_WT_rep1",base & obj.obs.mouse_id.astype(str).ne("WT_rep1").to_numpy())]
    branches += [(r,"regional_sensitivity",base) for r in cfg["mouse_qc"]["sensitivity_regions"]]
    for region,subset,mask in branches:
        mask=mask & obj.obs.region.astype(str).eq(region).to_numpy()
        key={"cohort":"GSE148405","population":"Microglia","region":region,"subset":subset}
        directory=out/"mouse"/region/subset
        directory.mkdir(parents=True,exist_ok=True)
        m,c,d=aggregate(obj,mask,"mouse_id","genotype",obj.obs.source_library_counts.to_numpy(float))
        m.to_csv(directory/"donor_aggregate_metadata.csv",index=False)
        c.to_csv(directory/"donor_counts.csv.gz",index=False)
        # For this small animal arm, age/sex and min50 donor profiles are sensitivities, not added covariates.
        models(m,c,cfgmouse,key,direct,{}, {"direct_gene_checks":direct,"modules":modules},directory,False,outputs)
        print(f"Completed mouse {region} {subset}: {int(mask.sum())} cells",flush=True)


def synthesize(outputs,out):
    for name,items in outputs.items():
        if name in ["inputs","rankings","status","correlations"]:
            frame=pd.DataFrame(items)
            if name=="correlations" and not frame.empty:
                adjust_fdr(frame,["subset"],"qval")
        else:
            frame=pd.concat(items,ignore_index=True) if items else pd.DataFrame()
            if not frame.empty:
                groups=["cohort","subset","variant","anchor"]
                if name in ["inflammatory_genes","inflammatory_modules"]:
                    groups += ["population","region"]
                adjust_fdr(frame,groups)
                frame["study_family"]=np.where(frame.cohort.eq("SEAAD_outputs"),"separate_SEAAD_output_panel",
                                      np.where(frame.cohort.eq("GSE148405"),"mouse","independent_human_cohorts"))
                adjust_fdr(frame,["study_family","subset","variant","anchor"],"qval_study_family")
        frame.to_csv(out/f"{name}.csv",index=False)
    for family,label in [("core_candidates","gene"),("candidates","gene"),("programs","module")]:
        frame=pd.read_csv(out/f"{family}.csv")
        selected=frame[(frame.variant=="primary") & (frame.subset=="primary") & ~frame.cohort.isin(["SEAAD_outputs","GSE148405"])]
        rows=[]
        for (population,anchor,item),group in selected.groupby(["population","anchor",label]):
            positive=sorted(group.loc[(group.qval<.05)&(group.effect>0),"cohort"].unique())
            negative=sorted(group.loc[(group.qval<.05)&(group.effect<0),"cohort"].unique())
            rows.append({"population":population,"anchor":anchor,label:item,"positive_fdr_cohorts":";".join(positive),
                         "negative_fdr_cohorts":";".join(negative),"n_positive":len(positive),"n_negative":len(negative),
                         "replicated":max(len(positive),len(negative))>=2 and not (positive and negative)})
        pd.DataFrame(rows).to_csv(out/f"replication_{family}.csv",index=False)


def enrichment(cfg,out):
    import gseapy as gp
    settings=cfg["enrichment"]
    directory=out/"enrichment"
    directory.mkdir(exist_ok=True)
    gmt=directory/f"{settings['library']}.gmt"
    if not gmt.exists():
        import requests
        url="https://maayanlab.cloud/Enrichr/geneSetLibrary"
        response=requests.get(url,params={"mode":"text","libraryName":settings["library"]},timeout=120)
        response.raise_for_status()
        if response.text.count("\t")<50: raise ValueError("Invalid Hallmark GMT response")
        gmt.write_text(response.text)
    write_json(directory/"library_provenance.json",{"source":"Enrichr","library":settings["library"],"retrieval_date":"2026-10-12",
                "sha256":hashlib.sha256(gmt.read_bytes()).hexdigest(),"gseapy_version":gp.__version__,**settings})
    rows=[]
    rankings=pd.read_csv(out/"rankings.csv")
    for row in rankings.itertuples(index=False):
        if row.population not in ["Microglia","Astrocytes","Endothelial cells"] or row.subset!="primary": continue
        rnk=pd.read_csv(row.file,sep="\t",header=None)
        # Constant-response genes were excluded when constructing the rank file.
        result=gp.prerank(rnk=rnk,gene_sets=str(gmt),permutation_num=settings["permutations"],seed=settings["seed"],
                          min_size=settings["minimum_genes"],max_size=settings["maximum_genes"],threads=4,outdir=None,no_plot=True,ascending=None)
        result.res2d.assign(cohort=row.cohort,population=row.population,anchor=row.anchor,rank_order="primary").to_csv(directory/f"{row.cohort}_{safe(row.population)}_{row.anchor}.csv",index=False)
        rows.append(result.res2d.assign(cohort=row.cohort,population=row.population,anchor=row.anchor,rank_order="primary"))
        reversed_ties=rnk.sort_values([1,0],ascending=[False,False],kind="stable")
        sensitivity=gp.prerank(rnk=reversed_ties,gene_sets=str(gmt),permutation_num=settings["permutations"],seed=settings["seed"],
                               min_size=settings["minimum_genes"],max_size=settings["maximum_genes"],threads=4,outdir=None,no_plot=True,ascending=None)
        rows.append(sensitivity.res2d.assign(cohort=row.cohort,population=row.population,anchor=row.anchor,rank_order="reverse_gene_order_within_ties"))
        print(f"Hallmark GSEA completed {row.cohort} {row.population} {row.anchor}",flush=True)
    pd.concat(rows,ignore_index=True).to_csv(directory/"hallmark_all_results.csv",index=False)


def refresh_metadata(cfg,out):
    registry=yaml.safe_load((ROOT/cfg["registry"]).read_text())
    expanded=yaml.safe_load((ROOT/cfg["expanded_registry"]).read_text())
    genes=list(dict.fromkeys(registry["candidate_genes"]+expanded["candidate_genes"]))
    statuses=pd.read_csv(out/"status.csv")
    statuses=statuses[(statuses.variant=="primary")&statuses.status.eq("completed")&statuses.cohort.ne("GSE148405")]
    direct,interactions=[],[]
    for row in statuses.itertuples(index=False):
        directory=out/"human"/row.cohort/safe(row.population)/safe(row.region)/row.subset
        metadata=pd.read_csv(directory/"donors_primary.csv")
        all_metadata=pd.read_csv(directory/"donor_qc_all.csv")
        counts=pd.read_csv(directory/"donor_counts.csv.gz",usecols=lambda g:g in genes)
        counts=counts.loc[all_metadata.n_cells.ge(cfg["minimum_cells"])].reset_index(drop=True)
        values=np.log1p(counts.div(metadata.library_counts,axis=0)*1e6)
        design=design_matrix(metadata,row.anchor)
        d,i=metadata_tests(metadata,row.anchor,design,values,genes,cfg["metadata_fields"])
        descriptor={k:getattr(row,k) for k in ["cohort","population","region","subset","variant","anchor"]}
        if not d.empty:direct.append(d.assign(**descriptor))
        if not i.empty:interactions.append(i.assign(**descriptor))
    for name,items in [("metadata_direct",direct),("metadata_interactions",interactions)]:
        frame=pd.concat(items,ignore_index=True)
        adjust_fdr(frame,["cohort","subset","variant","anchor"])
        frame["study_family"]=np.where(frame.cohort.eq("SEAAD_outputs"),"separate_SEAAD_output_panel","independent_human_cohorts")
        adjust_fdr(frame,["study_family","subset","variant","anchor"],"qval_study_family")
        frame.to_csv(out/f"{name}.csv",index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/jad_revision.yaml")
    parser.add_argument("--output-dir",help="Use a new result directory instead of the configured frozen-run directory")
    parser.add_argument("--report-dir",help="Directory for supplementary genotype-sensitivity artifacts")
    parser.add_argument("--stage",choices=["human","mouse","all","enrichment","metadata","genotype"],default="all")
    args=parser.parse_args()
    cfg=yaml.safe_load((ROOT/args.config).read_text())
    if args.output_dir:cfg["output_dir"]=args.output_dir
    cfg["report_dir"]=args.report_dir or ("report/"+Path(args.output_dir).name if args.output_dir else cfg.get("report_dir","report/jad_revised_manuscript_2026-10-12"))
    out=ROOT/cfg["output_dir"]
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"run_status.json",{"stage":args.stage,"status":"running"})
    (out/"effective_config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
    outputs={k:[] for k in ["inputs","rankings","status","correlations","candidates","core_candidates","programs","inflammatory_genes","inflammatory_modules","metadata_direct","metadata_interactions"]}
    if args.stage in ["human","mouse"]:
        for name in outputs:
            previous=out/f"{name}.csv"
            if previous.exists() and previous.stat().st_size>2:
                frame=pd.read_csv(previous)
                if "cohort" in frame:
                    keep=frame.cohort.eq("GSE148405") if args.stage=="human" else frame.cohort.ne("GSE148405")
                    frame=frame[keep]
                if name in ["inputs","rankings","status","correlations"]:
                    outputs[name]=frame.to_dict("records")
                elif not frame.empty: outputs[name]=[frame]
    if args.stage in ["human","all"]: human(cfg,out,outputs)
    if args.stage in ["mouse","all"]: mouse(cfg,out,outputs)
    if args.stage=="metadata":refresh_metadata(cfg,out)
    elif args.stage not in ["enrichment","genotype"]: synthesize(outputs,out)
    if args.stage in ["human","all","genotype"]:
        from scripts import run_jad_genotype_sensitivity as genotype
        genotype.DATA=out
        genotype.OUT=ROOT/cfg["report_dir"]
        (genotype.OUT/"tables").mkdir(parents=True,exist_ok=True)
        (genotype.OUT/"figures").mkdir(exist_ok=True)
        try:
            genotype.main()
            write_json(out/"genotype_status.json",{"status":"completed","report_dir":str(genotype.OUT)})
        except Exception as e:
            write_json(out/"genotype_status.json",{"status":"failed","error":str(e),"traceback":traceback.format_exc()})
            print(f"Genotype sensitivity unavailable: {e}",flush=True)
    if args.stage in ["all","enrichment"] and cfg["enrichment"]["enabled"]:
        try:
            enrichment(cfg,out)
            write_json(out/"enrichment_status.json",{"status":"completed"})
        except Exception as e:
            write_json(out/"enrichment_status.json",{"status":"failed","error":str(e),"traceback":traceback.format_exc()})
            print(f"Optional enrichment failed: {e}",flush=True)
    write_json(out/"run_status.json",{"stage":args.stage,"status":"completed","note":"Inspect model_status and optional enrichment status separately"})


if __name__=="__main__":
    main()
