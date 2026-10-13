"""Genotype-adjusted microglial sensitivity using the frozen donor count aggregates."""

from pathlib import Path
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.jad_revision_utils import design_matrix,fit_many,scores_for_modules

DATA=Path(os.getenv("RHBDF2_JAD_RESULTS_DIR",str(ROOT/"results/JAD_revision_2026-10-12")))
OUT=Path(os.getenv("RHBDF2_JAD_REPORT_DIR",str(ROOT/"report/jad_revised_manuscript_2026-10-12")))


def main():
    config=yaml.safe_load((ROOT/"configs/tnf_cross_species.yaml").read_text())
    effective=yaml.safe_load((DATA/"effective_config.yaml").read_text())
    minimum_cells=int(effective["minimum_cells"])
    genes=list(dict.fromkeys(config["direct_gene_checks"]+sum(config["modules"].values(),[])))
    outcomes=[];status=[]
    for cohort in ["GSE243292","GSE157827","GSE237718"]:
        directory=DATA/"human"/cohort/"Microglia/all/primary"
        m=pd.read_csv(directory/"donors_primary.csv")
        all_m=pd.read_csv(directory/"donor_qc_all.csv")
        c=pd.read_csv(directory/"donor_counts.csv.gz",usecols=lambda g:g in genes)
        c=c.loc[all_m.n_cells.ge(minimum_cells)].reset_index(drop=True)
        fields=[g for g in (["apoe","trem2"] if cohort=="GSE243292" else ["apoe_genotype"]) if g in m]
        if not fields:
            status.append({"cohort":cohort,"status":"unavailable","reason":"No donor genotype fields in prepared metadata"});continue
        complete=m[fields].notna().all(axis=1)
        m=m[complete].reset_index(drop=True);c=c[complete].reset_index(drop=True)
        v=np.log1p(c.div(m.library_counts,axis=0)*1e6)
        for anchor in ["RHBDF2","ADAM17"]:
            design=design_matrix(m,anchor)
            for field in fields:
                design=pd.concat([design,pd.get_dummies(m[field].astype(str),prefix=field,drop_first=True,dtype=float)],axis=1)
            descriptor={"cohort":cohort,"population":"Microglia","anchor":anchor,"n_donors":len(m),
                        "covariates":";".join(design.columns),"genotype_fields":";".join(fields),"variant":"genotype_adjusted"}
            if np.linalg.matrix_rank(design)!=design.shape[1]:
                status.append({**descriptor,"status":"ineligible","reason":"Genotype/pathology design is rank deficient"});continue
            try:
                direct=v[[g for g in config["direct_gene_checks"] if g in v and g!=anchor]]
                scores,coverage=scores_for_modules(v,config["modules"],anchor,4,.5)
                for family,response in [("direct_genes",direct),("modules",scores)]:
                    result=fit_many(response,design,"HC3",5)
                    result["qval"]=multipletests(result.pval,method="fdr_bh")[1]
                    outcomes.append(result.assign(family=family,**descriptor))
                coverage.assign(**descriptor).to_csv(OUT/"tables"/f"{cohort}_{anchor}_genotype_module_coverage.csv",index=False)
                status.append({**descriptor,"status":"completed","reason":""})
            except ValueError as e:
                status.append({**descriptor,"status":"ineligible","reason":str(e)})
    result=pd.concat(outcomes,ignore_index=True)
    result.to_csv(OUT/"tables/microglia_genotype_adjusted_sensitivity.csv",index=False)
    pd.DataFrame(status).to_csv(OUT/"tables/microglia_genotype_sensitivity_status.csv",index=False)
    primary=pd.read_csv(DATA/"inflammatory_genes.csv")
    primary=primary[(primary.subset=="primary")&(primary.variant=="primary")&(primary.anchor=="RHBDF2")
                    &(primary.population=="Microglia")&primary.gene.eq("TNFAIP3")&primary.cohort.isin(["GSE243292","GSE237718"])]
    primary=primary.assign(model="Pathology and depth")
    selected=result[result.anchor.eq("RHBDF2")&result.gene.eq("TNFAIP3")].assign(model="Plus available genotypes")
    plot=pd.concat([primary,selected],ignore_index=True).sort_values(["cohort","model"]).reset_index(drop=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"pdf.fonttype":42})
    fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained")
    for i,row in plot.iterrows():
        color="#277c82" if row.model.startswith("Plus") else "#a14d36"
        ax.plot([row.ci95_low,row.ci95_high],[i,i],color=color)
        ax.scatter(row.effect,i,color=color,s=35)
    labels=plot.apply(lambda r:f"{r.cohort} | {r.model} (FDR {r.qval:.3g})",axis=1)
    ax.set_yticks(plot.index,labels);ax.invert_yaxis();ax.axvline(0,color="gray",ls="--")
    ax.set_xlabel("TNFAIP3 coefficient and HC3 95% CI");ax.set_title("Genotype-adjusted microglial sensitivity")
    fig.savefig(OUT/"figures/Figure_S16_genotype_sensitivity.png",dpi=400,bbox_inches="tight")
    fig.savefig(OUT/"figures/Figure_S16_genotype_sensitivity.pdf",bbox_inches="tight")
    plt.close(fig)
    t=result[result.gene.eq("TNFAIP3")&result.anchor.eq("RHBDF2")]
    (OUT/"genotype_sensitivity_summary.json").write_text(json.dumps(t.to_dict("records"),indent=2)+"\n")
    print(t[["cohort","n_donors","effect","ci95_low","ci95_high","qval"]].to_string(index=False))


if __name__=="__main__":main()
