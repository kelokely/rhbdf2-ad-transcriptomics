"""Independent statistical checks, focused metadata estimates, and output integrity."""

from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_jad_revision import clean_ineligible

DATA=Path(os.getenv("RHBDF2_JAD_RESULTS_DIR",str(ROOT/"results/JAD_revision_2026-10-12"))).expanduser().resolve()
OUT=Path(os.getenv("RHBDF2_JAD_REPORT_DIR",str(ROOT/"report/jad_revised_manuscript_2026-10-12"))).expanduser().resolve()


def metadata_followup():
    m=pd.read_csv(DATA/"human/SEAAD_MTG/Microglia/all/primary/donors_primary.csv")
    direct=pd.read_csv(DATA/"metadata_direct.csv")
    selected=direct[(direct.cohort=="SEAAD_MTG")&(direct.population=="Microglia")&(direct.subset=="primary")
                   &(direct.variant=="primary")&(direct.anchor=="RHBDF2")&(direct.qval_study_family<.05)]
    rows=[]
    for row in selected.itertuples(index=False):
        field=row.field
        raw=m[field];numeric=pd.to_numeric(raw,errors="coerce")
        numeric_factor=raw.nunique()>5 and numeric.notna().sum()==raw.notna().sum()
        for variant in ["primary","reported_age_sex","public_numeric_age_sex"]:
            mask=numeric.notna() if numeric_factor else raw.notna()
            age=pd.to_numeric(m.age_at_death,errors="coerce")
            if variant=="public_numeric_age_sex":age=pd.to_numeric(m.age_at_death_public,errors="coerce")
            if variant!="primary":mask &= age.notna() & m.sex.notna()
            values=m.loc[mask].reset_index(drop=True)
            factor_raw=raw.loc[mask].reset_index(drop=True)
            y=np.log1p(1000*values.RHBDF2_detection_fraction)
            depth=np.log1p(values.library_counts/values.n_cells)
            x=pd.DataFrame({"intercept":1.,"depth":depth-depth.mean()})
            if numeric_factor:
                factor=pd.DataFrame({field:pd.to_numeric(factor_raw)-pd.to_numeric(factor_raw).mean()})
                baseline="per one score unit"
            else:
                orders={"condition":["Not AD","Low","Intermediate","High"],
                        "cerad_score":["Absent","Sparse","Moderate","Frequent"],
                        "cognitive_status":["No dementia","Dementia"]}
                levels=[v for v in orders.get(field,sorted(factor_raw.unique())) if v in factor_raw.unique()]
                if set(levels)!=set(factor_raw.unique()):levels=sorted(factor_raw.unique())
                factor=pd.get_dummies(pd.Categorical(factor_raw,categories=levels),prefix=field,drop_first=True,dtype=float)
                baseline=levels[0]
            x=pd.concat([x,factor],axis=1)
            if variant!="primary":
                x["age"]=age.loc[mask].to_numpy()-age.loc[mask].mean()
                x=pd.concat([x,pd.get_dummies(values.sex,prefix="sex",drop_first=True,dtype=float)],axis=1)
            if len(x)-x.shape[1]<5 or np.linalg.matrix_rank(x)!=x.shape[1]:continue
            model=sm.OLS(y,x).fit(cov_type="HC3",use_t=True)
            restriction=np.zeros((factor.shape[1],x.shape[1]))
            for i,term in enumerate(factor):restriction[i,x.columns.get_loc(term)]=1
            p=float(model.wald_test(restriction,use_f=True,scalar=True).pvalue)
            if variant=="primary" and not np.isclose(p,row.pval,rtol=1e-6,atol=1e-12):
                raise ValueError(f"Primary metadata test did not independently reproduce: {field}")
            for term in factor:
                ci=model.conf_int().loc[term]
                rows.append({"field":field,"variant":variant,"term":term,"reference":baseline,"n_donors":len(x),
                             "effect":model.params[term],"ci95_low":ci.iloc[0],"ci95_high":ci.iloc[1],
                             "joint_pval":p,"primary_study_family_qval":row.qval_study_family if variant=="primary" else np.nan,
                             "response":"log1p RHBDF2 detection per 1000 cells", "followup_selection":"post hoc follow-up of primary FDR hits"})
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/"tables/SEAAD_myeloid_metadata_effects_and_age_sensitivity.csv",index=False)
    groups=[]
    for field in ["condition","cerad_score","cognitive_status"]:
        for group,v in m.groupby(field,observed=True):
            groups.append({"field":field,"level":group,"n_donors":len(v),"mean_detection_percent":v.RHBDF2_detection_fraction.mean()*100,
                           "median_detection_percent":v.RHBDF2_detection_fraction.median()*100})
    pd.DataFrame(groups).to_csv(OUT/"tables/SEAAD_myeloid_metadata_group_detection.csv",index=False)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":13,"pdf.fonttype":42})
    fig,axes=plt.subplots(2,2,figsize=(9,7.5),layout="constrained")
    rng=np.random.default_rng(20261012)
    for ax,field,title in zip(axes.flat,["condition","cerad_score","cognitive_status","last_mmse_score"],
                              ["Neuropathology category","CERAD category","Cognitive status","Last MMSE score"]):
        if field=="last_mmse_score":
            numeric=pd.to_numeric(m[field],errors="coerce")
            ax.scatter(numeric,m.RHBDF2_detection_fraction*100,color="#277c82",s=25,alpha=.7)
        else:
            levels={"condition":["Not AD","Low","Intermediate","High"],"cerad_score":["Absent","Sparse","Moderate","Frequent"],
                    "cognitive_status":["No dementia","Dementia"]}[field]
            for i,level in enumerate(levels):
                value=m.loc[m[field].eq(level),"RHBDF2_detection_fraction"]*100
                ax.scatter(i+rng.uniform(-.1,.1,len(value)),value,color="#277c82",s=20,alpha=.7)
                if len(value):ax.plot([i-.2,i+.2],[value.mean()]*2,color="#a14d36",lw=2)
            ax.set_xticks(range(len(levels)),levels,rotation=15,ha="right")
        ax.set_title(title);ax.set_ylabel("RHBDF2-detected nuclei (%)")
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUT/"figures/Figure_S15_SEAAD_myeloid_metadata.png",dpi=400,bbox_inches="tight")
    fig.savefig(OUT/"figures/Figure_S15_SEAAD_myeloid_metadata.pdf",bbox_inches="tight")
    plt.close(fig)
    print(frame.to_string(index=False))


def main():
    checks=[]
    def check(name,passed,detail=""):
        checks.append({"check":name,"passed":bool(passed),"detail":detail})
        if not passed:raise ValueError(name+": "+detail)
    reference=Path(os.getenv("RHBDF2_JAD_REFERENCE_DIR",str(ROOT.parent/"RHBDF2_AD_student_transfer_2026-07-25/01_MANUSCRIPT")))
    original=reference/"RHBDF2_AD_manuscript_with_TNF_mouse_extension.docx"
    if original.is_file():
        check("Original manuscript unchanged",hashlib.sha256(original.read_bytes()).hexdigest()=="471c0f25aa34fad999166c97f92fda059ec7cef513dec4b141c191f82153508b")
    else:
        checks.append({"check":"Original manuscript integrity","passed":None,
                       "detail":"Not assessed: source document is outside this code-only repository"})
    status=pd.read_csv(DATA/"status.csv")
    for row in status[(status.status=="ineligible")&status.cohort.ne("GSE148405")].itertuples(index=False):
        directory=DATA/"human"/row.cohort/row.population.replace(" ","_")/row.region/row.subset
        clean_ineligible(directory,row.anchor,row.variant)
    for name in ["core_candidates","candidates","programs","inflammatory_genes","inflammatory_modules"]:
        frame=pd.read_csv(DATA/f"{name}.csv")
        grouping=["cohort","subset","variant","anchor"]
        if name.startswith("inflammatory"):grouping += ["population","region"]
        for _,group in frame.groupby(grouping):
            expected=multipletests(group.pval,method="fdr_bh")[1]
            check(f"{name} FDR family",np.allclose(expected,group.qval,atol=1e-12))
        check(f"{name} finite estimates",np.isfinite(frame.effect).all() and np.isfinite(frame.standard_error).all())
        check(f"{name} no anchor self results",(frame.gene.ne(frame.anchor).all() if "gene" in frame else True))
    direct=pd.read_csv(DATA/"inflammatory_genes.csv")
    for cohort in ["GSE243292","GSE157827","GSE237718"]:
        directory=DATA/"human"/cohort/"Microglia/all/primary"
        m=pd.read_csv(directory/"donors_primary.csv")
        all_meta=pd.read_csv(directory/"donor_qc_all.csv")
        c=pd.read_csv(directory/"donor_counts.csv.gz",usecols=["TNFAIP3"])
        c=c.loc[all_meta.n_cells.ge(20)].reset_index(drop=True)
        y=np.log1p(1e6*c.TNFAIP3/m.library_counts)
        x=pd.DataFrame({"intercept":1.,"anchor":np.log1p(1000*m.RHBDF2_detection_fraction),
                        "depth":np.log1p(m.library_counts/m.n_cells)})
        x=pd.concat([x,pd.get_dummies(m.condition,drop_first=True,dtype=float)],axis=1)
        model=sm.OLS(y,x).fit(cov_type="HC3",use_t=True)
        r=direct[(direct.cohort==cohort)&(direct.population=="Microglia")&(direct.subset=="primary")&(direct.variant=="primary")
                 &(direct.anchor=="RHBDF2")&(direct.gene=="TNFAIP3")].iloc[0]
        check(f"{cohort} TNFAIP3 independently reproduces",np.allclose([r.effect,r.standard_error,r.pval],
              [model.params['anchor'],model.bse['anchor'],model.pvalues['anchor']],rtol=1e-7,atol=1e-10))
    qc=pd.read_csv(DATA/"mouse/cell_qc.csv")
    check("Mouse cortex retains exactly 359 cells",qc.loc[qc.region.eq("cortex"),"primary_qc_pass"].sum()==359)
    check("Eight independent cortex animals retained",qc.loc[qc.region.eq("cortex")&qc.primary_qc_pass,"mouse_id"].nunique()==8)
    m=pd.read_csv(DATA/"human/SEAAD_MTG/Microglia/all/primary/donor_qc_all.csv")
    check("SEA-AD strict myeloid recovery",m.n_cells.sum()==35907 and len(m)==84)
    check("Primary human completed models satisfy donor QC",status[(status.status=="completed")&status.cohort.ne("GSE148405")].minimum_cells.ge(20).all())
    genetic=pd.read_csv(OUT/"tables/microglia_genotype_adjusted_sensitivity.csv")
    for cohort,fields in [("GSE243292",["apoe","trem2"]),("GSE237718",["apoe_genotype"])]:
        directory=DATA/"human"/cohort/"Microglia/all/primary"
        m=pd.read_csv(directory/"donors_primary.csv")
        all_m=pd.read_csv(directory/"donor_qc_all.csv")
        counts=pd.read_csv(directory/"donor_counts.csv.gz",usecols=["TNFAIP3"])
        counts=counts.loc[all_m.n_cells.ge(20)].reset_index(drop=True)
        x=pd.DataFrame({"intercept":1.,"anchor":np.log1p(1000*m.RHBDF2_detection_fraction),"depth":np.log1p(m.library_counts/m.n_cells)})
        x=pd.concat([x,pd.get_dummies(m.condition,drop_first=True,dtype=float)],axis=1)
        for field in fields:x=pd.concat([x,pd.get_dummies(m[field],prefix=field,drop_first=True,dtype=float)],axis=1)
        model=sm.OLS(np.log1p(1e6*counts.TNFAIP3/m.library_counts),x).fit(cov_type="HC3",use_t=True)
        row=genetic[(genetic.cohort==cohort)&genetic.anchor.eq("RHBDF2")&genetic.gene.eq("TNFAIP3")].iloc[0]
        check(f"{cohort} genotype-adjusted TNFAIP3 independently reproduces",np.allclose([row.effect,row.standard_error,row.pval],
              [model.params['anchor'],model.bse['anchor'],model.pvalues['anchor']],rtol=1e-7,atol=1e-10))
    metadata_followup()
    pd.DataFrame(checks).to_csv(OUT/"tables/independent_validation_checks.csv",index=False)
    hashes=[]
    for directory in [DATA,OUT/"tables"]:
        for path in directory.glob("*.csv"):
            hashes.append({"file":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size})
    pd.DataFrame(hashes).to_csv(OUT/"tables/table_sha256_manifest.csv",index=False)
    tests=subprocess.run([sys.executable,"-m","pytest","-q"],cwd=ROOT,capture_output=True,text=True,timeout=120)
    passed=re.search(r"(\d+) passed",tests.stdout)
    if tests.returncode or passed is None:
        raise ValueError("Unit test verification failed:\n"+tests.stdout+tests.stderr)
    verified=[row for row in checks if row["passed"] is not None]
    (OUT/"validation_status.json").write_text(json.dumps({"status":"passed","checks":len(verified),
        "not_assessed":len(checks)-len(verified),
        "unit_tests":int(passed.group(1)),"unit_test_execution":"python -m pytest -q"},indent=2)+"\n")
    print(f"Passed {len(verified)} independent checks; {len(checks)-len(verified)} not assessed")


if __name__=="__main__":main()
