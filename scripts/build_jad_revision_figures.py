"""Create data-linked publication figures and reconciliation tables."""

from pathlib import Path
import json
import os
from itertools import combinations
import sys
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.tnf_cross_species_utils import read_one_to_one_orthologs
from statsmodels.stats.multitest import multipletests
DATA=Path(os.getenv("RHBDF2_JAD_RESULTS_DIR",str(ROOT/"results/JAD_revision_2026-10-12"))).expanduser().resolve()
OUT=Path(os.getenv("RHBDF2_JAD_REPORT_DIR",str(ROOT/"report/jad_revised_manuscript_2026-10-12"))).expanduser().resolve()
FIG=OUT/"figures"
TABLE=OUT/"tables"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"axes.titlesize":14,
                     "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none"})
COLORS={"Microglia":"#a14d36","Astrocytes":"#277c82","Endothelial cells":"#c49836"}


def read(name):
    return pd.read_csv(DATA/f"{name}.csv")


def save(fig,name):
    fig.savefig(FIG/f"{name}.png",dpi=400,bbox_inches="tight",facecolor="white")
    fig.savefig(FIG/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)


def primary(frame,anchor="RHBDF2",human=True):
    selected=frame[(frame.subset=="primary")&(frame.variant=="primary")&(frame.anchor==anchor)]
    if human: selected=selected[~selected.cohort.isin(["GSE148405","SEAAD_outputs"])]
    return selected.copy()


def heatmap(frame,row,filename,title,limit=50):
    if frame.empty:
        fig,ax=plt.subplots(figsize=(8,3)); ax.axis("off")
        ax.text(.5,.5,"No qualifying rows under the stated rule",ha="center",va="center",transform=ax.transAxes)
        ax.set_title(title);save(fig,filename);return
    frame=frame.copy()
    frame["context"]=frame.cohort+np.where(frame.region.eq("all"),"",": "+frame.region)
    frame["label"]=frame.population+" | "+frame[row].map(lambda s:str(s).replace("_"," ") if row=="module" else s)
    labels=frame.groupby("label").qval.min().sort_values().head(limit).index
    frame=frame[frame.label.isin(labels)]
    if len(labels)>18 and filename.startswith(("Figure_S1","Figure_S2")):
        for start in range(0,len(labels),18):
            selected=labels[start:start+18]
            heatmap(frame[frame.label.isin(selected)],row,f"{filename}_part{start//18+1}",title+f" part {start//18+1}",limit=18)
    frame["signed_evidence"]=np.sign(frame.effect)*(-np.log10(frame.qval.clip(lower=1e-12)))
    matrix=frame.pivot_table(index="label",columns="context",values="signed_evidence",aggfunc="first")
    q=frame.pivot_table(index="label",columns="context",values="qval",aggfunc="first").reindex_like(matrix)
    marks=q.map(lambda value:"*" if value<.05 else "")
    fig,ax=plt.subplots(figsize=(9,max(3,.34*len(matrix)+2.3)),layout="constrained")
    sns.heatmap(matrix,ax=ax,cmap="BrBG_r",center=0,vmin=-3,vmax=3,annot=marks,fmt="",linewidths=.35,
                cbar_kws={"label":"sign(effect) × −log10(FDR)","shrink":.55})
    ax.set_title(title);ax.set_xlabel("");ax.set_ylabel("");ax.tick_params(axis="y",labelsize=13)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha="right",fontsize=12)
    save(fig,filename)


def forest(frame,label,filename,title):
    if frame.empty: return
    frame=frame.copy().reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(8.2,max(3,.42*len(frame)+1.6)),layout="constrained")
    for i,row in frame.iterrows():
        color=COLORS.get(row.get("population"),"#277c82") if row.qval<.05 else "#777777"
        ax.plot([row.ci95_low,row.ci95_high],[i,i],color=color,lw=1.6)
        ax.scatter(row.effect,i,color=color,s=30,zorder=3)
    ax.set_yticks(np.arange(len(frame)),frame[label]);ax.invert_yaxis();ax.axvline(0,color="#aaaaaa",ls="--",lw=1)
    ax.set_title(title);ax.set_xlabel("Coefficient and HC3 95% confidence interval")
    save(fig,filename)


def focused_dotplots():
    """Replot descriptive summaries with separate color and detection legends."""
    from matplotlib.colors import Normalize
    from scripts import build_ad_supporting_materials as focused

    specifications=[
        ("GSE243292","GSE243292_Microglia_GEO.h5ad","microglia_subpopulations",
         ["Homeostatic","Motile","ARM","Dystrophic","Oli_microglia","Ast_microglia","Ex_microglia"],
         ["Homeostatic","Motile","ARM","Dystrophic","Oli-like","Ast-like","Ex-like"],10),
        ("GSE146639","GSE146639_barcoded_smartseq2_microglia.h5ad","diagnosis",
         ["CTR","AD"],["Control","Alzheimer disease"],13)]
    for cohort,filename,column,groups,labels,index in specifications:
        cache=TABLE/f"{cohort}_focused_dotplot_summaries.csv"
        if cache.exists():
            combined=pd.read_csv(cache)
        else:
            adata=focused.load_normalized(ROOT/"data/raw"/filename)
            adata.obs["RHBDF2_state"]=np.where(focused.dense_vector(adata,"RHBDF2")>0,"Detected","Undetected")
            first=focused.group_summary(adata,column,groups).assign(panel="first")
            second=focused.group_summary(adata,"RHBDF2_state",["Detected","Undetected"]).assign(panel="state")
            combined=pd.concat([first,second],ignore_index=True)
            combined.to_csv(cache,index=False)
        first=combined[combined.panel.eq("first")]
        second=combined[combined.panel.eq("state")]
        norm=Normalize(0,max(first["mean"].quantile(.98),second["mean"].quantile(.98)))
        fig,axes=plt.subplots(2,1,figsize=(12,8.3))
        fig.subplots_adjust(left=.15,right=.80,top=.90,bottom=.11,hspace=.62)
        label="microglial subpopulation" if column=="microglia_subpopulations" else "diagnosis"
        focused.draw_dotplot(axes[0],first,groups,labels,f"A  Candidate panel by {label}",norm)
        focused.draw_dotplot(axes[1],second,["Detected","Undetected"],
                             ["RHBDF2 detected","RHBDF2 undetected"],
                             "B  Candidate panel by operational RHBDF2 state",norm)
        cax=fig.add_axes([.85,.60,.018,.24])
        colorbar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap="YlOrRd"),cax=cax)
        colorbar.set_label("Mean log1p expression",fontsize=9)
        lax=fig.add_axes([.845,.13,.14,.28]);lax.axis("off")
        lax.set_xlim(0,1);lax.set_ylim(0,1)
        for i,fraction in enumerate([.1,.3,.5,.8]):
            y=.82-i*.22
            lax.scatter(.18,y,s=12+fraction*560,color="#9ba7a8")
            lax.text(.45,y,f"{fraction:.0%}",va="center",fontsize=9)
        lax.text(.02,1.02,"Detected cells",fontsize=10,fontweight="bold")
        fig.suptitle(f"{cohort} candidate-panel expression",fontsize=15,fontweight="bold")
        save(fig,f"Figure_S{index}_focused_microglia_descriptive")


def main():
    FIG.mkdir(parents=True,exist_ok=True);TABLE.mkdir(exist_ok=True)
    document_data={"registry":yaml.safe_load((ROOT/"configs/ad_external_cohorts.yaml").read_text()),
                   "inflammation":yaml.safe_load((ROOT/"configs/tnf_cross_species.yaml").read_text())}
    (OUT/"document_data_config.json").write_text(json.dumps(document_data,indent=2)+"\n")
    candidates=read("candidates");core=read("core_candidates");programs=read("programs")
    inflammatory=read("inflammatory_genes");modules=read("inflammatory_modules");status=read("status")
    profiles=[]
    for path in (DATA/"human").glob("*/*/*/primary/donor_qc_all.csv"):
        m=pd.read_csv(path)
        cohort,population,region=path.parts[-5:-2]
        for row in m.itertuples(index=False):
            profiles.append({"cohort":cohort,"population":population.replace("_"," "),"region":region,
                             "donor_id":str(row.donor_id),"n_cells":row.n_cells,
                             "rh_detection_fraction":getattr(row,"RHBDF2_detection_fraction",np.nan),
                             "condition":row.condition,"pass20":row.n_cells>=20,"pass50":row.n_cells>=50})
    profiles=pd.DataFrame(profiles)
    profiles.to_csv(TABLE/"donor_profile_eligibility.csv",index=False)
    eligibility=profiles.groupby(["cohort","population","region"]).agg(n_cells=("n_cells","sum"),
        n_donors=("donor_id","size"),donors_ge20=("pass20","sum"),donors_ge50=("pass50","sum"),
        median_donor_rh_detection=("rh_detection_fraction","median")).reset_index()
    eligibility.to_csv(TABLE/"population_eligibility.csv",index=False)
    human=eligibility[eligibility.cohort.ne("SEAAD_outputs")].copy()
    human["label"]=human.cohort+" | "+human.population+np.where(human.region.eq("all"),"",": "+human.region)
    display=human[human.population.isin(["Microglia","Astrocytes","Endothelial cells"])].copy()
    fig,ax=plt.subplots(figsize=(9, max(5,.32*len(display)+1.5)),layout="constrained")
    y=np.arange(len(display));ax.barh(y,display.n_donors,color="#dddddd",label="All donor profiles")
    ax.barh(y,display.donors_ge20,color="#277c82",label="At least 20 cells")
    ax.plot(display.donors_ge50,y,"o",color="#a14d36",ms=3,label="At least 50 cells")
    ax.set_yticks(y,display.label,fontsize=12);ax.invert_yaxis();ax.set_xlabel("Donors contributing to each population")
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.14),ncol=3,fontsize=10);ax.set_title("Donor representation after explicit cell-count QC")
    save(fig,"Figure_1_donor_QC")
    matrix=human.groupby(["population","cohort"]).median_donor_rh_detection.median().unstack()*100
    fig,ax=plt.subplots(figsize=(9,5),layout="constrained")
    sns.heatmap(matrix,annot=True,fmt=".2f",cmap="YlGnBu",ax=ax,cbar_kws={"label":"Median donor detection (%)"})
    ax.set_xlabel("");ax.set_ylabel("");ax.set_title("RHBDF2 detection is descriptive and capture-dependent")
    save(fig,"Figure_2_detection")
    replicated=read("replication_core_candidates")
    rep=replicated[(replicated.anchor=="RHBDF2") & replicated.replicated]
    selected=primary(core).merge(rep[["population","gene"]],on=["population","gene"])
    heatmap(selected,"gene","Figure_3_replicated_candidates","Robust non-self candidate replication")
    pr=read("replication_programs");pr=pr[(pr.anchor=="RHBDF2")&pr.replicated]
    p=primary(programs).merge(pr[["population","module"]],on=["population","module"])
    heatmap(p,"module","Figure_4_replicated_programs","Replicated transcript-derived program associations")
    corr=read("correlations")
    corr=corr[(corr.subset=="primary") & corr.population.isin(["Microglia","Endothelial cells"]) & corr.cohort.ne("SEAAD_outputs")]
    corr=corr[~(corr.cohort.eq("GSE243292")&corr.population.eq("Endothelial cells"))].copy()
    corr["qval_focused_seven"]=multipletests(corr.pval,method="fdr_bh")[1]
    corr["label"]=corr.cohort+" | "+corr.population
    fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained")
    y=np.arange(len(corr));ax.scatter(corr.unadjusted_spearman_r,y,label="Unadjusted Spearman",color="#a14d36",s=40)
    ax.scatter(corr.partial_rank_r,y,label="Depth/pathology-adjusted ranks",color="#277c82",s=40)
    for i,row in enumerate(corr.itertuples()):ax.plot([row.unadjusted_spearman_r,row.partial_rank_r],[i,i],color="#bbbbbb")
    ax.set_yticks(y,corr.label,fontsize=9);ax.axvline(0,color="#cccccc",ls="--");ax.set_xlim(-1,1)
    ax.set_xlabel("Rank correlation");ax.legend(loc="upper center",bbox_to_anchor=(.5,-.18),ncol=1,fontsize=10);ax.set_title("Anchor concordance attenuates after adjustment")
    save(fig,"Figure_5_adjusted_anchor_correlations")
    tnf=primary(inflammatory);tnf=tnf[(tnf.population=="Microglia")&(tnf.gene=="TNFAIP3")&tnf.cohort.ne("SEAAD_MTG")]
    tnf["label"]=tnf.cohort+" (n="+tnf.n_donors.astype(str)+", FDR="+tnf.qval.map(lambda q:f"{q:.3g}")+")"
    forest(tnf,"label","Figure_6_TNFAIP3_HC3","Human microglial TNFAIP3 association with RHBDF2")
    mouse=modules[(modules.cohort=="GSE148405")&(modules.region=="cortex")&(modules.subset=="primary")&(modules.variant=="primary")&(modules.anchor=="Rhbdf2")].copy()
    mouse["label"]=mouse.module.str.replace("_"," ")+" (FDR="+mouse.qval.map(lambda q:f"{q:.3g}")+")"
    forest(mouse,"label","Figure_7_mouse_cortex","QC-filtered mouse cortex RNA module associations")
    outputs=["TNF","TNFRSF1A","TNFRSF1B","IL6R","IL1R2","SELL","TREM2","CSF1R","MERTK","CX3CL1","JAM3","TGFA","HBEGF","AREG","EREG","IL6ST","CX3CR1","TYROBP","SYK","RELA","NFKBIA","EGFR","APP","CSF1"]
    output=primary(candidates,human=False)
    output=output[output.cohort.ne("GSE148405") & output.population.isin(["Microglia","Endothelial cells"]) & output.gene.isin(outputs) & output.cohort.ne("SEAAD_MTG")]
    output_display=output[output.gene.isin(["TNF","TNFRSF1A","TNFRSF1B","IL6R","TREM2","CSF1R","MERTK","TYROBP","SYK"])]
    heatmap(output_display,"gene","Figure_8_ADAM17_output_context","RHBDF2 associations with selected substrate and readout genes",limit=24)
    heatmap(primary(core),"gene","Figure_S1_all_candidates","Full robust candidate-panel associations",limit=70)
    heatmap(primary(programs),"module","Figure_S2_all_programs","Full transcript-derived program associations",limit=80)
    astro=core[(core.cohort=="GSE268599")&(core.variant=="primary")&(core.anchor=="RHBDF2")&core.gene.isin(["FCER1G","TREM2","TYROBP","SYK","C3"])].copy()
    astro["label"]=astro.gene+" | "+astro.subset.str.replace("_"," ")
    for region,frame in astro.groupby("region"):
        forest(frame,"label",f"Figure_S3_astrocyte_selection_{region}",f"Astrocyte selection sensitivity in {region}")
    robust=core[(core.anchor=="RHBDF2")&(core.subset=="primary")&(core.variant.isin(["primary","min50","age_sex","OLS"]))].copy()
    grouped=robust.groupby(["cohort","variant"]).qval.apply(lambda q:(q<.05).sum()).unstack().fillna(0)
    fig,ax=plt.subplots(figsize=(10,4.5),layout="constrained");grouped.plot.bar(ax=ax,width=.75)
    ax.set_ylabel("Non-self candidate FDR rows");ax.set_xlabel("");ax.tick_params(axis="x",rotation=25)
    ax.set_title("Sensitivity branches change eligibility and precision");ax.legend(title="Model branch",fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.35),ncol=4)
    save(fig,"Figure_S4_model_sensitivity")
    qc=pd.read_csv(DATA/"mouse/cell_qc.csv")
    cortex=qc[qc.region=="cortex"].groupby("mouse_id").agg(before=("region","size"),after=("primary_qc_pass","sum"))
    fig,ax=plt.subplots(figsize=(9,4),layout="constrained");cortex.plot.bar(ax=ax,color=["#cccccc","#277c82"])
    ax.set_ylabel("Cells");ax.set_xlabel("");ax.tick_params(axis="x",rotation=25);ax.set_title("Mouse cortex cell QC by animal")
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.35),ncol=2)
    save(fig,"Figure_S5_mouse_cell_QC")
    focused_dotplots()
    gsea=pd.read_csv(DATA/"enrichment/hallmark_all_results.csv") if (DATA/"enrichment/hallmark_all_results.csv").exists() else pd.DataFrame()
    if not gsea.empty:
        gsea["FDR q-val"]=pd.to_numeric(gsea["FDR q-val"])
        gsea["NES"]=pd.to_numeric(gsea.NES)
        hallmark=gsea[(gsea.anchor=="RHBDF2") & (gsea.rank_order=="primary") & gsea.Term.str.contains("TNFA|TNF-alpha|INFLAMMATORY|INTERFERON",case=False)].copy()
        hallmark["label"]=hallmark.cohort+" | "+hallmark.population
        mat=hallmark.pivot_table(index="label",columns="Term",values="NES",aggfunc="first")
        fig,ax=plt.subplots(figsize=(10,max(3,.4*len(mat)+2)),layout="constrained")
        sns.heatmap(mat,ax=ax,cmap="BrBG_r",center=0,cbar_kws={"label":"Normalized enrichment score"})
        ax.set_xlabel("");ax.set_ylabel("");ax.set_xticklabels([t.get_text().replace("HALLMARK_","").replace("_"," ") for t in ax.get_xticklabels()],rotation=35,ha="right",fontsize=8)
        ax.set_title("Ranked Hallmark enrichment of donor-level HC3 statistics")
        save(fig,"Figure_S6_ranked_Hallmark")
        gsea.to_csv(TABLE/"ranked_Hallmark_all_results.csv",index=False)
    rep.to_csv(TABLE/"replicated_RHBDF2_core_candidates.csv",index=False)
    pr.to_csv(TABLE/"replicated_RHBDF2_programs.csv",index=False)
    tnf.to_csv(TABLE/"human_microglia_TNFAIP3.csv",index=False)
    output.to_csv(TABLE/"RHBDF2_ADAM17_output_context.csv",index=False)
    corr.to_csv(TABLE/"adjusted_anchor_correlations.csv",index=False)
    status.to_csv(TABLE/"all_model_status.csv",index=False)
    comparison=[]
    for family,item in [(read("replication_core_candidates"),"gene"),(read("replication_programs"),"module")]:
        eligible=family[family.replicated]
        for (population,label),group in eligible.groupby(["population",item]):
            anchors=group.anchor.unique()
            classification="shared" if len(anchors)==2 else anchors[0]+" associated only"
            comparison.append({"family":item,"population":population,"outcome":label,"classification":classification,
                               "anchors":";".join(anchors),"note":"association-level family-FDR replication, not mechanism"})
    pd.DataFrame(comparison).to_csv(TABLE/"dual_anchor_replicated_core_comparison.csv",index=False)
    for name in ["metadata_direct","metadata_interactions","inflammatory_genes","inflammatory_modules"]:
        read(name).to_csv(TABLE/f"{name}.csv",index=False)
    orthology=read_one_to_one_orthologs(ROOT/"data/external_mouse_cohorts/orthology/HOM_ProteinCoding.rpt")
    reverse=dict(zip(orthology.mouse_gene,orthology.human_gene))
    cross=[]
    for family,label in [(inflammatory,"gene"),(modules,"module")]:
        h=primary(family)
        h=h[(h.population=="Microglia")&h.cohort.isin(["GSE243292","GSE157827","GSE237718"])]
        m=family[(family.cohort=="GSE148405")&(family.subset=="primary")&(family.variant=="primary")&(family.region=="cortex")&(family.anchor=="Rhbdf2")].copy()
        for row in m.to_dict("records"):
            item=reverse.get(row[label],row[label]) if label=="gene" else row[label]
            hg=h[h[label]==item]
            concordant=hg[(hg.qval<.05)&(np.sign(hg.effect)==np.sign(row["effect"]))]
            cross.append({"family":label,"outcome":item,"human_same_direction_fdr_cohorts":";".join(concordant.cohort.unique()),
                          "n_human_same_direction_fdr_cohorts":concordant.cohort.nunique(),"mouse_effect":row["effect"],"mouse_qval":row["qval"],
                          "strict_cross_species_support":concordant.cohort.nunique()>=2 and row["qval"]<.05})
    pd.DataFrame(cross).to_csv(TABLE/"cross_species_corrected_comparison.csv",index=False)
    directory=DATA/"mouse/cortex/primary"
    m=pd.read_csv(directory/"donors_primary.csv")
    scores=pd.read_csv(directory/"scores_inflammatory_modules_Rhbdf2_primary.csv")
    disease=m.condition.eq("5xFAD").to_numpy()
    contrasts=[]
    for module in scores:
        values=scores[module].to_numpy()
        observed=values[disease].mean()-values[~disease].mean()
        allocations=list(combinations(range(len(values)),int(disease.sum())))
        differences=[]
        for allocation in allocations:
            mask=np.zeros(len(values),dtype=bool);mask[list(allocation)]=True
            differences.append(values[mask].mean()-values[~mask].mean())
        p=np.mean(np.abs(differences)>=abs(observed)-1e-12)
        contrasts.append({"module":module,"effect_FAD_minus_WT":observed,"pval":p,
                          "n_allocations":len(allocations),"n_FAD":int(disease.sum()),"n_WT":int((~disease).sum())})
    contrasts=pd.DataFrame(contrasts)
    contrasts["qval"]=multipletests(contrasts.pval,method="fdr_bh")[1]
    contrasts.to_csv(TABLE/"mouse_QC_genotype_module_contrasts.csv",index=False)
    from scripts.jad_detailed_figures import build_all
    build_all(DATA, OUT)
    summary={"replicated_RHBDF2_core_rows":len(rep),"replicated_RHBDF2_program_rows":len(pr),
             "completed_model_branches":int(status.status.eq("completed").sum()),"ineligible_model_branches":int(status.status.eq("ineligible").sum()),
             "TNFAIP3_full_microglia_positive_fdr_cohorts":int(((tnf.effect>0)&(tnf.qval<.05)).sum()),
             "mouse_cortex_cells_before":int(cortex.before.sum()),"mouse_cortex_cells_after":int(cortex.after.sum()),
             "mouse_cortex_Rhbdf2_module_FDR_hits":int((mouse.qval<.05).sum()),
             "strict_cross_species_support_rows":int(pd.DataFrame(cross).strict_cross_species_support.sum()),
             "astrocyte_replicated_genes":rep.loc[rep.population.eq("Astrocytes"),"gene"].tolist()}
    (OUT/"results_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
