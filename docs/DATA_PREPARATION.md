# Required Source Inputs

The code-only repository is not a research-data bundle. Acquire each source from its provider, retain accession/release, file size and SHA256, and inspect the prepared object's identifiers, count matrix, cell labels, donor labels, pathology, and regional metadata before analysis.

## Human Inputs

The corrected full run expects these local paths:

| Cohort/input | Required path | Preparation role |
|---|---|---|
| GSE243292 full brain | `data/raw/GSE243292_ADsnRNAseq_GEO.h5ad` | Author processed object; `sampleID`, `atscore`, `final_celltype` |
| GSE243292 astrocytes | `data/raw/GSE243292_Astrocyte_GEO.h5ad` | Author astrocyte labels and doublet exclusion |
| GSE243292 focused microglia | `data/raw/GSE243292_Microglia_GEO.h5ad` | Descriptive figures only |
| GSE146639 focused microglia | `data/raw/GSE146639_barcoded_smartseq2_microglia.h5ad` | Descriptive validation figures; prepare with `prepare_gse146639.py` |
| GSE157827 | `data/external_ad_cohorts/GSE157827/processed/GSE157827_canonical.h5ad` | Independent full-transcriptome donor cohort |
| GSE237718 | `data/external_ad_cohorts/GSE237718/processed/GSE237718_canonical.h5ad` | Independent full-transcriptome donor cohort |
| GSE268599 | `data/external_ad_cohorts/GSE268599/processed/GSE268599_canonical.h5ad` | Astrocyte targeted working matrix; strict selection and original full totals |
| SEA-AD primary panel | `data/external_ad_cohorts/SEAAD_MTG/processed/SEAAD_MTG_targeted_canonical.h5ad` | Targeted atlas panel; strict Micro-PVM labels |
| SEA-AD output panel | `data/external_ad_cohorts/SEAAD_MTG/processed/SEAAD_MTG_adam17_outputs_canonical.h5ad` | Expanded substrate/readout panel from the same donors |

Primary source records: [GSE243292](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243292), [GSE146639](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146639), [GSE157827](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE157827), [GSE237718](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE237718), [GSE268599](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE268599), and [SEA-AD](https://portal.brain-map.org/explore/seattle-alzheimers-disease).

`manage_ad_cohorts.py` downloads/prepares registered public cohorts. The expanded SEA-AD panel has a separate registry, `configs/seaad_adam17_outputs_registry.yaml`; pass that registry to the same cohort manager. Do not treat the two SEA-AD panels as independent replication. The primary panel requires the original `SEAAD_MTG_RNAseq_final-nuclei.2026-06-22.h5ad` to recover original UMI totals by nucleus identifier. A targeted matrix does not supply a valid normalization denominator by summing its selected genes.

The detailed clinical join additionally requires `data/external_ad_cohorts/SEAAD_MTG/source/sea-ad_cohort-donor-metadata_2025-07-25.xlsx`, obtained from the public SEA-AD resource, with sheet `SEA-AD_Cohort_Metadata`. Retain this exact workbook or document a deliberate release change. Run `python scripts/prepare_revision_inputs.py --seaad-metadata` to generate the local analysis-ready join. The broader donor-audit script also expects the GSE146639 supplementary workbook and other prepared objects; those are not necessary for this isolated SEA-AD join.

GSE138852 libraries pool donors and are not eligible for the corrected independent-donor inference. GSE269906 is retained as spatial support, not a primary full-transcriptome replication cohort. Controlled-access ROSMAP readiness code is present, but no ROSMAP result is part of this study.

## Mouse Inputs

[GSE148405](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE148405) supplies the full assigned-count matrix and GEO sample metadata. The configured MGI one-to-one orthology report maps the human panels to mouse genes. `python scripts/prepare_revision_inputs.py --mouse --download` downloads missing public sources, streams the original counts to the requested gene panel, preserves non-ERCC full-library totals, and writes the source-count QC table expected by the corrected runner.

Mouse QC must use the full source matrix to count detected genes; the targeted H5AD alone cannot provide that statistic. The output is `report/jad_submission_audit_2026-10-12/mouse_source_cell_qc_flags.csv`. The subsequent runner applies the configured gates. Assigned Smart-seq2 counts are not UMIs or the author's aligned-read statistic. Four WT and four 5xFAD animals in cortex are the independent units; same-animal hippocampus/cerebellum observations are sensitivity contexts.

## External Layers

UniProt, Reactome, OmniPath, GWAS Catalog, Open Targets, HPA, Expression Atlas, enrichment libraries, and the optional expanded context resources are queried by their respective modules. Queries can change as providers update data. Freeze responses, access dates, library releases and hashes locally. Missing access, failed queries, or empty results must not be converted into biological absence. No database export or unmatched proteomics table is redistributed here. The historical contextual bundle and exact frozen analytical outputs must be obtained separately from the authors when reproducing the complete manuscript, rather than substituting a newer API response.
