# RHBDF2 and ADAM17 Transcriptomic Associations in Alzheimer Disease

Python analysis code for donor-aware expression associations across human brain cell populations, inflammatory transcript modules, and a human-mouse comparison. Human cohorts are analyzed separately; brain regions and duplicate gene panels from the same donors are not counted as independent studies.

This repository contains code, configurations, tests, and documentation only. It does not redistribute individual-donor tables, count matrices, H5AD objects, proteomics files, manuscripts, figures, or analytical results. Public source identifiers and download locations are recorded in the cohort registries. Manuscript-authoring and local packaging tools are not part of the statistical analysis distribution.

## Installation

The publication copy was tested with Python 3.13. Create an environment inside a fresh clone:

```bash
git clone https://github.com/kelokely/rhbdf2-ad-transcriptomics.git
cd rhbdf2-ad-transcriptomics
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt -c constraints-tested.txt
python -m pip check
python -m pytest -q
```

`requirements-core.txt` contains the basic processed-H5AD analysis dependencies. `requirements-full.txt` adds enrichment, public-resource access, donor models, spreadsheet reading, and plotting dependencies. `requirements-dev.txt` installs the full set plus pytest. `constraints-tested.txt` pins directly used packages from the tested environment, not every transitive dependency. Controlled-access Synapse support is separate in `requirements-controlled-access.txt` and is not needed for the public study; ROSMAP was not analyzed.

## Repository Structure

```text
rhbdf2-ad-transcriptomics/
  run_pipeline.py                 processed-H5AD exploratory analysis
  run_jad_revision.py             corrected donor-level human/mouse analysis
  run_multicell_ad_extension.py    separate cell-population analysis
  run_committee_extensions.py     myeloid, endothelial, dual-anchor extensions
  manage_ad_cohorts.py            public-cohort download/preparation/readiness
  run_tnf_cross_species.py        inflammatory modules and mouse preparation
  src/                           input, metadata, statistics, evidence utilities
  configs/                       cohorts, genes, modules, QC, analysis settings
  scripts/                       input preparation, figures, numerical checks
  tests/                         unit and regression tests
  docs/                          data preparation and interpretation guidance
```

The reusable utilities include continuous, hurdle, cell-level DE, state assignment, paired pseudobulk, enrichment, protein-context, genetics, brain-context, spatial, and ligand/receptor components. Their availability in code does not mean every optional component contributes primary evidence in the manuscript. Historical exploratory entry points are retained for reproducibility and are distinct from the corrected donor-level analysis.

## Source Data Preparation

Read [the source-data guide](docs/DATA_PREPARATION.md) before starting the complete multi-cohort analysis. The repository does not contain the large objects or external donor workbook required for that analysis. Running tests requires no research data.

For one independently registered cohort:

```bash
python manage_ad_cohorts.py --registry configs/ad_external_cohorts.yaml status
python manage_ad_cohorts.py --registry configs/ad_external_cohorts.yaml download --cohort GSE157827
python manage_ad_cohorts.py --registry configs/ad_external_cohorts.yaml prepare --cohort GSE157827
python manage_ad_cohorts.py --registry configs/ad_external_cohorts.yaml inspect --cohort GSE157827
```

Repeat download and preparation for the other eligible public cohorts. The registry records file sizes and releases. Keep downloaded sources and hashes; do not silently substitute a newer atlas release. Several downloads exceed a gigabyte, and the full SEA-AD object is approximately 33 GB. Readiness is not guaranteed merely because a download completes.

Prepare the source-count mouse QC table and the SEA-AD donor metadata join once the documented inputs are present:

```bash
python scripts/prepare_revision_inputs.py --mouse --download
python scripts/prepare_revision_inputs.py --seaad-metadata
```

## Run the Corrected Analysis

After all required inputs are prepared:

```bash
python run_jad_revision.py --config configs/jad_revision.yaml \
  --output-dir results/AD_fresh_run --report-dir report/AD_fresh_run
```

`--config` selects frozen genes, modules, cohort registries, QC gates, covariates, and enrichment settings. A new `--output-dir` prevents replacement of an earlier acceptance run. `--report-dir` holds genotype tables and figures. Default `--stage all` runs human, mouse, synthesis, genotype sensitivity, and optional enrichment. Other stages are `human`, `mouse`, `metadata`, `genotype`, and `enrichment`. A human-only run does not require mouse input files; a mouse-only run still needs the source-count QC table. Metadata and genotype refreshes require existing donor aggregates.

The primary human model relates donor log1p CPM abundance to continuous log1p anchor detection per 1,000 cells, adjusting for library depth and pathology, with HC3 robust uncertainty. It is not an edgeR/DESeq2 negative-binomial model. The main gates are at least 20 cells per donor, eight donors, 500 population cells, 50 anchor detections, five donors with detection, and five residual degrees of freedom. Mouse gates and sensitivity branches differ and are explicit in the config. Full-library denominators must never be replaced by sums over a targeted panel.

For exploratory analysis of one processed microglia object:

```bash
python run_pipeline.py --config configs/gse243292.yaml
```

Do not interpret this cell-level high/low analysis as the corrected donor-level model. The guide describes how to distinguish them.

## Figures and Validation

After a complete corrected human/mouse run, including genotype sensitivity:

```bash
export RHBDF2_JAD_RESULTS_DIR="$PWD/results/AD_fresh_run"
export RHBDF2_JAD_REPORT_DIR="$PWD/report/AD_fresh_run"
python scripts/build_jad_revision_figures.py
python scripts/validate_jad_revision.py
```

The figure builder requires the complete analytical bundle and the original focused GSE243292/GSE146639 objects or their cached descriptive summaries. It does not synthesize missing observations. The validator reproduces numerical estimates, checks FDR families, and runs the tests. Historical source-manuscript integrity is assessed only when that document is supplied separately; it is not a code-only validation requirement. The analysis directory and original reference artifacts are excluded from this repository. Changing a config or input can change the expected acceptance checks; this validator is study-specific, not a generic validation certificate for arbitrary datasets.

Detailed figures combine donor observations, distributions, heatmaps, numeric estimates, and compact confidence intervals. Plotted observations are exported beside figures. Scientific estimates are checked against saved models; horizontal jitter is display-only.

## Reading the Results

- `status.csv` and `run_status.json` distinguish completed, ineligible, and failed work. Inspect `genotype_status.json` and `enrichment_status.json` separately.
- `core_candidates.csv` and `inflammatory_genes.csv` contain coefficients, confidence intervals, nominal P, family-specific FDR, and exact cohort/population/region/anchor/subset/model labels.
- `replication_core_candidates.csv` and `replication_programs.csv` count independent same-direction cohort support, excluding self-associations and duplicate atlas panels.
- `programs.csv`, `inflammatory_modules.csv`, and per-context coverage files describe RNA program scores and measured members, not protein activity.
- `correlations.csv` distinguishes raw from depth/pathology-adjusted detection concordance.
- `metadata_direct.csv` and `metadata_interactions.csv` describe direct associations and slope modification, not longitudinal disease progression.
- `mouse/cell_qc.csv` and animal/region donor aggregates retain mouse eligibility and independent-animal structure.
- `enrichment/hallmark_all_results.csv` supplies ranked gene-set enrichment, its FDR, and tie-order sensitivity.

Candidate-panel FDR and transcriptome-wide FDR are different testing families. An untested gene, an ineligible model, and a null result are not interchangeable. Independent replication and donor uncertainty matter more than the visual intensity of a heatmap.

## Scientific Scope

RNA associations and pathway scores support reproducibility and follow-up prioritization. They do not measure iRhom2 protein abundance, mature/surface ADAM17, TNF release, substrate shedding, phosphoprotein activity, or treatment benefit. Unmatched proteomics, interaction/PTM annotations, genetics, and spatial context are supporting evidence rather than causal validation. Lineage-discordant immune signatures require identity/admixture follow-up. A null mouse or endothelial finding remains part of the evidence.

## Reuse and Citation

Repository: https://github.com/kelokely/rhbdf2-ad-transcriptomics

Use the exact commit identifier when documenting a reproduction. A DOI and final manuscript citation are not yet assigned. No software license is included; the owner has not yet selected reuse terms. Database downloads remain subject to their providers' access and reuse conditions. Do not upload individual-donor metadata or count matrices to this code-only repository.
