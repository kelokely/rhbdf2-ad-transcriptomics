from __future__ import annotations

import numpy as np
import pandas as pd

from run_committee_extensions import adam17_output_gene_table, summarize_adam17_outputs


def test_adam17_output_summary_preserves_replication_and_stability(tmp_path):
    config = {
        "adam17_output_families": {
            "inflammatory_shedding": {
                "genes": ["TNF", "IL6R"],
                "evidence_class": "curated_substrate",
            }
        }
    }
    genes = adam17_output_gene_table(config)
    assert genes["gene"].tolist() == ["TNF", "IL6R"]

    candidate_path = tmp_path / "candidates.csv"
    pd.DataFrame({"names": ["TNF", "IL6R"]}).to_csv(candidate_path, index=False)
    manifest = pd.DataFrame([
        {
            "cohort": "cohort_a",
            "population": "Microglia",
            "anchor_gene": "RHBDF2",
            "matrix_scope": "full_transcriptome",
            "donor_status": "completed",
            "donor_candidate_results": str(candidate_path),
        },
        {
            "cohort": "cohort_b",
            "population": "Microglia",
            "anchor_gene": "RHBDF2",
            "matrix_scope": "full_transcriptome",
            "donor_status": "completed",
            "donor_candidate_results": str(candidate_path),
        },
    ])
    pd.DataFrame([
        {"cohort": "cohort_a", "population": "Microglia", "anchor_gene": "RHBDF2", "names": "TNF", "effect": 0.4, "pvals": 0.001, "pvals_adj": 0.01, "self_association": False},
        {"cohort": "cohort_b", "population": "Microglia", "anchor_gene": "RHBDF2", "names": "TNF", "effect": 0.2, "pvals": 0.002, "pvals_adj": 0.02, "self_association": False},
        {"cohort": "cohort_a", "population": "Microglia", "anchor_gene": "RHBDF2", "names": "IL6R", "effect": -0.1, "pvals": 0.2, "pvals_adj": 0.4, "self_association": False},
        {"cohort": "cohort_b", "population": "Microglia", "anchor_gene": "RHBDF2", "names": "IL6R", "effect": 0.1, "pvals": 0.3, "pvals_adj": 0.5, "self_association": False},
    ]).to_csv(tmp_path / "cross_cohort_candidate_results_long.csv", index=False)
    sensitivity = pd.DataFrame([
        {
            "excluded_subpopulation": "state_1",
            "names": "TNF",
            "anchor_gene": "RHBDF2",
            "pvals_adj": 0.03,
            "effect_per_log1p_rhbdf2_detection_per_1000_cells": 0.4,
            "effect_per_log1p_adam17_detection_per_1000_cells": np.nan,
        },
        {
            "excluded_subpopulation": "state_2",
            "names": "TNF",
            "anchor_gene": "RHBDF2",
            "pvals_adj": 0.2,
            "effect_per_log1p_rhbdf2_detection_per_1000_cells": 0.2,
            "effect_per_log1p_adam17_detection_per_1000_cells": np.nan,
        },
    ])

    panel_long, summary, coverage, stability = summarize_adam17_outputs(
        config,
        manifest,
        sensitivity,
        tmp_path,
    )

    tnf = summary[summary["gene"].eq("TNF")].iloc[0]
    assert bool(tnf["replicated_same_direction_donor_fdr"])
    assert tnf["fdr_positive_cohorts"] == 2
    assert coverage["availability"].eq("present").all()
    assert len(panel_long) == 4
    assert bool(stability.loc[stability["gene"].eq("TNF"), "all_same_nonzero_direction"].iloc[0])
    assert stability.loc[stability["gene"].eq("TNF"), "variants_fdr_supported"].iloc[0] == 1
