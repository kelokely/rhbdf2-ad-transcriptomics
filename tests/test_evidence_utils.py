from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from src.evidence_utils import build_candidate_evidence_table, summarize_state_metadata_associations


class EvidenceUtilsTests(unittest.TestCase):
    def test_build_candidate_evidence_table(self) -> None:
        candidate_summary = pd.DataFrame(
            [
                {"gene": "RHBDF2", "group_name": "RHBDF2_low", "mean_expression": 0.0, "fraction_expressing": 0.0},
                {"gene": "RHBDF2", "group_name": "RHBDF2_high", "mean_expression": 1.2, "fraction_expressing": 1.0},
                {"gene": "TREM2", "group_name": "RHBDF2_low", "mean_expression": 0.2, "fraction_expressing": 0.1},
                {"gene": "TREM2", "group_name": "RHBDF2_high", "mean_expression": 0.3, "fraction_expressing": 0.2},
            ]
        )
        correlations = pd.DataFrame(
            [{"gene": "RHBDF2", "pearson_r": 1.0, "spearman_r": 1.0}, {"gene": "TREM2", "pearson_r": 0.2, "spearman_r": 0.1}]
        )
        cell_de = pd.DataFrame(
            [{"names": "RHBDF2", "scores": 10.0, "logfoldchanges": 2.0, "pvals_adj": 0.001}]
        )
        continuous_results = pd.DataFrame(
            [
                {"names": "RHBDF2", "slope_per_log1p_rhbdf2": 1.0, "pvals_adj": 0.001},
                {"names": "TREM2", "slope_per_log1p_rhbdf2": 0.2, "pvals_adj": 0.02},
            ]
        )
        pseudobulk_de = pd.DataFrame(
            [{"names": "RHBDF2", "effect_size_log1p_cpm": 1.0, "pvals_adj": 0.01}]
        )
        pseudobulk_candidate_expression = pd.DataFrame(
            [
                {"sample_id": "d1_high", "RHBDF2": 5.0, "TREM2": 2.0},
                {"sample_id": "d1_low", "RHBDF2": 0.0, "TREM2": 1.0},
                {"sample_id": "d2_high", "RHBDF2": 4.0, "TREM2": 3.0},
                {"sample_id": "d2_low", "RHBDF2": 0.0, "TREM2": 1.0},
            ]
        )
        pseudobulk_paired_deltas = pd.DataFrame(
            [
                {"donor_id": "d1", "gene": "RHBDF2", "delta_high_minus_low": 5.0},
                {"donor_id": "d2", "gene": "RHBDF2", "delta_high_minus_low": 4.0},
                {"donor_id": "d1", "gene": "TREM2", "delta_high_minus_low": 1.0},
                {"donor_id": "d2", "gene": "TREM2", "delta_high_minus_low": 2.0},
            ]
        )
        candidate_hurdle_summary = pd.DataFrame(
            [
                {
                    "gene": "RHBDF2",
                    "detection_coef_per_log1p_rhbdf2": np.nan,
                    "detection_pval": np.nan,
                    "positive_expression_coef_per_log1p_rhbdf2": np.nan,
                    "positive_expression_pval": np.nan,
                },
                {
                    "gene": "TREM2",
                    "detection_coef_per_log1p_rhbdf2": 0.5,
                    "detection_pval": 0.03,
                    "positive_expression_coef_per_log1p_rhbdf2": 0.2,
                    "positive_expression_pval": 0.04,
                },
            ]
        )
        subpopulation_summary = pd.DataFrame(
            [
                {
                    "gene": "TREM2",
                    "subpopulation_pseudobulk_n_subgroups_tested": 3,
                    "subpopulation_pseudobulk_n_subgroups_significant": 1,
                    "subpopulation_pseudobulk_n_subgroups_positive": 2,
                    "subpopulation_pseudobulk_mean_effect_log1p_cpm": 0.15,
                    "subpopulation_pseudobulk_median_effect_log1p_cpm": 0.1,
                    "subpopulation_pseudobulk_best_subgroup": "ARM",
                }
            ]
        )
        brain_context_summary = pd.DataFrame(
            [
                {"gene": "RHBDF2", "hpa_microglia_nCPM": 100.0},
                {"gene": "TREM2", "hpa_microglia_nCPM": 250.0, "proteomics_gene_detected": True},
            ]
        )
        proteomics_overlap_summary = pd.DataFrame(
            [
                {
                    "gene": "RHBDF2",
                    "transcript_vs_public_proteomics_direction_match": False,
                    "public_proteomics_overlap_supported": False,
                },
                {
                    "gene": "TREM2",
                    "transcript_vs_public_proteomics_direction_match": True,
                    "public_proteomics_overlap_supported": True,
                },
            ]
        )
        protein_summary = pd.DataFrame(
            [
                {
                    "gene": "RHBDF2",
                    "uniprot_found": True,
                    "reactome_pathway_count": 2,
                    "annotated_ptm_feature_count": 1,
                    "candidate_panel_interaction_count": 1,
                    "candidate_panel_partners": "ADAM17",
                    "candidate_panel_ptm_out_count": 0,
                    "candidate_panel_ptm_in_count": 0,
                    "known_substrate_relationship_count": 0,
                    "known_cleavage_relationship_count": 0,
                    "literature_supported_context": True,
                    "curated_literature_pmid_count": 2,
                },
                {
                    "gene": "TREM2",
                    "uniprot_found": True,
                    "reactome_pathway_count": 1,
                    "annotated_ptm_feature_count": 0,
                    "candidate_panel_interaction_count": 0,
                    "candidate_panel_partners": "",
                    "candidate_panel_ptm_out_count": 0,
                    "candidate_panel_ptm_in_count": 0,
                    "known_substrate_relationship_count": 1,
                    "known_cleavage_relationship_count": 1,
                    "literature_supported_context": True,
                    "curated_literature_pmid_count": 4,
                },
            ]
        )
        genetics_summary = pd.DataFrame(
            [
                {
                    "gene": "RHBDF2",
                    "gwas_matching_trait_count": 0,
                    "gwas_association_count": 0,
                    "gwas_min_p_value": np.nan,
                    "gwas_best_trait": np.nan,
                    "opentargets_matching_disease_count": 1,
                    "opentargets_max_score": 0.2,
                    "opentargets_best_disease": "Alzheimer disease",
                    "combined_genetics_support": True,
                },
                {
                    "gene": "TREM2",
                    "gwas_matching_trait_count": 5,
                    "gwas_association_count": 10,
                    "gwas_min_p_value": 1e-10,
                    "gwas_best_trait": "Alzheimer disease",
                    "opentargets_matching_disease_count": 6,
                    "opentargets_max_score": 0.7,
                    "opentargets_best_disease": "Alzheimer disease",
                    "combined_genetics_support": True,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "candidate_evidence_summary.csv"
            evidence = build_candidate_evidence_table(
                candidate_genes=["RHBDF2", "TREM2"],
                candidate_summary=candidate_summary,
                correlations=correlations,
                cell_de_results=cell_de,
                continuous_results=continuous_results,
                pseudobulk_de_results=pseudobulk_de,
                pseudobulk_candidate_expression=pseudobulk_candidate_expression,
                pseudobulk_paired_candidate_deltas=pseudobulk_paired_deltas,
                candidate_hurdle_summary=candidate_hurdle_summary,
                subpopulation_pseudobulk_gene_summary=subpopulation_summary,
                protein_summary=protein_summary,
                genetics_summary=genetics_summary,
                brain_context_summary=brain_context_summary,
                proteomics_overlap_summary=proteomics_overlap_summary,
                gene_of_interest="RHBDF2",
                state_column="RHBDF2_state",
                high_label="RHBDF2_high",
                low_label="RHBDF2_low",
                evidence_config={
                    "module_weights": {
                        "cell_level": 1.0,
                        "pseudobulk": 3.0,
                        "continuous": 2.0,
                        "detection": 1.0,
                        "positive_expression": 1.0,
                        "subpopulation_pseudobulk": 2.0,
                        "protein_context": 1.0,
                        "genetics": 1.5,
                        "brain_context": 1.0,
                        "public_proteomics_overlap": 1.5,
                    }
                },
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())
            self.assertIn("cell_level_logfoldchange", evidence.columns)
            self.assertIn("continuous_slope_per_log1p_rhbdf2", evidence.columns)
            self.assertIn("pseudobulk_effect_log1p_cpm", evidence.columns)
            self.assertIn("pseudobulk_pearson_r_with_RHBDF2", evidence.columns)
            self.assertIn("paired_delta_mean_high_minus_low", evidence.columns)
            self.assertIn("detection_coef_per_log1p_rhbdf2", evidence.columns)
            self.assertIn("subpopulation_pseudobulk_n_subgroups_tested", evidence.columns)
            self.assertIn("hpa_microglia_nCPM", evidence.columns)
            self.assertIn("protein_context_supported", evidence.columns)
            self.assertIn("genetics_supported", evidence.columns)
            self.assertIn("public_proteomics_overlap_supported", evidence.columns)
            self.assertIn("weighted_total_evidence_score", evidence.columns)
            self.assertEqual(evidence.shape[0], 2)

    def test_summarize_state_metadata_associations(self) -> None:
        adata = ad.AnnData(X=np.array([[1.0], [2.0], [3.0], [4.0]]))
        adata.obs["RHBDF2_state"] = ["RHBDF2_high", "RHBDF2_high", "RHBDF2_low", "RHBDF2_low"]
        adata.obs["atscore"] = ["A+T+", "A+T+", "A-T-", "A-T-"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            summary, paths = summarize_state_metadata_associations(
                adata=adata,
                state_column="RHBDF2_state",
                metadata_columns=["atscore"],
                output_dir=Path(tmp_dir),
            )

            self.assertEqual(summary.shape[0], 1)
            self.assertEqual(summary.loc[0, "metadata_column"], "atscore")
            self.assertTrue(any(path.name == "state_metadata_association_summary.csv" for path in paths))


if __name__ == "__main__":
    unittest.main()
