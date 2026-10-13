from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.validation_utils import build_cross_dataset_candidate_summary


class ValidationUtilsTests(unittest.TestCase):
    def test_build_cross_dataset_candidate_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_a = tmp_path / "run_a"
            run_b = tmp_path / "run_b"
            for run_dir in [run_a, run_b]:
                (run_dir / "summaries").mkdir(parents=True)
                (run_dir / "tables").mkdir(parents=True)
                (run_dir / "continuous").mkdir(parents=True)

            for run_dir, dataset_name, logfc, pb_effect, continuous_slope, detection_coef, positive_coef, subgroup_support in [
                (run_a, "GSE1", 1.0, 0.5, 0.3, 0.2, 0.1, 1),
                (run_b, "GSE2", 0.8, 0.2, 0.4, 0.25, 0.12, 0),
            ]:
                (run_dir / "summaries" / "config_used.yaml").write_text(
                    yaml.safe_dump({"dataset_name": dataset_name}),
                    encoding="utf-8",
                )
                pd.DataFrame(
                    [
                        {
                            "gene": "RHBDF2",
                            "cell_level_logfoldchange": logfc,
                            "cell_level_padj": 0.01,
                            "pseudobulk_effect_log1p_cpm": pb_effect,
                            "pseudobulk_padj": 0.02,
                            "cell_level_significant_fdr_0_05": True,
                            "pseudobulk_significant_fdr_0_05": True,
                            "continuous_significant_fdr_0_05": True,
                            "detection_coef_per_log1p_rhbdf2": detection_coef,
                            "detection_padj": 0.03,
                            "positive_expression_coef_per_log1p_rhbdf2": positive_coef,
                            "positive_expression_padj": 0.04,
                            "subpopulation_pseudobulk_n_subgroups_significant": subgroup_support,
                            "subpopulation_pseudobulk_mean_effect_log1p_cpm": 0.15,
                            "protein_context_supported": True,
                            "genetics_supported": True,
                            "brain_context_supported": True,
                            "public_proteomics_overlap_supported": True,
                            "weighted_transcript_evidence_score": 7.0,
                            "weighted_context_evidence_score": 3.5,
                            "weighted_total_evidence_score": 10.5,
                            "transcript_support_count": 6,
                            "context_support_count": 3,
                            "evidence_support_count": 9,
                        }
                    ]
                ).to_csv(run_dir / "tables" / "candidate_evidence_summary.csv", index=False)
                pd.DataFrame(
                    [
                        {
                            "names": "RHBDF2",
                            "slope_per_log1p_rhbdf2": continuous_slope,
                            "pvals_adj": 0.01,
                        }
                    ]
                ).to_csv(run_dir / "continuous" / "continuous_association_with_rhbdf2.csv", index=False)
                pd.DataFrame(
                    [
                        {"RHBDF2_state": "RHBDF2_low", "count": 80},
                        {"RHBDF2_state": "RHBDF2_high", "count": 20},
                    ]
                ).to_csv(run_dir / "summaries" / "state_summary.csv", index=False)

            merged, summary_path = build_cross_dataset_candidate_summary(
                run_dirs=[run_a, run_b],
                output_dir=tmp_path / "merged",
            )

            self.assertTrue(summary_path.exists())
            self.assertEqual(merged.loc[0, "n_cell_level_significant_datasets"], 2)
            self.assertEqual(merged.loc[0, "n_pseudobulk_significant_datasets"], 2)
            self.assertEqual(merged.loc[0, "n_continuous_significant_datasets"], 2)
            self.assertEqual(merged.loc[0, "n_detection_significant_datasets"], 2)
            self.assertEqual(merged.loc[0, "n_positive_expression_significant_datasets"], 2)
            self.assertEqual(merged.loc[0, "n_subpopulation_supported_datasets"], 1)
            self.assertEqual(merged.loc[0, "n_protein_context_supported_datasets"], 2)
            self.assertEqual(merged.loc[0, "n_genetics_supported_datasets"], 2)
            self.assertEqual(merged.loc[0, "n_brain_context_supported_datasets"], 2)
            self.assertEqual(merged.loc[0, "n_public_proteomics_overlap_supported_datasets"], 2)
            self.assertTrue(bool(merged.loc[0, "cell_level_direction_consistent"]))
            self.assertTrue(bool(merged.loc[0, "pseudobulk_direction_consistent"]))
            self.assertTrue(bool(merged.loc[0, "continuous_direction_consistent"]))
            self.assertTrue(bool(merged.loc[0, "detection_direction_consistent"]))
            self.assertTrue(bool(merged.loc[0, "positive_expression_direction_consistent"]))
            self.assertAlmostEqual(float(merged.loc[0, "mean_weighted_total_evidence_score"]), 10.5)


if __name__ == "__main__":
    unittest.main()
