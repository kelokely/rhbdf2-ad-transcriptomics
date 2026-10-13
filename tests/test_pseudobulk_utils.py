from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from src.pseudobulk_utils import run_pseudobulk_differential_expression, run_within_subpopulation_pseudobulk


class PseudobulkUtilsTests(unittest.TestCase):
    def test_run_pseudobulk_differential_expression(self) -> None:
        x = np.array(
            [
                [10.0, 1.0],
                [9.0, 1.0],
                [0.0, 8.0],
                [0.0, 9.0],
                [8.0, 0.0],
                [7.0, 1.0],
                [0.0, 7.0],
                [0.0, 6.0],
            ]
        )
        adata = ad.AnnData(X=x)
        adata.var_names = ["RHBDF2", "TREM2"]
        adata.obs["sampleID"] = ["d1", "d1", "d1", "d1", "d2", "d2", "d2", "d2"]
        adata.obs["RHBDF2_state"] = [
            "RHBDF2_high",
            "RHBDF2_high",
            "RHBDF2_low",
            "RHBDF2_low",
            "RHBDF2_high",
            "RHBDF2_high",
            "RHBDF2_low",
            "RHBDF2_low",
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_pseudobulk_differential_expression(
                adata=adata,
                pseudobulk_config={
                    "enabled": True,
                    "donor_column": "sampleID",
                    "group_column": "RHBDF2_state",
                    "group_a": "RHBDF2_high",
                    "group_b": "RHBDF2_low",
                    "paired_only": True,
                    "min_cells_per_sample": 2,
                    "min_donors_per_group": 2,
                    "normalization_target_sum": 1000000,
                    "top_n_export": 10,
                },
                output_dir=Path(tmp_dir),
                candidate_genes=["RHBDF2", "TREM2"],
                state_column="RHBDF2_state",
                fallback_donor_column="sampleID",
            )

            self.assertEqual(result.n_pseudobulk_samples, 4)
            self.assertEqual(result.n_paired_donors, 2)
            self.assertIn("effect_size_log1p_cpm", result.full_results.columns)
            rhbdf2_row = result.full_results[result.full_results["names"] == "RHBDF2"].iloc[0]
            self.assertGreater(rhbdf2_row["effect_size_log1p_cpm"], 0.0)
            self.assertFalse(result.paired_candidate_deltas.empty)

    def test_run_within_subpopulation_pseudobulk(self) -> None:
        x = np.array(
            [
                [10.0, 1.0],
                [9.0, 1.0],
                [0.0, 8.0],
                [0.0, 9.0],
                [8.0, 0.0],
                [7.0, 1.0],
                [0.0, 7.0],
                [0.0, 6.0],
                [11.0, 1.0],
                [10.0, 1.0],
                [0.0, 7.0],
                [0.0, 8.0],
                [9.0, 0.0],
                [8.0, 1.0],
                [0.0, 6.0],
                [0.0, 5.0],
            ]
        )
        adata = ad.AnnData(X=x)
        adata.var_names = ["RHBDF2", "TREM2"]
        adata.obs["sampleID"] = [
            "d1", "d1", "d1", "d1",
            "d2", "d2", "d2", "d2",
            "d1", "d1", "d1", "d1",
            "d2", "d2", "d2", "d2",
        ]
        adata.obs["RHBDF2_state"] = [
            "RHBDF2_high", "RHBDF2_high", "RHBDF2_low", "RHBDF2_low",
            "RHBDF2_high", "RHBDF2_high", "RHBDF2_low", "RHBDF2_low",
            "RHBDF2_high", "RHBDF2_high", "RHBDF2_low", "RHBDF2_low",
            "RHBDF2_high", "RHBDF2_high", "RHBDF2_low", "RHBDF2_low",
        ]
        adata.obs["microglia_subpopulations"] = [
            "ARM", "ARM", "ARM", "ARM",
            "ARM", "ARM", "ARM", "ARM",
            "Homeostatic", "Homeostatic", "Homeostatic", "Homeostatic",
            "Homeostatic", "Homeostatic", "Homeostatic", "Homeostatic",
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_within_subpopulation_pseudobulk(
                adata=adata,
                pseudobulk_config={
                    "enabled": True,
                    "within_subpopulation": True,
                    "donor_column": "sampleID",
                    "group_column": "RHBDF2_state",
                    "group_a": "RHBDF2_high",
                    "group_b": "RHBDF2_low",
                    "paired_only": True,
                    "min_cells_per_sample": 3,
                    "min_donors_per_group": 2,
                    "normalization_target_sum": 1000000,
                    "top_n_export": 10,
                    "min_cells_per_subpopulation": 4,
                    "within_subpopulation_min_cells_per_sample": 2,
                },
                output_dir=Path(tmp_dir),
                candidate_genes=["RHBDF2", "TREM2"],
                state_column="RHBDF2_state",
                fallback_donor_column="sampleID",
                subgroup_column="microglia_subpopulations",
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(int((result.run_summary["status"] == "completed").sum()), 2)
            self.assertTrue((result.run_summary["effective_min_cells_per_sample"] == 2).all())
            self.assertIn("subpopulation_pseudobulk_n_subgroups_tested", result.gene_summary.columns)
            self.assertTrue(result.run_summary_path.exists())


if __name__ == "__main__":
    unittest.main()
