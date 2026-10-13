from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np

from src.continuous_evidence_utils import run_continuous_evidence


class ContinuousEvidenceUtilsTests(unittest.TestCase):
    def test_run_continuous_evidence(self) -> None:
        adata = ad.AnnData(
            X=np.array(
                [
                    [0.0, 0.0, 2.0],
                    [0.5, 0.0, 2.5],
                    [1.0, 1.0, 3.0],
                    [1.5, 1.0, 3.5],
                    [2.0, 2.0, 4.0],
                    [2.5, 3.0, 4.5],
                ]
            )
        )
        adata.var_names = ["RHBDF2", "TREM2", "ADAM17"]
        adata.obs["donor_id"] = ["d1", "d1", "d1", "d2", "d2", "d2"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_continuous_evidence(
                adata=adata,
                gene_of_interest="RHBDF2",
                candidate_genes=["RHBDF2", "TREM2"],
                continuous_config={
                    "enabled": True,
                    "donor_column": "donor_id",
                    "top_n_export": 2,
                    "min_positive_cells": 2,
                },
                output_dir=Path(tmp_dir),
                fallback_donor_column="donor_id",
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.full_results_path.exists())
            self.assertIn("slope_per_log1p_rhbdf2", result.full_results.columns)
            self.assertIn("detection_coef_per_log1p_rhbdf2", result.candidate_hurdle_summary.columns)
            trem2_row = result.candidate_hurdle_summary[result.candidate_hurdle_summary["gene"] == "TREM2"].iloc[0]
            self.assertGreaterEqual(int(trem2_row["n_positive_cells"]), 2)

    def test_adam17_anchor_uses_anchor_specific_columns_and_files(self) -> None:
        adata = ad.AnnData(
            X=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.5, 1.0, 0.0],
                    [1.0, 2.0, 1.0],
                    [1.5, 3.0, 1.0],
                    [2.0, 4.0, 2.0],
                    [2.5, 5.0, 3.0],
                ]
            )
        )
        adata.var_names = ["RHBDF2", "ADAM17", "SYK"]
        adata.obs["donor_id"] = ["d1", "d1", "d1", "d2", "d2", "d2"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_continuous_evidence(
                adata=adata,
                gene_of_interest="ADAM17",
                candidate_genes=["RHBDF2", "ADAM17", "SYK"],
                continuous_config={"enabled": True, "donor_column": "donor_id", "min_positive_cells": 2},
                output_dir=Path(tmp_dir),
                fallback_donor_column="donor_id",
            )

            assert result is not None
            self.assertEqual(result.full_results_path.name, "continuous_association_with_adam17.csv")
            self.assertIn("slope_per_log1p_adam17", result.full_results.columns)
            self.assertIn("detection_coef_per_log1p_adam17", result.candidate_hurdle_summary.columns)
            self.assertTrue((Path(tmp_dir) / "candidate_hurdle_summary_adam17.csv").exists())


if __name__ == "__main__":
    unittest.main()
