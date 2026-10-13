from __future__ import annotations

import unittest

import anndata as ad
import numpy as np

from src.state_utils import assign_expression_state, build_state_summary


class StateUtilsTests(unittest.TestCase):
    def test_assign_expression_state_with_median_split(self) -> None:
        adata = ad.AnnData(X=np.array([[0.0], [1.0], [2.0], [3.0]]))
        adata.var_names = ["RHBDF2"]

        updated_adata, result = assign_expression_state(
            adata,
            "RHBDF2",
            {
                "method": "median",
                "quantile": 0.75,
                "new_column_name": "RHBDF2_state",
                "high_label": "RHBDF2_high",
                "low_label": "RHBDF2_low",
            },
        )

        self.assertIn("RHBDF2_state", updated_adata.obs.columns)
        self.assertEqual(result.threshold_method, "median")
        self.assertAlmostEqual(result.threshold_value, 1.5)

        summary = build_state_summary(updated_adata, result)
        self.assertEqual(int(summary["count"].sum()), 4)
        self.assertCountEqual(summary["RHBDF2_state"].tolist(), ["RHBDF2_high", "RHBDF2_low"])

    def test_assign_expression_state_nonzero(self) -> None:
        adata = ad.AnnData(X=np.array([[0.0], [0.0], [5.0]]))
        adata.var_names = ["RHBDF2"]

        updated_adata, result = assign_expression_state(
            adata,
            "RHBDF2",
            {
                "method": "nonzero",
                "new_column_name": "RHBDF2_state",
                "high_label": "RHBDF2_high",
                "low_label": "RHBDF2_low",
            },
        )

        labels = updated_adata.obs["RHBDF2_state"].astype(str).tolist()
        self.assertEqual(labels.count("RHBDF2_high"), 1)
        self.assertEqual(labels.count("RHBDF2_low"), 2)
        self.assertFalse(result.fallback_used)

    def test_assign_expression_state_uses_fallback_for_tied_values(self) -> None:
        adata = ad.AnnData(X=np.array([[0.0], [0.0], [0.0], [0.0]]))
        adata.var_names = ["RHBDF2"]

        updated_adata, result = assign_expression_state(
            adata,
            "RHBDF2",
            {
                "method": "median",
                "new_column_name": "RHBDF2_state",
                "high_label": "RHBDF2_high",
                "low_label": "RHBDF2_low",
            },
        )

        labels = updated_adata.obs["RHBDF2_state"].astype(str).tolist()
        self.assertTrue(result.fallback_used)
        self.assertEqual(labels.count("RHBDF2_high"), 2)
        self.assertEqual(labels.count("RHBDF2_low"), 2)


if __name__ == "__main__":
    unittest.main()
