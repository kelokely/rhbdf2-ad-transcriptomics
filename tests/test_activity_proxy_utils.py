from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np

from src.activity_proxy_utils import run_pathway_activity_proxies


class ActivityProxyUtilsTests(unittest.TestCase):
    def test_run_pathway_activity_proxies(self) -> None:
        adata = ad.AnnData(
            X=np.array(
                [
                    [2.0, 2.0, 0.5, 0.2, 0.0],
                    [2.5, 2.3, 0.4, 0.1, 0.0],
                    [0.2, 0.3, 1.5, 1.2, 0.0],
                    [0.1, 0.2, 1.3, 1.1, 0.0],
                    [2.2, 2.1, 0.3, 0.2, 1.0],
                    [0.2, 0.1, 1.4, 1.0, 0.8],
                ]
            )
        )
        adata.var_names = ["TREM2", "TYROBP", "TNF", "RELA", "RHBDF2"]
        adata.obs["RHBDF2_state"] = [
            "RHBDF2_high",
            "RHBDF2_high",
            "RHBDF2_low",
            "RHBDF2_low",
            "RHBDF2_high",
            "RHBDF2_low",
        ]
        adata.obs["donor_id"] = ["d1", "d2", "d1", "d2", "d3", "d3"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_pathway_activity_proxies(
                adata=adata,
                gene_of_interest="RHBDF2",
                state_column="RHBDF2_state",
                high_label="RHBDF2_high",
                low_label="RHBDF2_low",
                proxy_config={
                    "enabled": True,
                    "donor_column": "donor_id",
                    "min_genes_present": 2,
                    "gene_sets": {
                        "trem2_tyrobp_syk_response": ["TREM2", "TYROBP"],
                        "inflammatory_nfkb": ["TNF", "RELA"],
                    },
                },
                output_dir=Path(tmp_dir),
                fallback_donor_column="donor_id",
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.summary_path.exists())
            self.assertTrue(result.cell_scores_path.exists())
            self.assertIn("cell_level_delta_high_minus_low", result.summary.columns)
            self.assertIn("donor_paired_mean_delta", result.summary.columns)
            self.assertEqual(set(result.summary["module_name"]), {"trem2_tyrobp_syk_response", "inflammatory_nfkb"})


if __name__ == "__main__":
    unittest.main()
