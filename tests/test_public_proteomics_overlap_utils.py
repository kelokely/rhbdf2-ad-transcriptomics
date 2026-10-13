from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.public_proteomics_overlap_utils import summarize_public_proteomics_overlap


class PublicProteomicsOverlapUtilsTests(unittest.TestCase):
    def test_summarize_public_proteomics_overlap(self) -> None:
        cell_de = pd.DataFrame(
            [
                {"names": "RHBDF2", "logfoldchanges": 2.0, "pvals_adj": 0.001},
                {"names": "TREM2", "logfoldchanges": 1.0, "pvals_adj": 0.01},
                {"names": "ADAM17", "logfoldchanges": -1.0, "pvals_adj": 0.2},
            ]
        )
        pseudobulk = pd.DataFrame(
            [
                {"names": "RHBDF2", "effect_size_log1p_cpm": 4.0, "pvals_adj": 0.001},
                {"names": "TREM2", "effect_size_log1p_cpm": 0.5, "pvals_adj": 0.2},
                {"names": "ADAM17", "effect_size_log1p_cpm": -0.3, "pvals_adj": 0.7},
            ]
        )
        continuous = pd.DataFrame(
            [
                {"names": "RHBDF2", "slope_per_log1p_rhbdf2": 1.0, "pvals_adj": 0.001},
                {"names": "TREM2", "slope_per_log1p_rhbdf2": 0.2, "pvals_adj": 0.03},
                {"names": "ADAM17", "slope_per_log1p_rhbdf2": -0.1, "pvals_adj": 0.5},
            ]
        )
        brain_context = pd.DataFrame(
            [
                {
                    "gene": "RHBDF2",
                    "proteomics_gene_detected": False,
                    "proteomics_mean_log2_fc_case_vs_control": None,
                },
                {
                    "gene": "TREM2",
                    "proteomics_gene_detected": True,
                    "proteomics_regions_with_matched_case_control": 8,
                    "proteomics_mean_log2_fc_case_vs_control": 0.4,
                },
                {
                    "gene": "ADAM17",
                    "proteomics_gene_detected": True,
                    "proteomics_regions_with_matched_case_control": 8,
                    "proteomics_mean_log2_fc_case_vs_control": -0.2,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = summarize_public_proteomics_overlap(
                candidate_genes=["RHBDF2", "TREM2", "ADAM17"],
                cell_de_results=cell_de,
                pseudobulk_de_results=pseudobulk,
                continuous_results=continuous,
                brain_context_summary=brain_context,
                output_dir=Path(tmp_dir),
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.summary_path.exists())
            trem2 = result.summary.set_index("gene").loc["TREM2"]
            adam17 = result.summary.set_index("gene").loc["ADAM17"]
            self.assertTrue(bool(trem2["public_proteomics_overlap_supported"]))
            self.assertTrue(bool(adam17["public_proteomics_overlap_supported"]))


if __name__ == "__main__":
    unittest.main()
