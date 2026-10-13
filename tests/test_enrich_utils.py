from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.enrich_utils import export_ranked_gene_lists, run_enrichr_enrichment, run_gprofiler_enrichment


class FakeGProfiler:
    def __init__(self, return_dataframe: bool = True):
        self.return_dataframe = return_dataframe

    def profile(self, **kwargs):
        return pd.DataFrame(
            [
                {
                    "source": "GO:BP",
                    "native": "GO:0000001",
                    "name": "test term",
                    "p_value": 0.001,
                    "significant": True,
                    "description": "desc",
                    "term_size": 10,
                    "query_size": 2,
                    "intersection_size": 2,
                    "effective_domain_size": 100,
                    "precision": 0.2,
                    "recall": 0.2,
                    "query": "query_1",
                    "parents": [],
                }
            ]
        )


class FakeEnrichrResult:
    def __init__(self):
        self.results = pd.DataFrame(
            [
                {
                    "Gene_set": "MSigDB_Hallmark_2020",
                    "Term": "Inflammatory Response",
                    "Overlap": "2/20",
                    "P-value": 0.001,
                    "Adjusted P-value": 0.01,
                    "Old P-value": 0.001,
                    "Old Adjusted P-value": 0.01,
                    "Odds Ratio": 5.0,
                    "Combined Score": 20.0,
                    "Genes": "RHBDF2;TYROBP",
                }
            ]
        )


class EnrichUtilsTests(unittest.TestCase):
    def test_export_ranked_gene_lists_writes_effect_and_signed_scores(self) -> None:
        results = pd.DataFrame(
            [
                {"names": "RHBDF2", "logfoldchanges": 2.5, "pvals_adj": 1e-4},
                {"names": "TREM2", "logfoldchanges": 0.5, "pvals_adj": 0.02},
                {"names": "ADAM17", "logfoldchanges": -0.7, "pvals_adj": 0.2},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            written = export_ranked_gene_lists(
                results=results,
                output_dir=Path(tmp_dir),
                effect_column="logfoldchanges",
                padj_column="pvals_adj",
                prefix="cell_level_",
            )

            self.assertEqual(len(written), 2)
            for path in written:
                self.assertTrue(path.exists())
            first_ranked = pd.read_csv(written[0], sep="\t", header=None)
            self.assertEqual(first_ranked.iloc[0, 0], "RHBDF2")

    @patch("src.enrich_utils.GProfiler", FakeGProfiler)
    def test_run_gprofiler_enrichment_writes_outputs(self) -> None:
        positive = pd.DataFrame({"names": ["RHBDF2", "TYROBP"]})
        negative = pd.DataFrame({"names": ["ADAM17", "IL6R"]})
        enrichment_config = {
            "enabled": True,
            "organism": "hsapiens",
            "sources": ["GO:BP", "REAC"],
            "user_threshold": 0.05,
            "significance_threshold_method": "g_SCS",
            "domain_scope": "custom",
            "use_dataset_background": True,
            "no_iea": False,
            "max_terms_report": 3,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_gprofiler_enrichment(
                positive_results=positive,
                negative_results=negative,
                enrichment_config=enrichment_config,
                output_dir=Path(tmp_dir),
                background_genes=["RHBDF2", "TYROBP", "ADAM17", "IL6R"],
            )

            self.assertIsNotNone(result)
            self.assertTrue(result.up.results_path.exists())
            self.assertTrue(result.down.results_path.exists())
            self.assertTrue(result.up.summary_path.exists())
            self.assertTrue(result.down.summary_path.exists())
            self.assertEqual(result.up.top_terms[0].startswith("GO:BP:test term"), True)

    @patch("src.enrich_utils.gp.enrichr", return_value=FakeEnrichrResult())
    def test_run_enrichr_enrichment_writes_outputs(self, mock_enrichr) -> None:
        positive = pd.DataFrame({"names": ["RHBDF2", "TYROBP"]})
        negative = pd.DataFrame({"names": ["ADAM17", "IL6R"]})
        enrichr_config = {
            "enabled": True,
            "organism": "human",
            "libraries": ["MSigDB_Hallmark_2020", "CellMarker_2024"],
            "cutoff": 0.05,
            "use_dataset_background": True,
            "max_terms_report": 3,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_enrichr_enrichment(
                positive_results=positive,
                negative_results=negative,
                enrichr_config=enrichr_config,
                output_dir=Path(tmp_dir),
                background_genes=["RHBDF2", "TYROBP", "ADAM17", "IL6R"],
            )

            self.assertIsNotNone(result)
            self.assertTrue(result.up.results_path.exists())
            self.assertTrue(result.down.summary_path.exists())
            self.assertEqual(result.backend_name, "enrichr")
            self.assertTrue(result.up.top_terms[0].startswith("MSigDB_Hallmark_2020:Inflammatory Response"))

    @patch(
        "src.enrich_utils.gp.enrichr",
        side_effect=[RuntimeError("background endpoint returned non-json"), FakeEnrichrResult(), FakeEnrichrResult()],
    )
    def test_run_enrichr_enrichment_retries_without_background_on_background_failure(self, mock_enrichr) -> None:
        positive = pd.DataFrame({"names": ["RHBDF2", "TYROBP"]})
        negative = pd.DataFrame({"names": ["ADAM17", "IL6R"]})
        enrichr_config = {
            "enabled": True,
            "organism": "human",
            "libraries": ["MSigDB_Hallmark_2020"],
            "cutoff": 0.05,
            "use_dataset_background": True,
            "max_terms_report": 3,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_enrichr_enrichment(
                positive_results=positive,
                negative_results=negative,
                enrichr_config=enrichr_config,
                output_dir=Path(tmp_dir),
                background_genes=["RHBDF2", "TYROBP", "ADAM17", "IL6R"],
            )

            self.assertIsNotNone(result)
            self.assertEqual(mock_enrichr.call_count, 3)
            self.assertEqual(mock_enrichr.call_args_list[0].kwargs["background"], ["RHBDF2", "TYROBP", "ADAM17", "IL6R"])
            self.assertIsNone(mock_enrichr.call_args_list[1].kwargs["background"])
            up_summary = result.up.summary_path.read_text(encoding="utf-8")
            self.assertIn("background_fallback_applied: true", up_summary)
            self.assertIn("retried without", up_summary)
            self.assertIn("dataset background", up_summary)


if __name__ == "__main__":
    unittest.main()
