from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.report_utils import build_report_text
from src.state_utils import StateAssignmentResult


class ReportUtilsTests(unittest.TestCase):
    def test_build_report_text_handles_missing_optional_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            text = build_report_text(
                config={
                    "dataset_name": "Example",
                    "_config_path": "configs/example.yaml",
                    "input_path": "data/raw/example.h5ad",
                    "gene_of_interest": "RHBDF2",
                },
                config_snapshot_path=Path(tmp_dir) / "config_used.yaml",
                dataset_summary={"n_cells": 10, "n_genes": 5},
                metadata_validation=pd.DataFrame(
                    [{"config_key": "diagnosis_column", "column_name": None, "configured": False, "exists": False}]
                ),
                metadata_suggestions=pd.DataFrame(),
                gene_presence=pd.DataFrame([{"gene": "RHBDF2", "present": True}]),
                state_result=StateAssignmentResult(
                    column_name="RHBDF2_state",
                    threshold_method="median",
                    threshold_value=0.0,
                    high_label="RHBDF2_high",
                    low_label="RHBDF2_low",
                    fallback_used=False,
                    note="Assigned using the configured threshold.",
                ),
                state_summary=pd.DataFrame(
                    [
                        {"RHBDF2_state": "RHBDF2_high", "count": 5},
                        {"RHBDF2_state": "RHBDF2_low", "count": 5},
                    ]
                ),
                preprocessing_summary={"normalize_total": True, "target_sum": 10000, "log1p": True},
                de_results=pd.DataFrame(columns=["names", "logfoldchanges", "pvals_adj"]),
                pseudobulk_result=None,
                gprofiler_result=None,
                pseudobulk_gprofiler_result=None,
                continuous_result=None,
                continuous_gprofiler_result=None,
                enrichr_result=None,
                pseudobulk_enrichr_result=None,
                continuous_enrichr_result=None,
                subpopulation_pseudobulk_result=None,
                metadata_association_summary=None,
                correlations=pd.DataFrame(columns=["gene", "pearson_r", "spearman_r"]),
                protein_evidence_result=None,
                genetics_evidence_result=None,
                brain_context_result=None,
                protein_context_result=None,
                public_proteomics_overlap_result=None,
                pathway_activity_proxy_result=None,
                module_status_summary=pd.DataFrame(
                    [{"module_name": "protein_evidence", "status": "skipped", "message": "Disabled in config."}]
                ),
                plot_status_summary=pd.DataFrame(
                    [{"plot_name": "violin_by_state", "status": "completed", "message": "ok"}]
                ),
                output_files=["summaries/config_used.yaml"],
            )

        self.assertIn("Module Status", text)
        self.assertIn("Protein And PTM Evidence", text)
        self.assertIn("protein_evidence", text)
        self.assertIn("Plot Status", text)


if __name__ == "__main__":
    unittest.main()
