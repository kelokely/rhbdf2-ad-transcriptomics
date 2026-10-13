from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.protein_context_utils import build_protein_context_summary


class ProteinContextUtilsTests(unittest.TestCase):
    def test_build_protein_context_summary(self) -> None:
        protein_summary = pd.DataFrame(
            [
                {
                    "gene": "RHBDF2",
                    "uniprot_found": True,
                    "protein_name": "Inactive rhomboid protein 2",
                    "annotation_score": 5.0,
                    "annotated_ptm_feature_count": 2,
                    "candidate_panel_interaction_count": 1,
                    "candidate_panel_ptm_out_count": 0,
                    "candidate_panel_ptm_in_count": 0,
                    "known_substrate_relationship_count": 0,
                    "known_cleavage_relationship_count": 0,
                    "literature_supported_context": True,
                    "curated_literature_pmid_count": 3,
                },
                {
                    "gene": "TREM2",
                    "uniprot_found": True,
                    "protein_name": "Triggering receptor expressed on myeloid cells 2",
                    "annotation_score": 5.0,
                    "annotated_ptm_feature_count": 0,
                    "candidate_panel_interaction_count": 0,
                    "candidate_panel_ptm_out_count": 0,
                    "candidate_panel_ptm_in_count": 1,
                    "known_substrate_relationship_count": 1,
                    "known_cleavage_relationship_count": 1,
                    "literature_supported_context": True,
                    "curated_literature_pmid_count": 4,
                },
            ]
        )
        brain_context_summary = pd.DataFrame(
            [
                {"gene": "RHBDF2", "hpa_microglia_nCPM": 300.0, "proteomics_gene_detected": False},
                {"gene": "TREM2", "hpa_microglia_nCPM": 100.0, "proteomics_gene_detected": True},
            ]
        )
        proteomics_overlap_summary = pd.DataFrame(
            [
                {"gene": "RHBDF2", "public_proteomics_overlap_supported": False},
                {"gene": "TREM2", "public_proteomics_overlap_supported": True},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = build_protein_context_summary(
                candidate_genes=["RHBDF2", "TREM2"],
                protein_summary=protein_summary,
                brain_context_summary=brain_context_summary,
                proteomics_overlap_summary=proteomics_overlap_summary,
                output_dir=Path(tmp_dir),
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.summary_path.exists())
            self.assertIn("protein_context_evidence_supported", result.summary.columns)
            self.assertTrue(result.summary.set_index("gene").loc["TREM2", "protein_context_evidence_supported"])


if __name__ == "__main__":
    unittest.main()
