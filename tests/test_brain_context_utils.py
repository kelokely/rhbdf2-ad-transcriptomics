from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.brain_context_utils import (
    _build_local_proteomics_summary,
    _extract_groupings,
    _summarize_hpa_record,
)


class BrainContextUtilsTests(unittest.TestCase):
    def test_extract_groupings(self) -> None:
        html = (
            'content: {"tabs":[{"props":{"groups":[{"name":"ORGANISM_PART","groupings":[["amygdala",["g1","g11"]],["visual cortex",["g10","g20"]]]},'
            '{"name":"DISEASE","groupings":[["Alzheimer\'s disease",["g1","g10"]],["normal",["g11","g20"]]]}]}}]};'
        )
        regions = _extract_groupings(html, "ORGANISM_PART")
        diseases = _extract_groupings(html, "DISEASE")

        self.assertEqual(regions[0][0], "amygdala")
        self.assertEqual(regions[0][1], ["g1", "g11"])
        self.assertEqual(diseases[1][0], "normal")

    def test_summarize_hpa_record(self) -> None:
        row = _summarize_hpa_record(
            gene="RHBDF2",
            ensembl_id="ENSG00000129667",
            record={
                "Brain expression cluster": "Cluster 20: Macrophages & Microglia - Immune response",
                "RNA brain regional specificity": "Low region specificity",
                "RNA brain regional distribution": "Detected in all",
                "RNA single cell type specificity": "Cell type enhanced",
                "RNA single nuclei brain specificity": "Cell type enriched",
                "RNA single cell type specific nCPM": {"Microglia": "321.0"},
                "RNA single nuclei brain specific nCPM": {"central nervous system macrophage": "386.7"},
            },
        )

        self.assertTrue(bool(row["hpa_record_found"]))
        self.assertAlmostEqual(float(row["hpa_microglia_nCPM"]), 321.0)
        self.assertAlmostEqual(float(row["hpa_cns_macrophage_nCPM"]), 386.7)

    def test_build_local_proteomics_summary_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            matrix_path = Path(tmp_dir) / "pd_proteomics.csv"
            pd.DataFrame(
                {
                    "Brain Region": ["Striatum", "Striatum", "Substantia nigra", "Substantia nigra"],
                    "Group": ["Control", "IPD", "Control", "IPD"],
                    "Sample ID": ["C1", "P1", "C2", "P2"],
                    "TBK1": [1.0, 1.7, 2.0, 2.4],
                    "SNCA": [2.0, 2.6, 3.0, 3.8],
                }
            ).to_csv(matrix_path, index=False)

            summary, region_level = _build_local_proteomics_summary(
                genes=["TBK1", "SNCA", "RHBDF2"],
                matrix_path=matrix_path,
                source_accession="PMC10853186",
                context_label="Idiopathic Parkinson disease brain-region proteomics",
                case_labels=["IPD"],
                control_labels=["Control"],
                sheet_name=None,
                region_column="Brain Region",
                group_column="Group",
                sample_id_column="Sample ID",
                values_are_log2=True,
            )

            tbk1 = summary.loc[summary["gene"] == "TBK1"].iloc[0]
            snca = summary.loc[summary["gene"] == "SNCA"].iloc[0]
            rhbdf2 = summary.loc[summary["gene"] == "RHBDF2"].iloc[0]

            self.assertTrue(bool(tbk1["proteomics_gene_detected"]))
            self.assertAlmostEqual(float(tbk1["proteomics_mean_log2_fc_case_vs_control"]), 0.55)
            self.assertEqual(tbk1["proteomics_strongest_case_region"], "Striatum")
            self.assertAlmostEqual(float(snca["proteomics_fraction_regions_higher_in_case"]), 1.0)
            self.assertFalse(bool(rhbdf2["proteomics_gene_detected"]))
            self.assertEqual(region_level.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
