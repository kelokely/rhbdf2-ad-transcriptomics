from __future__ import annotations

import unittest

import pandas as pd

from src.protein_evidence_utils import parse_uniprot_record, summarize_candidate_network


class ProteinEvidenceUtilsTests(unittest.TestCase):
    def test_parse_uniprot_record_extracts_ptm_and_reactome(self) -> None:
        record = {
            "primaryAccession": "Q6PJF5",
            "uniProtkbId": "RHDF2_HUMAN",
            "annotationScore": 5.0,
            "proteinExistence": "1: Evidence at protein level",
            "proteinDescription": {
                "recommendedName": {
                    "fullName": {"value": "Inactive rhomboid protein 2"},
                }
            },
            "comments": [
                {"commentType": "FUNCTION", "texts": [{"value": "Controls ADAM17 trafficking."}]},
                {
                    "commentType": "SUBCELLULAR LOCATION",
                    "subcellularLocations": [
                        {"location": {"value": "Cell membrane"}},
                    ],
                },
            ],
            "features": [
                {
                    "type": "Modified residue",
                    "location": {"start": {"value": 90}},
                    "description": "Phosphoserine",
                    "evidences": [{"source": "PubMed", "id": "18669648"}],
                }
            ],
            "uniProtKBCrossReferences": [
                {
                    "database": "Reactome",
                    "id": "R-HSA-9662834",
                    "properties": [{"key": "PathwayName", "value": "CD163 mediating an anti-inflammatory response"}],
                }
            ],
        }

        summary, ptm_rows, reactome_rows = parse_uniprot_record("RHBDF2", record)

        self.assertTrue(summary["uniprot_found"])
        self.assertEqual(summary["protein_name"], "Inactive rhomboid protein 2")
        self.assertEqual(summary["annotated_ptm_feature_count"], 1)
        self.assertIn("Controls ADAM17 trafficking.", summary["function_summary"])
        self.assertIn("18669648", summary["annotated_ptm_pmids"])
        self.assertEqual(len(ptm_rows), 1)
        self.assertEqual(ptm_rows[0]["position"], 90)
        self.assertEqual(len(reactome_rows), 1)
        self.assertEqual(reactome_rows[0]["reactome_id"], "R-HSA-9662834")

    def test_summarize_candidate_network(self) -> None:
        candidate_network = pd.DataFrame(
            [
                {"source_genesymbol": "ADAM17", "target_genesymbol": "TNF"},
                {"source_genesymbol": "TBK1", "target_genesymbol": "IRF3"},
            ]
        )
        candidate_ptm = pd.DataFrame(
            [
                {"enzyme_genesymbol": "ADAM17", "substrate_genesymbol": "TNF"},
                {"enzyme_genesymbol": "SYK", "substrate_genesymbol": "TYROBP"},
                {"enzyme_genesymbol": "MAPK1", "substrate_genesymbol": "IRF3"},
            ]
        )
        candidate_panel_ptm = pd.DataFrame(
            [
                {"enzyme_genesymbol": "ADAM17", "substrate_genesymbol": "TNF"},
                {"enzyme_genesymbol": "SYK", "substrate_genesymbol": "TYROBP"},
            ]
        )

        summary = summarize_candidate_network(
            candidate_genes=["ADAM17", "TNF", "IRF3"],
            candidate_network=candidate_network,
            candidate_ptm=candidate_ptm,
            candidate_panel_ptm=candidate_panel_ptm,
        )

        adam17 = summary.set_index("gene").loc["ADAM17"]
        self.assertEqual(adam17["candidate_panel_interaction_count"], 1)
        self.assertEqual(adam17["candidate_adjacent_ptm_out_count"], 1)
        self.assertEqual(adam17["candidate_panel_ptm_out_count"], 1)

        irf3 = summary.set_index("gene").loc["IRF3"]
        self.assertEqual(irf3["candidate_panel_interaction_count"], 1)
        self.assertEqual(irf3["candidate_adjacent_ptm_in_count"], 1)


if __name__ == "__main__":
    unittest.main()
