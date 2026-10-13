from __future__ import annotations

import unittest

from src.genetics_evidence_utils import fetch_matching_gwas_traits, summarize_gene_genetics


class FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload

    def get(self, *args, **kwargs):
        return FakeResponse(self.payload)


class GeneticsEvidenceUtilsTests(unittest.TestCase):
    def test_fetch_matching_gwas_traits_filters_keywords(self) -> None:
        session = FakeSession(
            {
                "_embedded": {
                    "efo_traits": [
                        {"efo_id": "MONDO_0004975", "efo_trait": "Alzheimer disease", "uri": "x"},
                        {"efo_id": "EFO_0000618", "efo_trait": "inflammatory bowel disease", "uri": "y"},
                    ]
                }
            }
        )

        rows = fetch_matching_gwas_traits(
            gene="RHBDF2",
            keywords=["alzheimer", "dementia"],
            page_size=100,
            session=session,
            timeout_seconds=5,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["efo_id"], "MONDO_0004975")
        self.assertIn("alzheimer", rows[0]["matched_keywords"])

    def test_summarize_gene_genetics(self) -> None:
        summary = summarize_gene_genetics(
            gene="TREM2",
            ensembl_id="ENSG00000095970",
            gwas_traits=[{"efo_id": "MONDO_0004975", "efo_trait": "Alzheimer disease"}],
            gwas_associations=[
                {
                    "efo_trait": "Alzheimer disease",
                    "reported_trait": "Late onset Alzheimer's disease",
                    "p_value": 1e-10,
                }
            ],
            opentargets_rows=[
                {
                    "disease_name": "Alzheimer disease",
                    "score": 0.57,
                }
            ],
        )

        self.assertEqual(summary["gene"], "TREM2")
        self.assertEqual(summary["gwas_association_count"], 1)
        self.assertEqual(summary["gwas_best_trait"], "Alzheimer disease")
        self.assertAlmostEqual(summary["opentargets_max_score"], 0.57)
        self.assertTrue(summary["combined_genetics_support"])


if __name__ == "__main__":
    unittest.main()
