from __future__ import annotations

import unittest

import anndata as ad
import numpy as np

from src.metadata_utils import suggest_metadata_columns


class MetadataUtilsTests(unittest.TestCase):
    def test_suggest_metadata_columns_finds_likely_roles(self) -> None:
        adata = ad.AnnData(X=np.array([[1.0], [2.0], [3.0]]))
        adata.obs["diagnosis"] = ["AD", "AD", "CTRL"]
        adata.obs["donor_id"] = ["d1", "d2", "d3"]
        adata.obs["microglia_subpopulation"] = ["ARM", "ARM", "Homeostatic"]

        suggestions = suggest_metadata_columns(adata)

        self.assertIn("diagnosis", suggestions["suggested_role"].tolist())
        self.assertIn("donor", suggestions["suggested_role"].tolist())
        self.assertIn("cluster", suggestions["suggested_role"].tolist())


if __name__ == "__main__":
    unittest.main()
