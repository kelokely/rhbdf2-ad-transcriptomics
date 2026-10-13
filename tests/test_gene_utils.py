from __future__ import annotations

import unittest

import anndata as ad
import numpy as np
from scipy import sparse

from src.gene_utils import check_gene_presence, ensure_unique_var_names, extract_expression_vector, gene_presence_table


class GeneUtilsTests(unittest.TestCase):
    def test_check_gene_presence_and_dense_extraction(self) -> None:
        adata = ad.AnnData(
            X=np.array([[0.0, 1.0], [2.0, 3.0]]),
        )
        adata.var_names = ["RHBDF2", "TREM2"]

        self.assertTrue(check_gene_presence(adata, "RHBDF2"))
        self.assertFalse(check_gene_presence(adata, "ADAM17"))
        np.testing.assert_array_equal(extract_expression_vector(adata, "TREM2"), np.array([1.0, 3.0]))

    def test_sparse_extraction_and_presence_table(self) -> None:
        adata = ad.AnnData(
            X=sparse.csr_matrix(np.array([[0.0, 4.0], [5.0, 6.0]])),
        )
        adata.var_names = ["RHBDF2", "TYROBP"]

        values = extract_expression_vector(adata, "RHBDF2")
        np.testing.assert_array_equal(values, np.array([0.0, 5.0]))

        presence = gene_presence_table(adata, ["RHBDF2", "ADAM17"])
        self.assertEqual(presence["present"].tolist(), [True, False])

    def test_duplicate_var_names_raise_clear_error(self) -> None:
        adata = ad.AnnData(X=np.array([[1.0, 2.0]]))
        adata.var_names = ["RHBDF2", "RHBDF2"]

        with self.assertRaises(ValueError):
            ensure_unique_var_names(adata)


if __name__ == "__main__":
    unittest.main()
