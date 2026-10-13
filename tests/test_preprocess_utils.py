from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from src.preprocess_utils import prepare_analysis_anndata


class PreprocessUtilsTests(unittest.TestCase):
    def test_prepare_analysis_anndata_normalizes_and_logs(self) -> None:
        adata = ad.AnnData(X=np.array([[1.0, 3.0], [2.0, 2.0]]))

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = prepare_analysis_anndata(
                adata,
                {"normalize_total": True, "target_sum": 10, "log1p": True},
                output_dir=Path(tmp_dir),
            )

            self.assertEqual(result.adata.shape, adata.shape)
            self.assertEqual(result.source_adata.shape, adata.shape)
            self.assertTrue(result.summary["normalize_total"])
            self.assertTrue(result.summary["log1p"])
            self.assertEqual(result.summary["matrix_source"], "X")
            self.assertGreater(result.summary["post_log1p_max"], 0.0)
            self.assertTrue(result.summary_path.exists())

    def test_prepare_analysis_anndata_can_use_raw_matrix(self) -> None:
        adata = ad.AnnData(
            X=np.array([[5.0, 1.0], [6.0, 2.0]]),
            obs=pd.DataFrame(index=["cell1", "cell2"]),
            var=pd.DataFrame(index=["GENE_A", "GENE_B"]),
        )
        raw = ad.AnnData(
            X=np.array([[10.0, 3.0, 1.0], [8.0, 4.0, 0.0]]),
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=["RHBDF2", "GENE_A", "GENE_C"]),
        )
        adata.raw = raw
        adata.obsm["X_umap"] = np.array([[0.0, 0.0], [1.0, 1.0]])

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = prepare_analysis_anndata(
                adata,
                {
                    "matrix_source": "raw",
                    "normalize_total": False,
                    "log1p": False,
                },
                output_dir=Path(tmp_dir),
            )

            self.assertEqual(result.summary["matrix_source"], "raw")
            self.assertEqual(result.source_adata.var_names.tolist(), ["RHBDF2", "GENE_A", "GENE_C"])
            self.assertIn("X_umap", result.source_adata.obsm.keys())

    def test_prepare_analysis_anndata_can_use_named_layer(self) -> None:
        adata = ad.AnnData(X=np.array([[1.0, 3.0], [2.0, 2.0]]))
        adata.layers["counts"] = np.array([[7.0, 1.0], [5.0, 2.0]])
        adata.var_names = ["RHBDF2", "TREM2"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = prepare_analysis_anndata(
                adata,
                {
                    "matrix_source": "layer",
                    "layer_name": "counts",
                    "normalize_total": False,
                    "log1p": False,
                },
                output_dir=Path(tmp_dir),
            )

            self.assertEqual(result.summary["matrix_source"], "layer")
            self.assertEqual(result.summary["matrix_source_layer_name"], "counts")
            np.testing.assert_array_equal(np.asarray(result.source_adata.X), np.asarray(adata.layers["counts"]))


if __name__ == "__main__":
    unittest.main()
