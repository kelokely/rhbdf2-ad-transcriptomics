"""Tests for targeted Stereo-seq GEF ingestion."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from src.stereo_spatial_utils import read_cellbin_gef_targeted


class StereoSpatialUtilsTests(unittest.TestCase):
    def test_read_cellbin_gef_targeted(self) -> None:
        cell_dtype = np.dtype(
            [("id", "<u4"), ("x", "<i4"), ("y", "<i4"), ("geneCount", "<u2"), ("expCount", "<u2")]
        )
        gene_dtype = np.dtype(
            [("geneID", "S32"), ("geneName", "S32"), ("offset", "<u4"), ("cellCount", "<u4")]
        )
        expression_dtype = np.dtype([("cellID", "<u4"), ("count", "<u2")])
        cells = np.array([(0, 10, 20, 300, 10), (1, 30, 40, 400, 20), (2, 50, 60, 500, 30)], dtype=cell_dtype)
        genes = np.array(
            [(b"ENSG1", b"RHBDF2", 0, 2), (b"ENSG2", b"AQP4", 2, 1)], dtype=gene_dtype
        )
        expression = np.array([(0, 2), (2, 1), (1, 4)], dtype=expression_dtype)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.cellbin.gef"
            with h5py.File(path, "w") as handle:
                group = handle.create_group("cellBin")
                group.create_dataset("cell", data=cells)
                group.create_dataset("gene", data=genes)
                group.create_dataset("geneExp", data=expression)
            result = read_cellbin_gef_targeted(path, ["RHBDF2", "AQP4", "TYROBP"])
        self.assertEqual(result.shape, (3, 3))
        np.testing.assert_array_equal(result.X.toarray(), [[2, 0, 0], [0, 4, 0], [1, 0, 0]])
        np.testing.assert_array_equal(result.obsm["spatial"], [[10, 20], [30, 40], [50, 60]])


if __name__ == "__main__":
    unittest.main()
