"""Tests for SEA-AD taxonomy and streaming matrix extraction helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy import sparse

from src.seaad_utils import extract_h5ad_csr_columns, map_seaad_cell_types


class SeaADUtilsTests(unittest.TestCase):
    def test_extract_h5ad_csr_columns_uses_requested_order(self) -> None:
        source = sparse.csr_matrix([[1, 0, 2], [0, 3, 4]], dtype=float)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "toy.h5ad"
            with h5py.File(path, "w") as handle:
                var = handle.create_group("var")
                var.attrs["_index"] = "index"
                var.create_dataset("index", data=np.asarray([b"A", b"B", b"C"]))
                layers = handle.create_group("layers")
                matrix = layers.create_group("UMIs")
                matrix.attrs["shape"] = source.shape
                matrix.create_dataset("data", data=source.data)
                matrix.create_dataset("indices", data=source.indices)
                matrix.create_dataset("indptr", data=source.indptr)
            result, var = extract_h5ad_csr_columns(path, "layers/UMIs", ["C", "A", "MISSING"], rows_per_block=1)
        np.testing.assert_array_equal(result.toarray(), [[2, 1], [4, 0]])
        self.assertEqual(var.index.tolist(), ["C", "A"])

    def test_map_seaad_cell_types_uses_author_taxonomy(self) -> None:
        classes = np.asarray(["Neuronal: Glutamatergic", "Neuronal: GABAergic", "Non-neuronal and Non-neural"])
        subclasses = np.asarray(["L2/3 IT", "Pvalb", "Immune"])
        result = map_seaad_cell_types(classes, subclasses)
        self.assertEqual(result.tolist(), ["Excitatory neurons", "Inhibitory neurons", "Microglia"])


if __name__ == "__main__":
    unittest.main()
