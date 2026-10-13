from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from src.dataset_onboarding_utils import scaffold_dataset_config


class DatasetOnboardingUtilsTests(unittest.TestCase):
    def test_scaffold_dataset_config_uses_metadata_suggestions_and_raw_when_needed(self) -> None:
        adata = ad.AnnData(
            X=np.array([[1.0, 0.0], [2.0, 1.0]]),
            obs=pd.DataFrame(
                {
                    "diagnosis": ["AD", "CTR"],
                    "donor_id": ["d1", "d2"],
                    "microglia_subpopulation": ["ARM", "Homeostatic"],
                },
                index=["cell1", "cell2"],
            ),
            var=pd.DataFrame(index=["GENE_A", "GENE_B"]),
        )
        adata.raw = ad.AnnData(
            X=np.array([[5.0, 0.0], [3.0, 1.0]]),
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=["RHBDF2", "GENE_A"]),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_config = Path(tmp_dir) / "configs" / "example.yaml"
            result = scaffold_dataset_config(
                adata=adata,
                dataset_name="GSETEST",
                input_path=Path(tmp_dir) / "data" / "raw" / "example.h5ad",
                output_config_path=output_config,
                gene_of_interest="RHBDF2",
            )

            self.assertTrue(result.output_config_path.exists())
            self.assertTrue(result.metadata_suggestions_path.exists())
            self.assertEqual(result.matrix_source, "raw")
            self.assertEqual(result.diagnosis_column, "diagnosis")
            self.assertEqual(result.donor_column, "donor_id")
            self.assertEqual(result.cluster_column, "microglia_subpopulation")

            config = yaml.safe_load(result.output_config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["preprocessing"]["matrix_source"], "raw")
            self.assertEqual(config["metadata"]["diagnosis_column"], "diagnosis")
            self.assertEqual(config["metadata"]["donor_column"], "donor_id")


if __name__ == "__main__":
    unittest.main()
