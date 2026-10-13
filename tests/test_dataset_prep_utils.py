from __future__ import annotations

import gzip
import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad

from src.dataset_prep_utils import (
    collapse_ensembl_matrix_to_gene_symbols,
    derive_gse161045_microglia_subset_h5ad,
    derive_gse157783_microglia_subset_h5ad,
    derive_gse157783_microglia_purity_qc_h5ad,
    load_gse157783_family_sample_metadata,
    parse_gse146639_completecounts_filename,
    parse_gse161045_triplet_filename,
)


class DatasetPrepUtilsTests(unittest.TestCase):
    def test_parse_gse146639_completecounts_filename(self) -> None:
        metadata = parse_gse146639_completecounts_filename(
            "GSM4403092_2018-018_CTR_11-1_S1_completeCounts.txt.gz"
        )

        self.assertEqual(metadata["gsm"], "GSM4403092")
        self.assertEqual(metadata["donor_id"], "2018-018")
        self.assertEqual(metadata["diagnosis"], "CTR")
        self.assertEqual(metadata["sample_id"], "11-1")
        self.assertEqual(metadata["library_id"], "S1")
        self.assertEqual(metadata["technology"], "bc_SmartSeq2")

    def test_collapse_ensembl_matrix_to_gene_symbols(self) -> None:
        matrix = sparse.csr_matrix(
            np.array(
                [
                    [1, 2, 5],
                    [0, 3, 7],
                ]
            )
        )
        collapsed, var, mapping = collapse_ensembl_matrix_to_gene_symbols(
            matrix=matrix,
            ensembl_ids=["ENSG1", "ENSG2", "ENSG3"],
            gene_symbol_map={"ENSG1": "RHBDF2", "ENSG2": "ADAM17", "ENSG3": "RHBDF2"},
        )

        self.assertEqual(collapsed.shape, (2, 2))
        self.assertListEqual(var.index.tolist(), ["RHBDF2", "ADAM17"])
        np.testing.assert_array_equal(
            collapsed.toarray(),
            np.array(
                [
                    [6, 2],
                    [7, 3],
                ]
            ),
        )
        self.assertEqual(mapping.shape[0], 3)

    def test_parse_gse161045_triplet_filename(self) -> None:
        metadata = parse_gse161045_triplet_filename("GSM4888895_PD1_matrix.mtx.gz")

        self.assertEqual(metadata["gsm"], "GSM4888895")
        self.assertEqual(metadata["sample_code"], "PD1")
        self.assertEqual(metadata["diagnosis"], "Parkinson's disease")
        self.assertEqual(metadata["diagnosis_short"], "PD")
        self.assertEqual(metadata["donor_id"], "PD1")
        self.assertEqual(metadata["file_kind"], "matrix")
        self.assertEqual(metadata["technology"], "10x_snRNAseq")

    def test_derive_gse161045_microglia_subset_h5ad(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "gse161045_glia.h5ad"
            output_path = Path(tmp_dir) / "gse161045_microglia.h5ad"

            obs = pd.DataFrame(
                {
                    "donor_id": ["CTRL1", "CTRL1", "PD1", "PD1"],
                    "diagnosis": ["Control", "Control", "Parkinson's disease", "Parkinson's disease"],
                },
                index=["cell1", "cell2", "cell3", "cell4"],
            )
            var_names = ["CSF1R", "CX3CR1", "TYROBP", "C1QA", "P2RY12", "GFAP", "AQP4", "PLP1", "MBP"]
            matrix = sparse.csr_matrix(
                np.array(
                    [
                        [8, 6, 7, 5, 4, 0, 0, 0, 0],
                        [7, 5, 6, 4, 3, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 7, 6, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 8, 7],
                    ],
                    dtype=np.float32,
                )
            )
            adata = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=var_names))
            adata.write_h5ad(input_path)

            summary = derive_gse161045_microglia_subset_h5ad(
                input_path=input_path,
                output_path=output_path,
                min_microglia_score=0.01,
            )

            self.assertEqual(summary.dataset_name, "GSE161045_microglia")
            self.assertEqual(summary.n_cells, 2)
            self.assertTrue(output_path.exists())

            subset = ad.read_h5ad(output_path)
            self.assertListEqual(subset.obs["diagnosis"].tolist(), ["Control", "Control"])
            self.assertTrue((subset.obs["dominant_broad_cell_class"] == "microglia").all())
            self.assertIn("microglia_activation_proxy_state", subset.obs.columns)

    def test_load_gse157783_family_sample_metadata(self) -> None:
        soft_text = "\n".join(
            [
                "^SAMPLE = GSM1",
                "!Sample_title = IPD1",
                "!Sample_geo_accession = GSM1",
                "!Sample_source_name_ch1 = IPD_midbrain",
                "!Sample_characteristics_ch1 = disease state: Idiopathic Parkinson's disease (IPD)",
                "!Sample_characteristics_ch1 = age at death: 84",
                "!Sample_characteristics_ch1 = Sex: F",
                "!Sample_characteristics_ch1 = pmi (h): 25",
                "!Sample_characteristics_ch1 = tissue: Midbrain",
                "!Sample_characteristics_ch1 = molecule subtype: nuclear RNA",
                "^SAMPLE = GSM2",
                "!Sample_title = C1",
                "!Sample_geo_accession = GSM2",
                "!Sample_source_name_ch1 = Control_midbrain",
                "!Sample_characteristics_ch1 = disease state: Healthy control",
                "!Sample_characteristics_ch1 = age at death: 70",
                "!Sample_characteristics_ch1 = Sex: M",
                "!Sample_characteristics_ch1 = pmi (h): 10",
                "!Sample_characteristics_ch1 = tissue: Midbrain",
                "!Sample_characteristics_ch1 = molecule subtype: nuclear RNA",
                "",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            soft_path = Path(tmp_dir) / "GSE157783_family.soft.gz"
            with gzip.open(soft_path, "wt", encoding="utf-8") as handle:
                handle.write(soft_text)

            metadata = load_gse157783_family_sample_metadata(soft_path)

            self.assertEqual(metadata.shape[0], 2)
            self.assertListEqual(metadata["donor_id"].tolist(), ["PD1", "C1"])
            self.assertListEqual(metadata["diagnosis"].tolist(), ["Parkinson's disease", "Control"])
            self.assertListEqual(metadata["sample_geo_accession"].tolist(), ["GSM1", "GSM2"])
            self.assertEqual(metadata.loc[0, "age_at_death"], 84)
            self.assertEqual(metadata.loc[1, "sex"], "M")

    def test_derive_gse157783_microglia_subset_h5ad(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "gse157783_midbrain.h5ad"
            output_path = Path(tmp_dir) / "gse157783_microglia.h5ad"

            obs = pd.DataFrame(
                {
                    "donor_id": ["PD1", "PD1", "C1", "C1"],
                    "diagnosis": ["Parkinson's disease", "Parkinson's disease", "Control", "Control"],
                    "cell_ontology": ["Microglia", "Astrocytes", "Microglia", "Neurons"],
                },
                index=["PD1|cell1", "PD1|cell2", "C1|cell3", "C1|cell4"],
            )
            var_names = ["P2RY12", "CX3CR1", "CSF1R", "TMEM119", "C1QA", "APOE", "GPNMB", "SYK"]
            matrix = sparse.csr_matrix(
                np.array(
                    [
                        [7, 6, 5, 4, 4, 2, 1, 3],
                        [0, 0, 0, 0, 0, 8, 1, 0],
                        [6, 5, 5, 4, 3, 0, 0, 2],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.float32,
                )
            )
            adata = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=var_names))
            adata.write_h5ad(input_path)

            summary = derive_gse157783_microglia_subset_h5ad(
                input_path=input_path,
                output_path=output_path,
            )

            self.assertEqual(summary.dataset_name, "GSE157783_microglia")
            self.assertEqual(summary.n_cells, 2)
            self.assertTrue(output_path.exists())

            subset = ad.read_h5ad(output_path)
            self.assertTrue((subset.obs["cell_ontology"] == "Microglia").all())
            self.assertIn("microglia_activation_proxy_state", subset.obs.columns)
            self.assertEqual(subset.obs["donor_id"].nunique(), 2)

    def test_derive_gse157783_microglia_purity_qc_h5ad(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "gse157783_microglia_input.h5ad"
            output_path = Path(tmp_dir) / "gse157783_microglia_purity_qc.h5ad"

            obs = pd.DataFrame(
                {
                    "donor_id": ["PD1", "PD1", "C1"],
                    "diagnosis": ["Parkinson's disease", "Parkinson's disease", "Control"],
                    "cell_ontology": ["Microglia", "Microglia", "Microglia"],
                },
                index=["PD1|cell1", "PD1|cell2", "C1|cell3"],
            )
            var_names = [
                "P2RY12",
                "CX3CR1",
                "CSF1R",
                "AIF1",
                "TYROBP",
                "FCER1G",
                "APOE",
                "SYK",
                "CD3D",
                "CD3E",
                "TRBC1",
                "IL7R",
            ]
            matrix = sparse.csr_matrix(
                np.array(
                    [
                        [8, 6, 6, 5, 4, 3, 1, 2, 0, 0, 0, 0],
                        [7, 5, 5, 4, 4, 3, 2, 2, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 7, 6, 5, 4],
                    ],
                    dtype=np.float32,
                )
            )
            adata = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=var_names))
            adata.write_h5ad(input_path)

            summary = derive_gse157783_microglia_purity_qc_h5ad(
                input_path=input_path,
                output_path=output_path,
            )

            self.assertEqual(summary.dataset_name, "GSE157783_microglia_purity_qc")
            self.assertEqual(summary.n_cells, 2)
            self.assertTrue(output_path.exists())
            self.assertTrue(output_path.with_name(output_path.stem + "_purity_qc_summary.csv").exists())
            self.assertTrue(output_path.with_name(output_path.stem + "_purity_qc_cell_annotations.csv").exists())

            subset = ad.read_h5ad(output_path)
            self.assertEqual(subset.n_obs, 2)
            self.assertIn("purity_qc_keep", subset.obs.columns)
            self.assertTrue(subset.obs["purity_qc_keep"].all())
            self.assertNotIn("C1|cell3", subset.obs_names.tolist())
            self.assertIn("microglia_activation_proxy_state", subset.obs.columns)


if __name__ == "__main__":
    unittest.main()
