"""Tests for multi-cell AD extension helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src.multicell_ad_utils import (
    evaluate_ligand_receptor_gate,
    filter_myeloid_like_cells,
    run_continuous_donor_pseudobulk,
    summarize_program_specificity,
    test_donor_rhbdf2_pathology_association,
)


class MulticellADUtilsTests(unittest.TestCase):
    def test_continuous_donor_pseudobulk_writes_results(self) -> None:
        donors = [f"D{i}" for i in range(6)]
        obs_rows = []
        matrix_rows = []
        for donor_index, donor in enumerate(donors):
            for cell_index in range(30):
                detected = cell_index < donor_index + 1
                obs_rows.append({"sampleID": donor, "atscore": "A+T+" if donor_index >= 3 else "A+T-"})
                matrix_rows.append([1.0 if detected else 0.0, float(donor_index + 1), float(6 - donor_index)])
        adata = ad.AnnData(
            X=sparse.csr_matrix(np.asarray(matrix_rows)),
            obs=pd.DataFrame(obs_rows),
            var=pd.DataFrame(index=["RHBDF2", "POS", "NEG"]),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_continuous_donor_pseudobulk(
                adata=adata,
                output_dir=Path(temp_dir),
                candidate_genes=["RHBDF2", "POS", "NEG"],
                donor_column="sampleID",
                pathology_column="atscore",
                minimum_donors=5,
            )
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.sample_metadata.shape[0], 6)
            self.assertTrue((Path(temp_dir) / "donor_continuous_pseudobulk_candidates.csv").exists())

    def test_adam17_donor_pseudobulk_writes_independent_outputs(self) -> None:
        obs_rows = []
        matrix_rows = []
        for donor_index in range(6):
            for cell_index in range(30):
                obs_rows.append({"sampleID": f"D{donor_index}", "atscore": "A+T+" if donor_index >= 3 else "A+T-"})
                matrix_rows.append([
                    float(cell_index < donor_index + 2),
                    float(cell_index < (donor_index + 1) * 3),
                    float(donor_index + 1),
                ])
        adata = ad.AnnData(
            X=sparse.csr_matrix(np.asarray(matrix_rows)),
            obs=pd.DataFrame(obs_rows),
            var=pd.DataFrame(index=["RHBDF2", "ADAM17", "SYK"]),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_continuous_donor_pseudobulk(
                adata=adata,
                output_dir=Path(temp_dir),
                candidate_genes=["RHBDF2", "ADAM17", "SYK"],
                donor_column="sampleID",
                pathology_column="atscore",
                minimum_donors=5,
                anchor_gene="ADAM17",
            )
            self.assertEqual(result.status, "completed")
            self.assertIn("adam17_detection_fraction", result.sample_metadata.columns)
            self.assertIn("effect_per_log1p_adam17_detection_per_1000_cells", result.full_results.columns)
            self.assertTrue((Path(temp_dir) / "donor_continuous_pseudobulk_candidates_adam17.csv").exists())

    def test_targeted_pseudobulk_uses_source_total_umis_for_normalization(self) -> None:
        obs_rows = []
        matrix_rows = []
        for donor_index in range(6):
            for cell_index in range(8):
                obs_rows.append({
                    "sampleID": f"D{donor_index}",
                    "atscore": "A+T+" if donor_index >= 3 else "A+T-",
                    "source_total_umis": 1000.0,
                })
                matrix_rows.append([
                    float(cell_index <= donor_index),
                    2.0,
                ])
        adata = ad.AnnData(
            X=sparse.csr_matrix(np.asarray(matrix_rows)),
            obs=pd.DataFrame(obs_rows),
            var=pd.DataFrame(index=["RHBDF2", "TNF"]),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_continuous_donor_pseudobulk(
                adata=adata,
                output_dir=Path(temp_dir),
                candidate_genes=["RHBDF2", "TNF"],
                donor_column="sampleID",
                pathology_column="atscore",
                minimum_donors=5,
                minimum_cells_per_donor=1,
            )
        self.assertTrue(result.sample_metadata["normalization_library_source"].eq("summed_source_total_umis").all())
        self.assertTrue(np.allclose(result.sample_metadata["normalization_library_size"], 8000.0))
        self.assertAlmostEqual(result.log_cpm.loc[0, "TNF"], np.log1p(16.0 / 8000.0 * 1_000_000.0))

    def test_targeted_full_qc_totals_are_not_panel_totals(self) -> None:
        from src.multicell_ad_utils import _aggregate_by_donor
        obj = ad.AnnData(X=sparse.csr_matrix([[1, 2], [0, 3]]),
                         obs=pd.DataFrame({"donor": ["d", "d"], "pathology": ["AD", "AD"],
                                           "total_counts": [1000, 2000]}, index=["a", "b"]),
                         var=pd.DataFrame(index=["RHBDF2", "TNF"]))
        _, metadata = _aggregate_by_donor(obj, "donor", "pathology")
        self.assertEqual(metadata.normalization_library_size.iloc[0], 3000)
        self.assertEqual(metadata.raw_library_size.iloc[0], 6)

    def test_hc3_vector_model_matches_statsmodels(self) -> None:
        import statsmodels.api as sm
        from src.multicell_ad_utils import _fit_continuous_matrix_model
        x = np.linspace(0.2, 2, 15)
        covariates = pd.DataFrame({"condition": [0]*7 + [1]*8})
        y = (x + np.sin(x*8)*x)[:, None]
        result = _fit_continuous_matrix_model(y, ["GENE"], x, covariates, "effect", covariance_method="HC3")
        fit = sm.OLS(y[:, 0], np.column_stack([np.ones(len(x)), x, covariates])).fit(cov_type="HC3", use_t=True)
        self.assertAlmostEqual(result.effect.iloc[0], fit.params[1], places=10)
        self.assertAlmostEqual(result.standard_error.iloc[0], fit.bse[1], places=10)
        self.assertAlmostEqual(result.pvals.iloc[0], fit.pvalues[1], places=10)

    def test_low_cell_donor_profiles_are_excluded(self) -> None:
        obs, counts = [], []
        for i, size in enumerate([1, 30, 30, 30, 30, 30, 30]):
            for j in range(size):
                obs.append({"donor": str(i), "pathology": "Control"})
                counts.append([float(j <= i), float(i + 2), float(9-i)])
        obj = ad.AnnData(X=sparse.csr_matrix(counts), obs=pd.DataFrame(obs),
                         var=pd.DataFrame(index=["RHBDF2", "A", "B"]))
        with tempfile.TemporaryDirectory() as directory:
            result = run_continuous_donor_pseudobulk(obj, Path(directory), ["A", "B"], "donor", "pathology")
        self.assertEqual(len(result.sample_metadata), 6)
        self.assertNotIn("0", result.sample_metadata.donor_id.tolist())

    def test_program_specificity_requires_fdr_support(self) -> None:
        frame = pd.DataFrame(
            {
                "program": ["p1", "p1", "p2"],
                "cell_type": ["Microglia", "Astrocytes", "Microglia"],
                "effect": [1.0, 0.5, -0.2],
                "pvals_adj_global": [0.01, 0.02, 0.8],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = summarize_program_specificity(frame, Path(temp_dir) / "summary.csv")
        classes = result.set_index("program")["classification"].to_dict()
        self.assertEqual(classes["p1"], "shared_glial")
        self.assertEqual(classes["p2"], "unsupported_at_donor_fdr")

    def test_myeloid_like_filter_removes_two_marker_cells(self) -> None:
        adata = ad.AnnData(
            X=sparse.csr_matrix([[1, 1, 1], [1, 1, 0], [0, 0, 0]]),
            obs=pd.DataFrame(index=["a", "b", "c"]),
            var=pd.DataFrame(index=["RHBDF2", "TYROBP", "AIF1"]),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            filtered = filter_myeloid_like_cells(
                adata,
                marker_genes=["TYROBP", "AIF1"],
                output_path=Path(temp_dir) / "summary.csv",
                minimum_markers=2,
            )
        self.assertEqual(filtered.n_obs, 2)

    def test_ligand_receptor_gate_excludes_self_association(self) -> None:
        programs = pd.DataFrame(
            {
                "cell_type": ["Astrocytes", "OPCs"],
                "pvals_adj_global": [0.01, 0.01],
            }
        )
        astro = pd.DataFrame(
            {
                "names": ["RHBDF2"],
                "pvals_adj": [0.001],
                "self_association": [True],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = evaluate_ligand_receptor_gate(programs, astro, Path(temp_dir) / "gate.csv")
        self.assertFalse(bool(result.iloc[0]["gate_passed"]))

    def test_pathology_association_returns_one_row_per_cell_type(self) -> None:
        frame = pd.DataFrame(
            {
                "cell_type": ["Astrocytes"] * 6,
                "atscore": ["A-T-", "A-T-", "A+T-", "A+T-", "A+T+", "A+T+"],
                "rhbdf2_detection_rate_log1p_per_1000": [0.1, 0.2, 0.8, 0.9, 1.5, 1.6],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = test_donor_rhbdf2_pathology_association(
                frame, "atscore", Path(temp_dir) / "pathology.csv"
            )
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.iloc[0]["cell_type"], "Astrocytes")


if __name__ == "__main__":
    unittest.main()
