"""Tests for sample-aware TNF module and mouse comparison utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src.tnf_cross_species_utils import (
    aggregate_backed_population,
    compare_species,
    exact_two_group_test,
    fit_module_associations,
    module_scores,
    read_one_to_one_orthologs,
)


class TNFCrossSpeciesUtilsTests(unittest.TestCase):
    def test_targeted_counts_use_full_library_and_animal_unit(self) -> None:
        obs = pd.DataFrame({
            "cell_type": ["Microglia", "Microglia", "Microglia", "Microglia", "Other"],
            "mouse_id": ["WT_rep1", "WT_rep1", "5xFAD_rep1", "5xFAD_rep1", "WT_rep1"],
            "genotype": ["WT", "WT", "5xFAD", "5xFAD", "WT"],
            "region": ["cortex", "cortex", "cortex", "cortex", "cortex"],
            "source_library_counts": [100, 100, 200, 200, 100],
        }, index=["cell1", "cell2", "cell3", "cell4", "cell5"])
        obj = ad.AnnData(
            X=sparse.csr_matrix([[1, 2], [3, 4], [2, 1], [2, 3], [9, 9]]),
            obs=obs,
            var=pd.DataFrame(index=["Rhbdf2", "Tnf"]),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "targeted.h5ad"
            obj.write_h5ad(path)
            metadata, log_cpm = aggregate_backed_population(
                path, "cell_type", "Microglia", "mouse_id", "genotype",
                ["Rhbdf2", "Tnf"], "region", "cortex", chunk_size=1,
            )
        self.assertEqual(metadata["sample_id"].tolist(), ["5xFAD_rep1", "WT_rep1"])
        self.assertEqual(metadata["n_cells"].tolist(), [2, 2])
        self.assertAlmostEqual(log_cpm.loc[1, "Rhbdf2"], np.log1p(4 / 200 * 1e6))
        self.assertAlmostEqual(metadata.loc[0, "Rhbdf2_detection_fraction"], 1.0)

    def test_module_coverage_excludes_anchor(self) -> None:
        log_cpm = pd.DataFrame({
            "RHBDF2": [1, 2, 3, 4, 5, 6],
            "TNF": [1, 2, 3, 4, 5, 6],
            "NFKBIA": [2, 1, 4, 3, 6, 5],
            "TNFAIP3": [1, 1, 2, 2, 3, 3],
            "BIRC3": [3, 2, 5, 4, 7, 6],
        })
        scores, coverage = module_scores(
            log_cpm, {"tnf": ["RHBDF2", "TNF", "NFKBIA", "TNFAIP3", "BIRC3"]},
            "RHBDF2", 4, 0.5, 0.15,
        )
        self.assertEqual(coverage.loc[0, "n_requested_after_anchor_exclusion"], 4)
        self.assertNotIn("RHBDF2", coverage.loc[0, "scored_genes"])
        self.assertIn("tnf", scores.columns)

    def test_association_reports_covariates_and_leave_one_out(self) -> None:
        fractions = np.array([0.06, 0.10, 0.14, 0.20, 0.28, 0.34, 0.42, 0.50])
        predictor = np.log1p(1000 * fractions)
        libraries = np.array([100, 120, 90, 110, 130, 115, 140, 125], dtype=float)
        condition = np.array(["WT"] * 4 + ["5xFAD"] * 4)
        metadata = pd.DataFrame({
            "condition": condition,
            "n_cells": [40] * 8,
            "source_library_counts": libraries * 40,
            "Rhbdf2_detection_fraction": fractions,
        })
        scores = pd.DataFrame({"tnf": 2 * predictor + 0.1 * np.log(libraries) + (condition == "5xFAD")})
        result = fit_module_associations(metadata, scores, "Rhbdf2", "test", "Microglia cortex", 6)
        self.assertEqual(len(result), 1)
        self.assertTrue(bool(result.loc[0, "condition_adjusted"]))
        self.assertTrue(bool(result.loc[0, "library_adjusted"]))
        self.assertEqual(result.loc[0, "loo_n_evaluable"], 8)
        self.assertAlmostEqual(result.loc[0, "loo_sign_concordance"], 1.0)

    def test_exact_genotype_test_uses_70_animal_allocations(self) -> None:
        metadata = pd.DataFrame({"condition": ["WT"] * 4 + ["5xFAD"] * 4})
        scores = pd.DataFrame({"module": [0, 0, 0, 0, 1, 1, 1, 1]})
        result = exact_two_group_test(metadata, scores, "WT", "5xFAD")
        self.assertEqual(result.loc[0, "n_permutations"], 70)
        self.assertAlmostEqual(result.loc[0, "pval_exact"], 2 / 70)

    def test_cross_species_requires_replicated_human_and_mouse_fdr(self) -> None:
        human = pd.DataFrame({
            "cohort": ["A", "B"], "population": ["Microglia"] * 2,
            "anchor": ["RHBDF2"] * 2, "module": ["tnf"] * 2,
            "effect": [0.4, 0.6], "qval_within_cohort_population_anchor": [0.01, 0.02],
        })
        mouse = pd.DataFrame({
            "population": ["Microglia cortex"], "anchor": ["Rhbdf2"],
            "module": ["tnf"], "effect": [0.2], "qval_within_cohort_population_anchor": [0.3],
        })
        result = compare_species(human, mouse)
        self.assertEqual(result.loc[0, "human_replicated_direction"], 1)
        self.assertFalse(bool(result.loc[0, "cross_species_fdr_support"]))

    def test_orthology_rejects_nonunique_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "orthology.tsv"
            path.write_text("MGI:1\tA\t1\tHGNC:1\tX\t1\nMGI:2\tB\t2\tHGNC:2\tX\t2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-unique"):
                read_one_to_one_orthologs(path)


if __name__ == "__main__":
    unittest.main()
