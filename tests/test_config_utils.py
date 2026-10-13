from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config_utils import load_config


VALID_CONFIG = """
dataset_name: GSE243292
input_path: data/raw/example.h5ad
output_root: results
report_root: reports
gene_of_interest: RHBDF2
candidate_genes:
  - RHBDF2
metadata:
  diagnosis_column: null
  cluster_column: null
  donor_column: null
state_definition:
  method: median
  quantile: 0.75
  new_column_name: RHBDF2_state
  high_label: RHBDF2_high
  low_label: RHBDF2_low
differential_expression:
  method: wilcoxon
  group_column: RHBDF2_state
  group_a: RHBDF2_high
  group_b: RHBDF2_low
  top_n_export: 25
preprocessing:
  matrix_source: X
  layer_name: null
  normalize_total: true
  target_sum: 10000
  log1p: true
pseudobulk:
  enabled: true
  donor_column: sampleID
  group_column: RHBDF2_state
  group_a: RHBDF2_high
  group_b: RHBDF2_low
  paired_only: true
  min_cells_per_sample: 10
  min_donors_per_group: 2
  normalization_target_sum: 1000000
  top_n_export: 25
enrichment:
  enabled: true
  organism: hsapiens
  sources:
    - GO:BP
    - REAC
  user_threshold: 0.05
  significance_threshold_method: g_SCS
  domain_scope: custom
  use_dataset_background: true
  no_iea: false
  max_terms_report: 5
enrichr:
  enabled: true
  organism: human
  libraries:
    - MSigDB_Hallmark_2020
    - CellMarker_2024
  cutoff: 0.05
  use_dataset_background: true
  max_terms_report: 5
evidence:
  enabled: true
  metadata_association_columns:
    - atscore
    - apoe
  module_weights:
    cell_level: 1.0
    pseudobulk: 3.0
    continuous: 2.0
    detection: 1.0
    positive_expression: 1.0
    subpopulation_pseudobulk: 2.0
    protein_context: 1.0
    genetics: 1.5
    brain_context: 1.0
continuous_evidence:
  enabled: true
  donor_column: sampleID
  top_n_export: 25
  min_positive_cells: 5
protein_evidence:
  enabled: true
  organism_id: 9606
  reviewed_only: true
  timeout_seconds: 30
genetics_evidence:
  enabled: true
  disease_focus_label: Alzheimer\'s disease
  disease_keywords:
    - alzheimer
    - dementia
  gwas_trait_page_size: 500
  gwas_association_page_size: 200
  gwas_max_association_pages: 5
  opentargets_page_size: 500
  timeout_seconds: 30
brain_context_evidence:
  enabled: true
  context_label: Alzheimer\'s disease brain proteomics
  expression_atlas_experiment: E-PROT-61
  case_labels:
    - Alzheimer\'s disease
  control_labels:
    - normal
  timeout_seconds: 30
plots:
  save_umap_if_available: true
  save_violin: true
  save_dotplot: true
  save_matrixplot: true
"""


class ConfigUtilsTests(unittest.TestCase):
    def test_load_config_resolves_paths_relative_to_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            config_dir = project_root / "configs"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "example.yaml"
            config_path.write_text(VALID_CONFIG, encoding="utf-8")

            config = load_config(config_path)

            self.assertEqual(Path(config["input_path"]).resolve(), (project_root / "data/raw/example.h5ad").resolve())
            self.assertEqual(Path(config["output_root"]).resolve(), (project_root / "results").resolve())
            self.assertEqual(Path(config["report_root"]).resolve(), (project_root / "reports").resolve())
            self.assertEqual(config["run_name"], "GSE243292_run")
            self.assertTrue(config["preprocessing"]["normalize_total"])
            self.assertEqual(config["preprocessing"]["matrix_source"], "X")
            self.assertTrue(config["pseudobulk"]["enabled"])
            self.assertIsNone(config["pseudobulk"]["within_subpopulation_min_cells_per_sample"])
            self.assertIsNone(config["pseudobulk"]["within_subpopulation_min_donors_per_group"])
            self.assertTrue(config["enrichment"]["enabled"])
            self.assertIn("GO:BP", config["enrichment"]["sources"])
            self.assertTrue(config["enrichr"]["enabled"])
            self.assertIn("MSigDB_Hallmark_2020", config["enrichr"]["libraries"])
            self.assertEqual(config["evidence"]["metadata_association_columns"], ["atscore", "apoe"])
            self.assertEqual(config["evidence"]["module_weights"]["pseudobulk"], 3.0)
            self.assertTrue(config["continuous_evidence"]["enabled"])
            self.assertEqual(config["continuous_evidence"]["donor_column"], "sampleID")
            self.assertTrue(config["protein_evidence"]["enabled"])
            self.assertEqual(config["protein_evidence"]["organism_id"], 9606)
            self.assertTrue(config["genetics_evidence"]["enabled"])
            self.assertEqual(config["genetics_evidence"]["disease_focus_label"], "Alzheimer's disease")
            self.assertIn("alzheimer", config["genetics_evidence"]["disease_keywords"])
            self.assertTrue(config["brain_context_evidence"]["enabled"])
            self.assertEqual(config["brain_context_evidence"]["context_label"], "Alzheimer's disease brain proteomics")
            self.assertEqual(config["brain_context_evidence"]["expression_atlas_experiment"], "E-PROT-61")

    def test_load_config_raises_for_missing_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "bad.yaml"
            config_path.write_text("dataset_name: only_one_field\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_config(config_path)

    def test_load_config_rejects_invalid_boolean_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "bad_bool.yaml"
            config_path.write_text(
                VALID_CONFIG.replace("save_violin: true", "save_violin: yes_please"),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_config(config_path)

    def test_load_config_rejects_invalid_quantile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "bad_quantile.yaml"
            config_path.write_text(
                VALID_CONFIG.replace("method: median", "method: quantile").replace("quantile: 0.75", "quantile: 1.2"),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_config(config_path)

    def test_load_config_rejects_invalid_matrix_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "bad_matrix_source.yaml"
            config_path.write_text(
                VALID_CONFIG.replace("matrix_source: X", "matrix_source: counts_layer"),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_config(config_path)

    def test_load_config_rejects_negative_module_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "bad_weight.yaml"
            config_path.write_text(
                VALID_CONFIG.replace("pseudobulk: 3.0", "pseudobulk: -1.0"),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_config(config_path)

    def test_load_config_rejects_invalid_within_subpopulation_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "bad_subpopulation_threshold.yaml"
            config_path.write_text(
                VALID_CONFIG.replace(
                    "  top_n_export: 25\n",
                    "  top_n_export: 25\n  within_subpopulation_min_cells_per_sample: 0\n",
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_config(config_path)

    def test_load_config_resolves_local_proteomics_matrix_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            config_dir = project_root / "configs"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "pd.yaml"
            config_path.write_text(
                VALID_CONFIG.replace(
                    "  expression_atlas_experiment: E-PROT-61\n",
                    "  local_matrix_path: data/external/pd_proteomics.xlsx\n"
                    "  local_matrix_sheet_name: Log scale (A.U)\n"
                    "  local_matrix_region_column: Brain Region\n"
                    "  local_matrix_group_column: Group\n"
                    "  local_matrix_sample_id_column: Sample ID\n"
                    "  local_matrix_values_are_log2: true\n",
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(
                Path(config["brain_context_evidence"]["local_matrix_path"]).resolve(),
                (project_root / "data/external/pd_proteomics.xlsx").resolve(),
            )
            self.assertTrue(config["brain_context_evidence"]["local_matrix_values_are_log2"])


if __name__ == "__main__":
    unittest.main()
