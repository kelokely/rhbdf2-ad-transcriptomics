from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.phased_study_utils import (
    build_candidate_phase_outputs,
    build_module_phase_outputs,
    build_network_phase_outputs,
    build_run_manifest,
    load_run_artifacts,
    load_phased_study_config,
    build_run_specs,
)


class PhasedStudyUtilsTests(unittest.TestCase):
    def test_load_phased_study_config_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "configs").mkdir()
            config_path = root / "configs" / "study.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "study_name": "demo",
                        "output_dir": "results/demo",
                        "report_path": "report/demo.md",
                        "backbone_context": "AD",
                        "extension_context": "PD",
                        "anchor_gene": "RHBDF2",
                        "contexts": {
                            "AD": {"runs": [{"path": "runs/ad"}]},
                            "PD": {"runs": [{"path": "runs/pd"}]},
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_phased_study_config(config_path)

            resolved_root = root.resolve()
            self.assertTrue(Path(config["output_dir"]).resolve().is_relative_to(resolved_root))
            self.assertTrue(Path(config["report_path"]).resolve().is_relative_to(resolved_root))
            self.assertTrue(Path(config["contexts"]["AD"]["runs"][0]["path"]).resolve().is_relative_to(resolved_root))

    def test_build_phased_outputs_classifies_shared_and_context_specific_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_specs = []
            for context_name, label in [("AD", "ad_one"), ("AD", "ad_two"), ("PD", "pd_one")]:
                run_dir = tmp_path / label
                (run_dir / "summaries").mkdir(parents=True)
                (run_dir / "tables").mkdir(parents=True)
                (run_dir / "activity_proxies").mkdir(parents=True)
                (run_dir / "external_evidence").mkdir(parents=True)
                (run_dir / "summaries" / "config_used.yaml").write_text(
                    yaml.safe_dump({"dataset_name": label, "gene_of_interest": "RHBDF2"}),
                    encoding="utf-8",
                )
                pd.DataFrame(
                    [
                        {"RHBDF2_state": "RHBDF2_low", "count": 70},
                        {"RHBDF2_state": "RHBDF2_high", "count": 30},
                    ]
                ).to_csv(run_dir / "summaries" / "state_summary.csv", index=False)
                pd.DataFrame(
                    [
                        {
                            "gene": "RHBDF2",
                            "cell_level_significant_fdr_0_05": True,
                            "continuous_significant_fdr_0_05": True,
                            "pseudobulk_significant_fdr_0_05": True,
                            "subpopulation_supported": True,
                            "protein_context_supported": True,
                            "genetics_supported": True,
                            "brain_context_supported": True,
                            "public_proteomics_overlap_supported": False,
                            "cell_level_logfoldchange": 1.5,
                            "continuous_slope_per_log1p_rhbdf2": 1.0,
                            "pseudobulk_effect_log1p_cpm": 0.9,
                            "subpopulation_pseudobulk_mean_effect_log1p_cpm": 0.4,
                            "weighted_transcript_evidence_score": 7.0,
                            "weighted_context_evidence_score": 3.0,
                            "weighted_total_evidence_score": 10.0,
                            "reactome_pathway_count": 1,
                            "annotated_ptm_feature_count": 6,
                            "candidate_panel_interaction_count": 1,
                            "candidate_panel_ptm_out_count": 0,
                            "candidate_panel_ptm_in_count": 0,
                        },
                        {
                            "gene": "TYROBP",
                            "cell_level_significant_fdr_0_05": context_name == "AD",
                            "continuous_significant_fdr_0_05": context_name == "AD",
                            "pseudobulk_significant_fdr_0_05": False,
                            "subpopulation_supported": False,
                            "protein_context_supported": True,
                            "genetics_supported": True,
                            "brain_context_supported": True,
                            "public_proteomics_overlap_supported": False,
                            "cell_level_logfoldchange": 0.8 if context_name == "AD" else 0.1,
                            "continuous_slope_per_log1p_rhbdf2": 0.2 if context_name == "AD" else 0.01,
                            "pseudobulk_effect_log1p_cpm": 0.05,
                            "subpopulation_pseudobulk_mean_effect_log1p_cpm": 0.02,
                            "weighted_transcript_evidence_score": 4.0 if context_name == "AD" else 1.0,
                            "weighted_context_evidence_score": 2.0,
                            "weighted_total_evidence_score": 6.0 if context_name == "AD" else 3.0,
                            "reactome_pathway_count": 6,
                            "annotated_ptm_feature_count": 3,
                            "candidate_panel_interaction_count": 1,
                            "candidate_panel_ptm_out_count": 0,
                            "candidate_panel_ptm_in_count": 4,
                        },
                        {
                            "gene": "SYK",
                            "cell_level_significant_fdr_0_05": context_name == "PD",
                            "continuous_significant_fdr_0_05": context_name == "PD",
                            "pseudobulk_significant_fdr_0_05": False,
                            "subpopulation_supported": False,
                            "protein_context_supported": True,
                            "genetics_supported": True,
                            "brain_context_supported": False,
                            "public_proteomics_overlap_supported": False,
                            "cell_level_logfoldchange": 0.2 if context_name == "AD" else 0.9,
                            "continuous_slope_per_log1p_rhbdf2": 0.01 if context_name == "AD" else 0.3,
                            "pseudobulk_effect_log1p_cpm": 0.01,
                            "subpopulation_pseudobulk_mean_effect_log1p_cpm": 0.01,
                            "weighted_transcript_evidence_score": 1.0 if context_name == "AD" else 5.0,
                            "weighted_context_evidence_score": 2.0,
                            "weighted_total_evidence_score": 3.0 if context_name == "AD" else 7.0,
                            "reactome_pathway_count": 21,
                            "annotated_ptm_feature_count": 31,
                            "candidate_panel_interaction_count": 2,
                            "candidate_panel_ptm_out_count": 8,
                            "candidate_panel_ptm_in_count": 0,
                        },
                    ]
                ).to_csv(run_dir / "tables" / "candidate_evidence_summary.csv", index=False)
                pd.DataFrame(
                    [
                        {
                            "module_name": "trem2_tyrobp_syk_response",
                            "cell_level_delta_high_minus_low": 0.12 if context_name == "AD" else 0.01,
                            "cell_level_padj": 0.001 if context_name == "AD" else 0.8,
                            "donor_paired_mean_delta": 0.04 if context_name == "AD" else 0.005,
                            "donor_paired_padj": 0.2 if context_name == "AD" else 0.9,
                        }
                    ]
                ).to_csv(run_dir / "activity_proxies" / "pathway_activity_proxy_summary.csv", index=False)
                (run_dir / "activity_proxies" / "pathway_activity_proxy_gene_sets.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "trem2_tyrobp_syk_response": {
                                "requested_genes": ["TREM2", "TYROBP", "SYK"],
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                pd.DataFrame(
                    [
                        {
                            "source": "s1",
                            "target": "t1",
                            "source_genesymbol": "RHBDF2",
                            "target_genesymbol": "SYK",
                            "is_directed": True,
                        }
                    ]
                ).to_csv(run_dir / "external_evidence" / "protein_candidate_network_edges.csv", index=False)

            config_path = tmp_path / "configs.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "study_name": "toy",
                        "output_dir": str(tmp_path / "out"),
                        "backbone_context": "AD",
                        "extension_context": "PD",
                        "anchor_gene": "RHBDF2",
                        "contexts": {
                            "AD": {
                                "min_replicated_continuous_runs": 2,
                                "runs": [
                                    {"label": "ad_one", "path": str(tmp_path / "ad_one")},
                                    {"label": "ad_two", "path": str(tmp_path / "ad_two")},
                                ],
                            },
                            "PD": {
                                "min_replicated_continuous_runs": 1,
                                "runs": [{"label": "pd_one", "path": str(tmp_path / "pd_one")}],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_phased_study_config(config_path)
            specs = build_run_specs(config)
            artifacts = load_run_artifacts(specs)

            manifest = build_run_manifest(artifacts, tmp_path / "manifest.csv")
            candidate_phase = build_candidate_phase_outputs(
                artifacts=artifacts,
                backbone_context="AD",
                extension_context="PD",
                anchor_gene="RHBDF2",
                output_dir=tmp_path / "phase1",
            )
            module_phase = build_module_phase_outputs(
                artifacts=artifacts,
                backbone_context="AD",
                extension_context="PD",
                output_dir=tmp_path / "phase2",
                delta_support_threshold=0.02,
            )
            network_phase = build_network_phase_outputs(
                artifacts=artifacts,
                candidate_phase=candidate_phase,
                module_phase=module_phase,
                backbone_context="AD",
                extension_context="PD",
                anchor_gene="RHBDF2",
                output_dir=tmp_path / "phase3",
            )

            self.assertEqual(len(manifest), 3)
            ranked = candidate_phase["ranked"]
            self.assertEqual(
                ranked.loc[ranked["gene"] == "RHBDF2", "phase1_classification"].iloc[0],
                "shared_core_anchor",
            )
            self.assertEqual(
                ranked.loc[ranked["gene"] == "TYROBP", "phase1_classification"].iloc[0],
                "backbone_specific_periphery",
            )
            self.assertIn(
                ranked.loc[ranked["gene"] == "SYK", "phase1_classification"].iloc[0],
                {"extension_specific_periphery", "extension_supported_only"},
            )
            network_priority = network_phase["priority"]
            self.assertEqual(
                network_priority.loc[network_priority["gene"] == "RHBDF2", "network_role"].iloc[0],
                "core_anchor",
            )
            self.assertTrue((tmp_path / "phase1" / "phase1_summary.md").exists())
            self.assertTrue((tmp_path / "phase1" / "phase1_candidate_support_scatter.png").exists())
            self.assertTrue((tmp_path / "phase2" / "phase2_summary.md").exists())
            self.assertTrue((tmp_path / "phase2" / "phase2_module_support_comparison.png").exists())
            self.assertTrue((tmp_path / "phase3" / "phase3_summary.md").exists())
            self.assertTrue((tmp_path / "phase3" / "phase3_priority_score_scatter.png").exists())
            self.assertTrue((tmp_path / "phase3" / "prioritized_network_edges.csv").exists())


if __name__ == "__main__":
    unittest.main()
