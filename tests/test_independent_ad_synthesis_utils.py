"""Tests for conservative cross-cohort replication rules."""

from __future__ import annotations

import unittest

import pandas as pd

from src.independent_ad_synthesis_utils import build_ligand_receptor_gate, summarize_replication


class IndependentADSynthesisTests(unittest.TestCase):
    def test_replication_requires_distinct_cohorts(self) -> None:
        frame = pd.DataFrame(
            {
                "cell_type": ["Astrocytes", "Astrocytes", "Astrocytes"],
                "names": ["SYK", "SYK", "SYK"],
                "cohort": ["A", "A", "B"],
                "analysis_stratum": ["R1", "R2", "all"],
                "effect": [1.0, 1.2, 0.8],
                "p": [0.001, 0.002, 0.01],
                "q": [0.01, 0.02, 0.03],
            }
        )
        summary = summarize_replication(frame, "names", "effect", "p", "q")
        self.assertEqual(summary.iloc[0]["replication_tier"], "replicated_fdr")
        self.assertEqual(summary.iloc[0]["cohorts_tested"], 2)
        self.assertEqual(summary.iloc[0]["contexts_tested"], 3)

    def test_gate_stays_closed_without_nonself_replication(self) -> None:
        candidates = pd.DataFrame({"replication_tier": ["replicated_nominal"]})
        programs = pd.DataFrame({"replication_tier": ["single_cohort_fdr"]})
        gate = build_ligand_receptor_gate(candidates, programs, included_donor_cohorts=2)
        self.assertEqual(gate.iloc[0]["status"], "withheld")


if __name__ == "__main__":
    unittest.main()
