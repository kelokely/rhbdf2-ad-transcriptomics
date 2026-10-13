"""Tests for conservative ligand-receptor replication accounting."""

from __future__ import annotations

import unittest

import pandas as pd

from src.ligand_receptor_utils import summarize_lr_replication


class LigandReceptorUtilsTests(unittest.TestCase):
    def test_replication_requires_distinct_cohorts_and_matching_cell_pair(self) -> None:
        edges = pd.DataFrame(
            {
                "source_genesymbol": ["LY86", "LY86", "LY86"],
                "target_genesymbol": ["CD180", "CD180", "CD180"],
                "sender_cell_type": ["Unassigned", "Unassigned", "Astrocytes"],
                "receiver_cell_type": ["Microglia", "Microglia", "Microglia"],
                "node_direction": ["both_positive", "both_positive", "both_positive"],
                "cohort": ["A", "B", "B"],
                "joint_node_strength": [2.0, 3.0, 4.0],
            }
        )
        result = summarize_lr_replication(edges)
        replicated = result[result["replication_status"].eq("replicated_two_or_more_cohorts")]
        self.assertEqual(replicated.shape[0], 1)
        self.assertEqual(replicated.iloc[0]["replicating_cohort_count"], 2)
        self.assertEqual(replicated.iloc[0]["sender_cell_type"], "Unassigned")


if __name__ == "__main__":
    unittest.main()
