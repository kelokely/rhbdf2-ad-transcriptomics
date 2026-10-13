"""Tests for transparent major-cell-type preparation helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from src.cell_type_annotation_utils import calibrated_target_selection


class CellTypeAnnotationUtilsTests(unittest.TestCase):
    def test_calibrated_selection_keeps_strict_calls_before_ranked_fill(self) -> None:
        obs = pd.DataFrame(
            {
                "cell_type": ["Astrocytes", "Microglia", "Unassigned", "Astrocytes"],
                "marker_score_astrocytes": [1.0, -1.0, 0.8, 0.7],
                "marker_score_microglia": [0.0, 2.0, 0.1, 0.2],
                "marker_score_neurons": [-0.2, 0.0, 0.2, 0.1],
            }
        )
        selected, specificity = calibrated_target_selection(obs, "Astrocytes", 3)
        self.assertEqual(selected.tolist(), [True, False, True, True])
        self.assertGreater(specificity[2], specificity[1])


if __name__ == "__main__":
    unittest.main()
