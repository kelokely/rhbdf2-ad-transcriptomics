"""Plot data must preserve donor identity, library totals, and adjusted estimates."""

import numpy as np
import pandas as pd
import pytest

from scripts.jad_detailed_figures import donor_abundance, partial_coordinates


def test_donor_alignment_and_full_library_denominator():
    metadata = pd.DataFrame({"donor_id": ["a", "b"], "library_counts": [1000, 2000]})
    selected = metadata.iloc[[1, 0]].reset_index(drop=True)
    counts = pd.DataFrame({"TNFAIP3": [2, 8]})
    result = donor_abundance(metadata, counts, selected)
    np.testing.assert_allclose(result.TNFAIP3, np.log1p([4000, 2000]))


@pytest.mark.parametrize("case", ["duplicate", "missing", "denominator", "negative", "nonfinite", "length"])
def test_invalid_donor_coordinates_rejected(case):
    metadata = pd.DataFrame({"donor_id": ["a", "b"], "library_counts": [1000, 2000]})
    selected = metadata.copy()
    counts = pd.DataFrame({"gene": [2.0, 8.0]})
    if case == "duplicate": selected.loc[1, "donor_id"] = "a"
    if case == "missing": selected.loc[1, "donor_id"] = "z"
    if case == "denominator": selected.loc[1, "library_counts"] = 10
    if case == "negative": counts.loc[0, "gene"] = -1
    if case == "nonfinite": counts.loc[0, "gene"] = np.inf
    if case == "length": counts = counts.iloc[:1]
    with pytest.raises(ValueError): donor_abundance(metadata, counts, selected)


def test_added_variable_slope_matches_full_model():
    rng = np.random.default_rng(77)
    design = pd.DataFrame({"intercept": np.ones(20), "anchor": rng.normal(size=20), "depth": rng.normal(size=20)})
    response = 2 + design.anchor.to_numpy()*1.5 + design.depth.to_numpy()*3 + rng.normal(size=20)
    x, y, slope = partial_coordinates(design, response)
    beta = np.linalg.lstsq(design, response, rcond=None)[0][1]
    assert slope == pytest.approx(beta)
    assert x.mean() == pytest.approx(0, abs=1e-12)
    assert y.mean() == pytest.approx(0, abs=1e-12)


def test_partial_coordinates_reject_unidentifiable_anchor():
    design = pd.DataFrame({"intercept": [1, 1, 1], "anchor": [2, 2, 2]})
    with pytest.raises(ValueError, match="independent variation"):
        partial_coordinates(design, [1, 2, 3])


def test_partial_coordinates_reject_response_shape():
    design = pd.DataFrame({"intercept": [1, 1, 1], "anchor": [1, 2, 3]})
    with pytest.raises(ValueError, match="finite vector"):
        partial_coordinates(design, [1, 2])
