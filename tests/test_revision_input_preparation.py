import gzip

import numpy as np
import pandas as pd
import pytest

from scripts.prepare_revision_inputs import source_count_qc


def test_source_qc_aligns_cells_and_excludes_spike_ins(tmp_path):
    source = tmp_path/"counts.csv.gz"
    with gzip.open(source, "wt") as handle:
        handle.write("GeneID,a,b\ngene1,2,0\ngene2,0,3\nERCC-1,100,100\n")
    obs = pd.DataFrame({"mouse_id": ["WT_rep1", "5xFAD_rep1"], "genotype": ["WT", "5xFAD"],
                        "region": ["cortex", "cortex"], "source_library_counts": [3, 2]}, index=["b", "a"])
    result = source_count_qc(source, obs)
    np.testing.assert_array_equal(result.assigned_gene_counts, [3, 2])
    np.testing.assert_array_equal(result.source_genes_detected, [1, 1])


def test_source_qc_rejects_wrong_denominator(tmp_path):
    source = tmp_path/"counts.csv.gz"
    with gzip.open(source, "wt") as handle:
        handle.write("GeneID,a\ngene1,2\n")
    obs = pd.DataFrame({"mouse_id": ["WT_rep1"], "genotype": ["WT"], "region": ["cortex"],
                        "source_library_counts": [20]}, index=["a"])
    with pytest.raises(ValueError, match="denominators"):
        source_count_qc(source, obs)
