import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import statsmodels.api as sm

from src.jad_revision_utils import aggregate, full_libraries, design_matrix, fit_many, scores_for_modules
from run_jad_revision import contexts


class RevisionTests(unittest.TestCase):
    def test_aggregation_preserves_full_library_denominator_and_detection(self):
        obj=ad.AnnData(sparse.csr_matrix([[1,0],[0,2],[3,0],[1,1]]),
                      obs=pd.DataFrame({"donor":["a","a","b","b"],"pathology":["N","N","AD","AD"],
                                        "total_counts":[100,200,300,400]},index=["a1","a2","b1","b2"]),
                      var=pd.DataFrame(index=["RHBDF2","ADAM17"]))
        libraries,_=full_libraries(obj)
        metadata,counts,detection=aggregate(obj,np.ones(4,dtype=bool),"donor","pathology",libraries,chunk_size=1)
        np.testing.assert_array_equal(metadata.library_counts,[300,700])
        np.testing.assert_array_equal(counts.RHBDF2,[1,4])
        np.testing.assert_array_equal(detection.RHBDF2,[1,2])
        np.testing.assert_array_equal(metadata.RHBDF2_detection_fraction,[.5,1])

    def test_robust_fit_matches_statsmodels_and_constant_gene_is_not_significant(self):
        rng=np.random.default_rng(9)
        m=pd.DataFrame({"RHBDF2_detection_fraction":rng.uniform(.1,.8,30),"library_counts":rng.uniform(1000,10000,30),
                        "n_cells":rng.integers(20,300,30),"condition":["AD"]*15+["N"]*15})
        design=design_matrix(m,"RHBDF2")
        v=pd.DataFrame({"test":2*design.anchor+rng.normal(size=30),"constant":np.ones(30)})
        result=fit_many(v,design)
        expected=sm.OLS(v.test,design).fit(cov_type="HC3",use_t=True)
        self.assertAlmostEqual(result.iloc[0].standard_error,expected.bse["anchor"],places=8)
        self.assertAlmostEqual(result.iloc[0].pval,expected.pvalues["anchor"],places=8)
        self.assertEqual(result.iloc[1].pval,1)
        self.assertFalse(result.iloc[1].tested)

    def test_modules_exclude_anchor_and_record_partial_coverage(self):
        values=pd.DataFrame({"RHBDF2":[0,1,2],"A":[0,2,3],"B":[1,3,4]})
        scores,coverage=scores_for_modules(values,{"proxy":["RHBDF2","A","B","missing"]},"RHBDF2",2,.5)
        self.assertEqual(len(scores),3)
        self.assertEqual(coverage.iloc[0].genes,"A;B")
        self.assertEqual(coverage.iloc[0].interpretation,"partial gene-set proxy")

    def test_seaad_uses_micro_pvm_supertype_not_immune_subclass_as_microglia(self):
        obj=ad.AnnData(np.ones((3,1)),obs=pd.DataFrame({"cell_type":["Microglia"]*3,
            "source_subclass":["Immune"]*3,"subpopulation":["Micro-PVM_1","Lymphocyte","Monocyte"]},
            index=["m","l","o"]),var=pd.DataFrame(index=["RHBDF2"]))
        groups=contexts("SEAAD_MTG",obj)
        self.assertEqual(len(groups),1)
        np.testing.assert_array_equal(groups[0][-1],[True,False,False])


if __name__=="__main__":
    unittest.main()
