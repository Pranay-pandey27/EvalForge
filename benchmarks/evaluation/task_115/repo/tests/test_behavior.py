import numpy as np
from numerics.logsumexp import logsumexp
def test_large_values_remain_finite():
    assert np.isfinite(logsumexp([1000.0,1001.0]))
def test_typical_values_match_direct_formula():
    assert logsumexp([0.,1.]) == np.log(1 + np.e)
