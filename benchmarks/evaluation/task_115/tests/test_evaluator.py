import numpy as np
from numerics.logsumexp import logsumexp
def test_evaluator_translation_identity():
    np.testing.assert_allclose(logsumexp([10.,11.]), 10 + logsumexp([0.,1.]))
def test_evaluator_single_value():
    assert logsumexp([-500.]) == -500.0
