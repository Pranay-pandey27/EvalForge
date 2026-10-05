import json
import numpy as np
import pytest
from results_io.json_results import serialize_results
def test_evaluator_nested_numpy_scalars():
    assert json.loads(serialize_results({'x':[np.int64(3)]}))=={'x':[3]}
def test_evaluator_infinity_rejected():
    with pytest.raises(ValueError): serialize_results(np.array([np.inf]))
