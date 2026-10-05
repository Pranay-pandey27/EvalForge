import json
import numpy as np
import pytest
from results_io.json_results import serialize_results
def test_numpy_values_are_serialized():
    encoded=serialize_results({'score':np.float64(.5),'labels':np.array([1,2])})
    assert json.loads(encoded)=={'labels':[1,2],'score':.5}
def test_nonfinite_values_are_rejected():
    with pytest.raises(ValueError): serialize_results({'loss':np.nan})
