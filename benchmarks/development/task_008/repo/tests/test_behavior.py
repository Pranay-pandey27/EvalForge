import numpy as np
from scientific_units.length import meters_to_nanometers

def test_one_meter_is_one_billion_nanometers():
    assert meters_to_nanometers(1.0) == 1e9

def test_array_conversion():
    np.testing.assert_allclose(meters_to_nanometers([1e-9, 2e-9]), [1.0, 2.0])
