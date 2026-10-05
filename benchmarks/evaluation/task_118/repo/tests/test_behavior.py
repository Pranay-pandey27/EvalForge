import numpy as np
from physics_units.temperature import kelvin_to_celsius
def test_freezing_point():
    assert kelvin_to_celsius(273.15) == 0.0
def test_array_conversion():
    np.testing.assert_allclose(kelvin_to_celsius([273.15,373.15]),[0,100])
