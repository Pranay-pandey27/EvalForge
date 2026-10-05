from physics_units.temperature import kelvin_to_celsius
def test_evaluator_absolute_zero():
    assert kelvin_to_celsius(0.0) == -273.15
def test_evaluator_scalar_is_zero_dimensional():
    assert kelvin_to_celsius(300.0).shape == ()
