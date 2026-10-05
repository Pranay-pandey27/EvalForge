import numpy as np
from scientific_units.length import meters_to_nanometers

def test_evaluator_zero_is_preserved():
    assert meters_to_nanometers(0.0) == 0.0

def test_evaluator_negative_displacement():
    assert meters_to_nanometers(-3e-9) == np.asarray(-3.0)
