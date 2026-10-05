from reproducibility.bootstrap import bootstrap_mean

def test_evaluator_returns_float():
    assert isinstance(bootstrap_mean([1, 2, 3], 1), float)

def test_evaluator_seed_changes_sampling_path():
    values = list(range(100))
    assert bootstrap_mean(values, 1) != bootstrap_mean(values, 2)
