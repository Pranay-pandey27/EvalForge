"""Generate EvalForge's compact, reproducible benchmark task packages.

The source blueprints live here so every buggy state, reference repair, visible
test, evaluator test, and scripted infrastructure trace can be audited together.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


@dataclass(frozen=True)
class Blueprint:
    suite: str
    number: int
    task_id: str
    title: str
    repository: str
    category: str
    difficulty: str
    package: str
    module: str
    issue: str
    buggy: str
    fixed: str
    visible_tests: str
    evaluator_tests: str
    failing_test: str
    expected_failure: str
    tags: tuple[str, ...]
    dependencies: tuple[str, ...] = ("numpy>=1.26", "pytest>=8.2")

    @property
    def directory_name(self) -> str:
        return f"task_{self.number:03d}"

    @property
    def source_path(self) -> str:
        return f"{self.package}/{self.module}.py"


def clean(value: str) -> str:
    return dedent(value).strip() + "\n"


BLUEPRINTS: list[Blueprint] = [
    Blueprint(
        suite="development",
        number=2,
        task_id="regression_rmse_002",
        title="Root mean squared error implementation",
        repository="ml-metrics-mini",
        category="regression_metrics",
        difficulty="easy",
        package="ml_metrics",
        module="regression",
        issue="`rmse` is reporting mean squared error rather than root mean squared error. Repair the metric, preserve NumPy input support, and return a Python float.",
        buggy="""
        import numpy as np

        def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            errors = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
            return float(np.mean(errors ** 2))
        """,
        fixed="""
        import numpy as np

        def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            errors = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
            return float(np.sqrt(np.mean(errors ** 2)))
        """,
        visible_tests="""
        import numpy as np
        from ml_metrics.regression import rmse

        def test_rmse_takes_square_root():
            assert rmse(np.array([0, 0]), np.array([3, 4])) == pytest.approx(np.sqrt(12.5))

        def test_rmse_is_zero_for_exact_predictions():
            assert rmse(np.array([1, 2]), np.array([1, 2])) == 0.0

        import pytest
        """,
        evaluator_tests="""
        import numpy as np
        from ml_metrics.regression import rmse

        def test_evaluator_returns_python_float():
            assert isinstance(rmse(np.array([1]), np.array([2])), float)

        def test_evaluator_handles_fractional_errors():
            assert rmse(np.array([0.0, 1.0]), np.array([0.5, 1.5])) == 0.5
        """,
        failing_test="tests/test_behavior.py::test_rmse_takes_square_root",
        expected_failure="RMSE omits the outer square root and reports MSE",
        tags=("numpy", "metrics", "regression"),
    ),
    Blueprint(
        suite="development",
        number=3,
        task_id="numerical_log_003",
        title="Stable binary cross entropy",
        repository="stable-losses-mini",
        category="numerical_stability",
        difficulty="easy",
        package="stable_losses",
        module="binary",
        issue="Binary cross entropy returns infinities for probabilities at 0 or 1. Make it finite without mutating inputs.",
        buggy="""
        import numpy as np

        def binary_cross_entropy(labels: np.ndarray, probabilities: np.ndarray) -> float:
            y = np.asarray(labels, dtype=float)
            p = np.asarray(probabilities, dtype=float)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        """,
        fixed="""
        import numpy as np

        def binary_cross_entropy(labels: np.ndarray, probabilities: np.ndarray) -> float:
            y = np.asarray(labels, dtype=float)
            p = np.asarray(probabilities, dtype=float)
            safe = np.clip(p, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
            return float(-np.mean(y * np.log(safe) + (1 - y) * np.log(1 - safe)))
        """,
        visible_tests="""
        import numpy as np
        from stable_losses.binary import binary_cross_entropy

        def test_boundary_probabilities_are_finite():
            value = binary_cross_entropy(np.array([0, 1]), np.array([0.0, 1.0]))
            assert np.isfinite(value)

        def test_typical_probabilities_match_formula():
            assert binary_cross_entropy(np.array([1]), np.array([0.5])) == np.log(2)
        """,
        evaluator_tests="""
        import numpy as np
        from stable_losses.binary import binary_cross_entropy

        def test_evaluator_does_not_mutate_probabilities():
            p = np.array([0.0, 0.8])
            before = p.copy()
            binary_cross_entropy(np.array([0, 1]), p)
            np.testing.assert_array_equal(p, before)

        def test_evaluator_near_boundary_is_nonnegative():
            assert binary_cross_entropy(np.array([1]), np.array([1.0])) >= 0.0
        """,
        failing_test="tests/test_behavior.py::test_boundary_probabilities_are_finite",
        expected_failure="logarithms at probability boundaries produce NaN or infinity",
        tags=("numpy", "loss", "log0", "numerical-stability"),
    ),
    Blueprint(
        suite="development",
        number=4,
        task_id="pandas_leakage_004",
        title="Train-only centering statistics",
        repository="tabular-preprocessing-mini",
        category="pandas_data_leakage",
        difficulty="medium",
        package="tabular_preprocessing",
        module="centering",
        issue="The centering transformer learns its mean from both train and test rows. Fit statistics on training data only and apply them to both splits.",
        buggy="""
        import pandas as pd

        def center_splits(train: pd.DataFrame, test: pd.DataFrame):
            combined = pd.concat([train, test], ignore_index=True)
            mean = combined.mean()
            return train - mean, test - mean, mean
        """,
        fixed="""
        import pandas as pd

        def center_splits(train: pd.DataFrame, test: pd.DataFrame):
            mean = train.mean()
            return train - mean, test - mean, mean
        """,
        visible_tests="""
        import pandas as pd
        from tabular_preprocessing.centering import center_splits

        def test_statistics_are_fit_on_train_only():
            train = pd.DataFrame({'x': [0.0, 2.0]})
            test = pd.DataFrame({'x': [100.0]})
            centered_train, _, mean = center_splits(train, test)
            assert mean['x'] == 1.0
            assert centered_train['x'].mean() == 0.0

        def test_output_columns_are_preserved():
            train = pd.DataFrame({'a': [1], 'b': [2]})
            test = pd.DataFrame({'a': [3], 'b': [4]})
            assert list(center_splits(train, test)[0].columns) == ['a', 'b']
        """,
        evaluator_tests="""
        import pandas as pd
        from tabular_preprocessing.centering import center_splits

        def test_evaluator_test_values_do_not_change_fitted_mean():
            train = pd.DataFrame({'x': [2.0, 4.0]})
            _, _, mean_a = center_splits(train, pd.DataFrame({'x': [-999.0]}))
            _, _, mean_b = center_splits(train, pd.DataFrame({'x': [999.0]}))
            pd.testing.assert_series_equal(mean_a, mean_b)

        def test_evaluator_applies_train_mean_to_test():
            train = pd.DataFrame({'x': [1.0, 3.0]})
            test = pd.DataFrame({'x': [5.0]})
            assert center_splits(train, test)[1].iloc[0, 0] == 3.0
        """,
        failing_test="tests/test_behavior.py::test_statistics_are_fit_on_train_only",
        expected_failure="preprocessing statistics include held-out test rows",
        tags=("pandas", "data-leakage", "preprocessing", "split"),
        dependencies=("pandas>=2.2", "pytest>=8.2"),
    ),
    Blueprint(
        suite="development",
        number=5,
        task_id="pytorch_zero_grad_005",
        title="Clear gradients before a training batch",
        repository="torch-loop-mini",
        category="pytorch_training_loop",
        difficulty="easy",
        package="torch_loop",
        module="training",
        issue="`train_batch` accumulates stale gradients because the optimizer is never cleared. Follow standard PyTorch batch ordering.",
        buggy="""
        def train_batch(model, inputs, targets, optimizer, loss_fn):
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            return loss
        """,
        fixed="""
        def train_batch(model, inputs, targets, optimizer, loss_fn):
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            return loss
        """,
        visible_tests="""
        from torch_loop.training import train_batch

        class Recorder:
            def __init__(self, events): self.events = events
            def zero_grad(self): self.events.append('zero_grad')
            def step(self): self.events.append('step')

        class Loss:
            def __init__(self, events): self.events = events
            def backward(self): self.events.append('backward')

        def test_zero_grad_precedes_backward_and_step():
            events = []
            model = lambda x: x
            loss = Loss(events)
            train_batch(model, 1, 1, Recorder(events), lambda p, t: loss)
            assert events == ['zero_grad', 'backward', 'step']

        def test_returns_loss_object():
            events, loss = [], Loss([])
            assert train_batch(lambda x: x, 1, 1, Recorder(events), lambda p, t: loss) is loss
        """,
        evaluator_tests="""
        from torch_loop.training import train_batch

        class Optimizer:
            def __init__(self): self.zero_calls = 0; self.steps = 0
            def zero_grad(self): self.zero_calls += 1
            def step(self): self.steps += 1
        class Loss:
            def backward(self): pass

        def test_evaluator_clears_once_per_batch():
            optimizer = Optimizer()
            train_batch(lambda x: x, 0, 0, optimizer, lambda p, t: Loss())
            assert (optimizer.zero_calls, optimizer.steps) == (1, 1)

        def test_evaluator_model_receives_input():
            seen = []
            train_batch(lambda x: seen.append(x) or x, 7, 0, Optimizer(), lambda p, t: Loss())
            assert seen == [7]
        """,
        failing_test="tests/test_behavior.py::test_zero_grad_precedes_backward_and_step",
        expected_failure="optimizer gradients are not cleared before backward",
        tags=("pytorch", "gradient", "optimizer", "training-loop"),
        dependencies=("pytest>=8.2",),
    ),
    Blueprint(
        suite="development",
        number=6,
        task_id="device_batch_006",
        title="Move all batch tensors to the target device",
        repository="tensor-device-mini",
        category="device_dtype_mismatch",
        difficulty="easy",
        package="tensor_device",
        module="batch",
        issue="Only inputs are moved to the configured device. Move both inputs and targets and preserve their order.",
        buggy="""
        def move_batch(inputs, targets, device):
            return inputs.to(device), targets
        """,
        fixed="""
        def move_batch(inputs, targets, device):
            return inputs.to(device), targets.to(device)
        """,
        visible_tests="""
        from tensor_device.batch import move_batch

        class Tensor:
            def __init__(self, name): self.name = name; self.devices = []
            def to(self, device): self.devices.append(device); return self

        def test_moves_inputs_and_targets():
            x, y = Tensor('x'), Tensor('y')
            move_batch(x, y, 'cuda')
            assert x.devices == ['cuda'] and y.devices == ['cuda']

        def test_preserves_tuple_order():
            x, y = Tensor('x'), Tensor('y')
            assert move_batch(x, y, 'cpu') == (x, y)
        """,
        evaluator_tests="""
        from tensor_device.batch import move_batch

        class Tensor:
            def __init__(self): self.calls = 0
            def to(self, device): self.calls += 1; self.device = device; return self

        def test_evaluator_moves_each_tensor_exactly_once():
            x, y = Tensor(), Tensor()
            move_batch(x, y, 'mps')
            assert (x.calls, y.calls) == (1, 1)

        def test_evaluator_uses_requested_device():
            x, y = Tensor(), Tensor()
            move_batch(x, y, 'accelerator:1')
            assert x.device == y.device == 'accelerator:1'
        """,
        failing_test="tests/test_behavior.py::test_moves_inputs_and_targets",
        expected_failure="target tensor remains on a different device from inputs",
        tags=("pytorch", "device", "tensor", "batch"),
        dependencies=("pytest>=8.2",),
    ),
    Blueprint(
        suite="development",
        number=7,
        task_id="seed_isolation_007",
        title="Local reproducible bootstrap RNG",
        repository="reproducibility-mini",
        category="random_seed_handling",
        difficulty="medium",
        package="reproducibility",
        module="bootstrap",
        issue="Bootstrap sampling resets NumPy's global RNG. Make sampling reproducible from the supplied seed without perturbing global random state.",
        buggy="""
        import numpy as np

        def bootstrap_mean(values, seed):
            np.random.seed(seed)
            data = np.asarray(values)
            sample = np.random.choice(data, size=len(data), replace=True)
            return float(sample.mean())
        """,
        fixed="""
        import numpy as np

        def bootstrap_mean(values, seed):
            rng = np.random.default_rng(seed)
            data = np.asarray(values)
            sample = rng.choice(data, size=len(data), replace=True)
            return float(sample.mean())
        """,
        visible_tests="""
        import numpy as np
        from reproducibility.bootstrap import bootstrap_mean

        def test_does_not_reset_global_rng():
            np.random.seed(123)
            expected = np.random.random(2)
            np.random.seed(123)
            first = np.random.random()
            bootstrap_mean([1, 2, 3], 7)
            second = np.random.random()
            np.testing.assert_allclose([first, second], expected)

        def test_same_seed_is_reproducible():
            assert bootstrap_mean([1, 2, 9], 4) == bootstrap_mean([1, 2, 9], 4)
        """,
        evaluator_tests="""
        from reproducibility.bootstrap import bootstrap_mean

        def test_evaluator_returns_float():
            assert isinstance(bootstrap_mean([1, 2, 3], 1), float)

        def test_evaluator_seed_changes_sampling_path():
            values = list(range(100))
            assert bootstrap_mean(values, 1) != bootstrap_mean(values, 2)
        """,
        failing_test="tests/test_behavior.py::test_does_not_reset_global_rng",
        expected_failure="function mutates NumPy global random state",
        tags=("numpy", "random", "seed", "reproducibility"),
    ),
    Blueprint(
        suite="development",
        number=8,
        task_id="units_nanometer_008",
        title="Meters to nanometers conversion",
        repository="scientific-units-mini",
        category="scientific_unit_conversion",
        difficulty="easy",
        package="scientific_units",
        module="length",
        issue="The meters-to-nanometers conversion is off by three orders of magnitude. Correct the SI factor for scalars and arrays.",
        buggy="""
        import numpy as np

        def meters_to_nanometers(value):
            return np.asarray(value) * 1e6
        """,
        fixed="""
        import numpy as np

        def meters_to_nanometers(value):
            return np.asarray(value) * 1e9
        """,
        visible_tests="""
        import numpy as np
        from scientific_units.length import meters_to_nanometers

        def test_one_meter_is_one_billion_nanometers():
            assert meters_to_nanometers(1.0) == 1e9

        def test_array_conversion():
            np.testing.assert_allclose(meters_to_nanometers([1e-9, 2e-9]), [1.0, 2.0])
        """,
        evaluator_tests="""
        import numpy as np
        from scientific_units.length import meters_to_nanometers

        def test_evaluator_zero_is_preserved():
            assert meters_to_nanometers(0.0) == 0.0

        def test_evaluator_negative_displacement():
            assert meters_to_nanometers(-3e-9) == np.asarray(-3.0)
        """,
        failing_test="tests/test_behavior.py::test_one_meter_is_one_billion_nanometers",
        expected_failure="SI conversion uses micrometer factor instead of nanometer factor",
        tags=("physics", "units", "si", "conversion"),
    ),
    Blueprint(
        suite="evaluation", number=101, task_id="numpy_axis_101",
        title="Feature-wise L2 norms", repository="array-ops-heldout",
        category="numpy_shape_broadcasting", difficulty="easy",
        package="array_ops", module="norms",
        issue="Compute one L2 norm per feature column for a two-dimensional sample matrix.",
        buggy="""
        import numpy as np
        def feature_l2_norms(matrix):
            values = np.asarray(matrix, dtype=float)
            return np.sqrt(np.sum(values ** 2, axis=1))
        """,
        fixed="""
        import numpy as np
        def feature_l2_norms(matrix):
            values = np.asarray(matrix, dtype=float)
            return np.sqrt(np.sum(values ** 2, axis=0))
        """,
        visible_tests="""
        import numpy as np
        from array_ops.norms import feature_l2_norms
        def test_returns_one_norm_per_feature():
            result = feature_l2_norms([[3, 0], [4, 12]])
            np.testing.assert_allclose(result, [5, 12])
        def test_output_shape_tracks_columns():
            assert feature_l2_norms(np.ones((2, 3))).shape == (3,)
        """,
        evaluator_tests="""
        import numpy as np
        from array_ops.norms import feature_l2_norms
        def test_evaluator_single_row():
            np.testing.assert_allclose(feature_l2_norms([[-3, 4]]), [3, 4])
        def test_evaluator_zero_column():
            np.testing.assert_allclose(feature_l2_norms([[0, 1], [0, 2]]), [0, np.sqrt(5)])
        """,
        failing_test="tests/test_behavior.py::test_returns_one_norm_per_feature",
        expected_failure="reduction uses the sample axis instead of the feature axis",
        tags=("numpy", "axis", "shape", "norm"),
    ),
    Blueprint(
        suite="evaluation", number=102, task_id="numpy_softmax_102",
        title="Row-wise stable softmax", repository="array-ops-heldout",
        category="numpy_shape_broadcasting", difficulty="medium",
        package="array_ops", module="softmax",
        issue="Softmax must normalize each row independently and remain stable for large logits.",
        buggy="""
        import numpy as np
        def softmax_rows(logits):
            values = np.asarray(logits, dtype=float)
            shifted = values - values.max()
            exponentials = np.exp(shifted)
            return exponentials / exponentials.sum()
        """,
        fixed="""
        import numpy as np
        def softmax_rows(logits):
            values = np.asarray(logits, dtype=float)
            shifted = values - values.max(axis=1, keepdims=True)
            exponentials = np.exp(shifted)
            return exponentials / exponentials.sum(axis=1, keepdims=True)
        """,
        visible_tests="""
        import numpy as np
        from array_ops.softmax import softmax_rows
        def test_each_row_sums_to_one():
            result = softmax_rows([[1, 2], [3, 4]])
            np.testing.assert_allclose(result.sum(axis=1), [1, 1])
        def test_large_logits_are_finite():
            assert np.isfinite(softmax_rows([[1000, 1001]])).all()
        """,
        evaluator_tests="""
        import numpy as np
        from array_ops.softmax import softmax_rows
        def test_evaluator_translation_invariance_per_row():
            base = np.array([[1., 2.], [4., 8.]])
            shifts = np.array([[100.], [-50.]])
            np.testing.assert_allclose(softmax_rows(base), softmax_rows(base + shifts))
        def test_evaluator_shape_preserved():
            assert softmax_rows(np.zeros((4, 3))).shape == (4, 3)
        """,
        failing_test="tests/test_behavior.py::test_each_row_sums_to_one",
        expected_failure="softmax uses global rather than row-wise normalization",
        tags=("numpy", "broadcasting", "softmax", "axis"),
    ),
    Blueprint(
        suite="evaluation", number=103, task_id="tensor_dims_103",
        title="Spatial channel means", repository="tensor-shapes-heldout",
        category="bad_tensor_dimensions", difficulty="easy",
        package="tensor_shapes", module="images",
        issue="For NCHW image batches, return one spatial mean per batch item and channel with shape (N, C).",
        buggy="""
        import numpy as np
        def spatial_channel_mean(images):
            values = np.asarray(images, dtype=float)
            return values.mean(axis=(1, 2))
        """,
        fixed="""
        import numpy as np
        def spatial_channel_mean(images):
            values = np.asarray(images, dtype=float)
            return values.mean(axis=(2, 3))
        """,
        visible_tests="""
        import numpy as np
        from tensor_shapes.images import spatial_channel_mean
        def test_output_is_batch_by_channel():
            assert spatial_channel_mean(np.ones((2, 3, 4, 5))).shape == (2, 3)
        def test_channel_values_are_independent():
            image = np.stack([np.ones((2, 2)), np.full((2, 2), 7)])[None, ...]
            np.testing.assert_allclose(spatial_channel_mean(image), [[1, 7]])
        """,
        evaluator_tests="""
        import numpy as np
        from tensor_shapes.images import spatial_channel_mean
        def test_evaluator_non_square_spatial_axes():
            data = np.arange(24).reshape(1, 2, 3, 4)
            np.testing.assert_allclose(spatial_channel_mean(data), data.mean(axis=(2, 3)))
        def test_evaluator_dtype_is_floating():
            assert spatial_channel_mean(np.ones((1, 1, 1, 1), dtype=int)).dtype.kind == 'f'
        """,
        failing_test="tests/test_behavior.py::test_output_is_batch_by_channel",
        expected_failure="mean reduces the channel dimension and leaves a spatial dimension",
        tags=("tensor", "nchw", "dimensions", "numpy"),
    ),
    Blueprint(
        suite="evaluation", number=104, task_id="pandas_category_leak_104",
        title="Train-fitted one-hot categories", repository="tabular-heldout",
        category="pandas_data_leakage", difficulty="hard",
        package="tabular", module="encoding",
        issue="Fit one-hot categories on training rows only. Test-only categories must not create new feature columns.",
        buggy="""
        import pandas as pd
        def encode_splits(train, test, column):
            combined = pd.concat([train[[column]], test[[column]]], ignore_index=True)
            encoded = pd.get_dummies(combined[column], prefix=column, dtype=int)
            return encoded.iloc[:len(train)].reset_index(drop=True), encoded.iloc[len(train):].reset_index(drop=True)
        """,
        fixed="""
        import pandas as pd
        def encode_splits(train, test, column):
            categories = list(pd.Index(train[column].dropna().unique()))
            train_values = pd.Categorical(train[column], categories=categories)
            test_values = pd.Categorical(test[column], categories=categories)
            columns = [f"{column}_{value}" for value in categories]
            train_encoded = pd.get_dummies(train_values, prefix=column, dtype=int).reindex(columns=columns, fill_value=0)
            test_encoded = pd.get_dummies(test_values, prefix=column, dtype=int).reindex(columns=columns, fill_value=0)
            return train_encoded.reset_index(drop=True), test_encoded.reset_index(drop=True)
        """,
        visible_tests="""
        import pandas as pd
        from tabular.encoding import encode_splits
        def test_test_only_category_does_not_create_column():
            train = pd.DataFrame({'color': ['red', 'blue']})
            test = pd.DataFrame({'color': ['green']})
            train_x, test_x = encode_splits(train, test, 'color')
            assert set(train_x.columns) == {'color_red', 'color_blue'}
            assert list(test_x.iloc[0]) == [0, 0]
        def test_split_row_counts_are_preserved():
            a, b = encode_splits(pd.DataFrame({'x':['a','b']}), pd.DataFrame({'x':['a']}), 'x')
            assert (len(a), len(b)) == (2, 1)
        """,
        evaluator_tests="""
        import pandas as pd
        from tabular.encoding import encode_splits
        def test_evaluator_column_order_follows_training_observation():
            train_x, _ = encode_splits(pd.DataFrame({'x':['z','a']}), pd.DataFrame({'x':['a']}), 'x')
            assert list(train_x.columns) == ['x_z', 'x_a']
        def test_evaluator_known_test_category_is_encoded():
            _, test_x = encode_splits(pd.DataFrame({'x':['a','b']}), pd.DataFrame({'x':['b']}), 'x')
            assert test_x['x_b'].iloc[0] == 1
        """,
        failing_test="tests/test_behavior.py::test_test_only_category_does_not_create_column",
        expected_failure="test-only categories leak into the fitted feature space",
        tags=("pandas", "one-hot", "leakage", "categories"),
        dependencies=("pandas>=2.2", "pytest>=8.2"),
    ),
    Blueprint(
        suite="evaluation", number=105, task_id="target_leak_105",
        title="Exclude target from model features", repository="tabular-heldout",
        category="data_split_mistakes", difficulty="easy",
        package="tabular", module="features",
        issue="Return feature columns without the named target column, while leaving the input frame unchanged.",
        buggy="""
        def split_features_target(frame, target):
            return frame.copy(), frame[target].copy()
        """,
        fixed="""
        def split_features_target(frame, target):
            return frame.drop(columns=[target]).copy(), frame[target].copy()
        """,
        visible_tests="""
        import pandas as pd
        from tabular.features import split_features_target
        def test_target_is_not_in_features():
            x, y = split_features_target(pd.DataFrame({'a':[1], 'label':[0]}), 'label')
            assert list(x.columns) == ['a'] and y.tolist() == [0]
        def test_input_is_not_mutated():
            frame = pd.DataFrame({'a':[1], 'y':[2]}); before = frame.copy()
            split_features_target(frame, 'y'); pd.testing.assert_frame_equal(frame, before)
        """,
        evaluator_tests="""
        import pandas as pd
        from tabular.features import split_features_target
        def test_evaluator_target_can_be_first_column():
            x, _ = split_features_target(pd.DataFrame({'y':[1], 'x':[2]}), 'y')
            assert list(x.columns) == ['x']
        def test_evaluator_preserves_feature_order():
            x, _ = split_features_target(pd.DataFrame({'b':[1], 'y':[0], 'a':[2]}), 'y')
            assert list(x.columns) == ['b', 'a']
        """,
        failing_test="tests/test_behavior.py::test_target_is_not_in_features",
        expected_failure="target values are included directly in the feature matrix",
        tags=("pandas", "target-leakage", "features", "split"),
        dependencies=("pandas>=2.2", "pytest>=8.2"),
    ),
    Blueprint(
        suite="evaluation", number=106, task_id="classification_logits_106",
        title="Binary accuracy from logits", repository="metrics-heldout",
        category="classification_metrics", difficulty="easy",
        package="heldout_metrics", module="classification",
        issue="`binary_accuracy_from_logits` receives raw logits, so use the correct decision boundary and validate matching shapes.",
        buggy="""
        import numpy as np
        def binary_accuracy_from_logits(logits, labels):
            predictions = np.asarray(logits) > 0.5
            return float(np.mean(predictions == np.asarray(labels)))
        """,
        fixed="""
        import numpy as np
        def binary_accuracy_from_logits(logits, labels):
            scores = np.asarray(logits)
            targets = np.asarray(labels)
            if scores.shape != targets.shape:
                raise ValueError("logits and labels must have matching shapes")
            predictions = scores > 0.0
            return float(np.mean(predictions == targets))
        """,
        visible_tests="""
        import numpy as np
        import pytest
        from heldout_metrics.classification import binary_accuracy_from_logits
        def test_zero_is_logit_decision_boundary():
            assert binary_accuracy_from_logits(np.array([0.2, -0.2]), np.array([1, 0])) == 1.0
        def test_shape_mismatch_is_rejected():
            with pytest.raises(ValueError): binary_accuracy_from_logits(np.array([1,2]), np.array([1]))
        """,
        evaluator_tests="""
        import numpy as np
        from heldout_metrics.classification import binary_accuracy_from_logits
        def test_evaluator_negative_logits_predict_zero():
            assert binary_accuracy_from_logits(np.array([-9., -0.01]), np.array([0,0])) == 1.0
        def test_evaluator_returns_float():
            assert isinstance(binary_accuracy_from_logits(np.array([1]), np.array([1])), float)
        """,
        failing_test="tests/test_behavior.py::test_zero_is_logit_decision_boundary",
        expected_failure="raw logits are thresholded at 0.5 instead of zero",
        tags=("classification", "logits", "accuracy", "metrics"),
    ),
    Blueprint(
        suite="evaluation", number=107, task_id="regression_mae_107",
        title="Mean absolute error", repository="metrics-heldout",
        category="regression_metrics", difficulty="easy",
        package="heldout_metrics", module="regression",
        issue="The `mae` function calculates a root-mean-square quantity. Implement mean absolute error and reject shape mismatches.",
        buggy="""
        import numpy as np
        def mae(y_true, y_pred):
            errors = np.asarray(y_true) - np.asarray(y_pred)
            return float(np.sqrt(np.mean(errors ** 2)))
        """,
        fixed="""
        import numpy as np
        def mae(y_true, y_pred):
            truth = np.asarray(y_true)
            prediction = np.asarray(y_pred)
            if truth.shape != prediction.shape:
                raise ValueError("inputs must have matching shapes")
            return float(np.mean(np.abs(truth - prediction)))
        """,
        visible_tests="""
        import pytest
        from heldout_metrics.regression import mae
        def test_mae_uses_absolute_errors():
            assert mae([0, 0], [3, 4]) == 3.5
        def test_shape_mismatch_rejected():
            with pytest.raises(ValueError): mae([1,2], [1])
        """,
        evaluator_tests="""
        from heldout_metrics.regression import mae
        def test_evaluator_signed_errors_do_not_cancel():
            assert mae([0, 2], [1, 1]) == 1.0
        def test_evaluator_exact_prediction():
            assert mae([1.5], [1.5]) == 0.0
        """,
        failing_test="tests/test_behavior.py::test_mae_uses_absolute_errors",
        expected_failure="function computes RMSE instead of mean absolute error",
        tags=("regression", "mae", "metrics", "numpy"),
    ),
    Blueprint(
        suite="evaluation", number=108, task_id="macro_f1_108",
        title="Macro-averaged multiclass F1", repository="metrics-heldout",
        category="classification_metrics", difficulty="hard",
        package="heldout_metrics", module="macro_f1",
        issue="Compute unweighted macro F1 across every class present in truth or predictions, with zero for undefined class F1.",
        buggy="""
        import numpy as np
        def macro_f1(y_true, y_pred):
            truth, pred = np.asarray(y_true), np.asarray(y_pred)
            return float(np.mean(truth == pred))
        """,
        fixed="""
        import numpy as np
        def macro_f1(y_true, y_pred):
            truth, pred = np.asarray(y_true), np.asarray(y_pred)
            classes = np.union1d(truth, pred)
            scores = []
            for label in classes:
                tp = np.sum((truth == label) & (pred == label))
                fp = np.sum((truth != label) & (pred == label))
                fn = np.sum((truth == label) & (pred != label))
                denominator = 2 * tp + fp + fn
                scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
            return float(np.mean(scores)) if scores else 0.0
        """,
        visible_tests="""
        import pytest
        from heldout_metrics.macro_f1 import macro_f1
        def test_macro_f1_differs_from_accuracy_for_imbalance():
            assert macro_f1([0,0,0,1], [0,0,0,0]) == pytest.approx(3/7)
        def test_perfect_predictions_score_one():
            assert macro_f1([0,1,2], [0,1,2]) == 1.0
        """,
        evaluator_tests="""
        import pytest
        from heldout_metrics.macro_f1 import macro_f1
        def test_evaluator_includes_predicted_only_class():
            assert macro_f1([0,0], [0,2]) == pytest.approx((2/3 + 0)/2)
        def test_evaluator_empty_inputs_return_zero():
            assert macro_f1([], []) == 0.0
        """,
        failing_test="tests/test_behavior.py::test_macro_f1_differs_from_accuracy_for_imbalance",
        expected_failure="implementation reports accuracy instead of per-class macro F1",
        tags=("classification", "f1", "macro", "imbalance"),
    ),
    Blueprint(
        suite="evaluation", number=109, task_id="pytorch_train_mode_109",
        title="Activate training mode for an epoch", repository="torch-loops-heldout",
        category="pytorch_training_loop", difficulty="easy",
        package="torch_loops", module="train_mode",
        issue="Set the model to training mode exactly once before processing the epoch's batches.",
        buggy="""
        def train_epoch(model, batches, update_batch):
            losses = []
            for batch in batches:
                losses.append(update_batch(model, batch))
            return losses
        """,
        fixed="""
        def train_epoch(model, batches, update_batch):
            model.train()
            losses = []
            for batch in batches:
                losses.append(update_batch(model, batch))
            return losses
        """,
        visible_tests="""
        from torch_loops.train_mode import train_epoch
        class Model:
            def __init__(self): self.train_calls = 0
            def train(self): self.train_calls += 1
        def test_model_enters_train_mode_once():
            model = Model(); train_epoch(model, [1,2], lambda m,b: b)
            assert model.train_calls == 1
        def test_returns_one_loss_per_batch():
            assert train_epoch(Model(), [1,2], lambda m,b: b * 2) == [2,4]
        """,
        evaluator_tests="""
        from torch_loops.train_mode import train_epoch
        class Model:
            def __init__(self): self.events=[]
            def train(self): self.events.append('train')
        def test_evaluator_mode_set_before_first_update():
            model=Model(); train_epoch(model,[1],lambda m,b: m.events.append('update') or 0)
            assert model.events == ['train','update']
        def test_evaluator_empty_epoch_still_sets_mode():
            model=Model(); assert train_epoch(model,[],lambda m,b: 0)==[] and model.events==['train']
        """,
        failing_test="tests/test_behavior.py::test_model_enters_train_mode_once",
        expected_failure="training loop never calls model.train()",
        tags=("pytorch", "train-mode", "dropout", "batchnorm"),
        dependencies=("pytest>=8.2",),
    ),
    Blueprint(
        suite="evaluation", number=110, task_id="pytorch_eval_mode_110",
        title="Evaluation mode and no-grad context", repository="torch-loops-heldout",
        category="pytorch_training_loop", difficulty="medium",
        package="torch_loops", module="evaluation",
        issue="Evaluation must call `model.eval()` and process every batch inside the supplied `no_grad` context manager.",
        buggy="""
        def evaluate(model, batches, score_batch, no_grad):
            model.train()
            return [score_batch(model, batch) for batch in batches]
        """,
        fixed="""
        def evaluate(model, batches, score_batch, no_grad):
            model.eval()
            with no_grad():
                return [score_batch(model, batch) for batch in batches]
        """,
        visible_tests="""
        from contextlib import contextmanager
        from torch_loops.evaluation import evaluate
        class Model:
            def __init__(self): self.mode=None
            def train(self): self.mode='train'
            def eval(self): self.mode='eval'
        def test_uses_eval_mode_and_no_grad():
            events=[]
            @contextmanager
            def no_grad(): events.append('enter'); yield; events.append('exit')
            model=Model(); evaluate(model,[1],lambda m,b: events.append('score') or b,no_grad)
            assert model.mode=='eval' and events==['enter','score','exit']
        def test_returns_scores():
            @contextmanager
            def no_grad(): yield
            assert evaluate(Model(),[2,3],lambda m,b:b+1,no_grad)==[3,4]
        """,
        evaluator_tests="""
        from contextlib import contextmanager
        from torch_loops.evaluation import evaluate
        class Model:
            def __init__(self): self.events=[]
            def train(self): self.events.append('train')
            def eval(self): self.events.append('eval')
        def test_evaluator_eval_precedes_context():
            model=Model()
            @contextmanager
            def no_grad(): model.events.append('enter'); yield
            evaluate(model,[],lambda m,b:0,no_grad)
            assert model.events==['eval','enter']
        def test_evaluator_context_exits_on_error():
            events=[]
            @contextmanager
            def no_grad():
                events.append('enter')
                try: yield
                finally: events.append('exit')
            try: evaluate(Model(),[1],lambda m,b: (_ for _ in ()).throw(RuntimeError()),no_grad)
            except RuntimeError: pass
            assert events==['enter','exit']
        """,
        failing_test="tests/test_behavior.py::test_uses_eval_mode_and_no_grad",
        expected_failure="evaluation uses training mode and tracks gradients",
        tags=("pytorch", "eval-mode", "no-grad", "inference"),
        dependencies=("pytest>=8.2",),
    ),
    Blueprint(
        suite="evaluation", number=111, task_id="gradient_order_111",
        title="Optimizer operation order", repository="torch-loops-heldout",
        category="incorrect_gradient_handling", difficulty="medium",
        package="torch_loops", module="gradient_order",
        issue="Clear gradients before forward/backward, then step after backward. Preserve the returned loss.",
        buggy="""
        def optimize_batch(model, batch, optimizer, loss_fn):
            loss = loss_fn(model(batch))
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            return loss
        """,
        fixed="""
        def optimize_batch(model, batch, optimizer, loss_fn):
            optimizer.zero_grad()
            loss = loss_fn(model(batch))
            loss.backward()
            optimizer.step()
            return loss
        """,
        visible_tests="""
        from torch_loops.gradient_order import optimize_batch
        class Optimizer:
            def __init__(self,e): self.e=e
            def zero_grad(self): self.e.append('zero')
            def step(self): self.e.append('step')
        class Loss:
            def __init__(self,e): self.e=e
            def backward(self): self.e.append('backward')
        def test_gradient_operation_order():
            e=[]; optimize_batch(lambda x:e.append('forward') or x,1,Optimizer(e),lambda x:Loss(e))
            assert e==['zero','forward','backward','step']
        def test_returns_loss():
            e=[]; loss=Loss(e)
            assert optimize_batch(lambda x:x,1,Optimizer(e),lambda x:loss) is loss
        """,
        evaluator_tests="""
        from torch_loops.gradient_order import optimize_batch
        class O:
            def __init__(self): self.zero=0; self.steps=0
            def zero_grad(self): self.zero+=1
            def step(self): self.steps+=1
        class L:
            def backward(self): pass
        def test_evaluator_one_clear_and_step():
            o=O(); optimize_batch(lambda x:x,0,o,lambda x:L()); assert (o.zero,o.steps)==(1,1)
        def test_evaluator_batch_reaches_model():
            seen=[]; optimize_batch(lambda x:seen.append(x) or x,9,O(),lambda x:L()); assert seen==[9]
        """,
        failing_test="tests/test_behavior.py::test_gradient_operation_order",
        expected_failure="zero_grad is called after backward and erases current gradients",
        tags=("pytorch", "gradient", "zero-grad", "ordering"),
        dependencies=("pytest>=8.2",),
    ),
    Blueprint(
        suite="evaluation", number=112, task_id="gradient_accumulation_112",
        title="Gradient accumulation windows", repository="torch-loops-heldout",
        category="incorrect_gradient_handling", difficulty="hard",
        package="torch_loops", module="accumulation",
        issue="Accumulate gradients for `accumulation_steps` batches: clear at each window start and step at each full window or final partial window.",
        buggy="""
        def train_accumulated(batches, optimizer, backward_batch, accumulation_steps):
            for batch in batches:
                optimizer.zero_grad()
                backward_batch(batch)
                optimizer.step()
        """,
        fixed="""
        def train_accumulated(batches, optimizer, backward_batch, accumulation_steps):
            if accumulation_steps < 1:
                raise ValueError("accumulation_steps must be positive")
            total = len(batches)
            for index, batch in enumerate(batches):
                if index % accumulation_steps == 0:
                    optimizer.zero_grad()
                backward_batch(batch)
                if (index + 1) % accumulation_steps == 0 or index + 1 == total:
                    optimizer.step()
        """,
        visible_tests="""
        import pytest
        from torch_loops.accumulation import train_accumulated
        class O:
            def __init__(self,e): self.e=e
            def zero_grad(self): self.e.append('zero')
            def step(self): self.e.append('step')
        def test_two_batch_windows():
            e=[]; train_accumulated([1,2,3],O(e),lambda b:e.append(f'b{b}'),2)
            assert e==['zero','b1','b2','step','zero','b3','step']
        def test_rejects_nonpositive_window():
            with pytest.raises(ValueError): train_accumulated([],O([]),lambda b:None,0)
        """,
        evaluator_tests="""
        from torch_loops.accumulation import train_accumulated
        class O:
            def __init__(self): self.zeros=0; self.steps=0
            def zero_grad(self): self.zeros+=1
            def step(self): self.steps+=1
        def test_evaluator_exact_full_windows():
            o=O(); train_accumulated([1,2,3,4],o,lambda b:None,2); assert (o.zeros,o.steps)==(2,2)
        def test_evaluator_empty_batches_do_nothing():
            o=O(); train_accumulated([],o,lambda b:None,2); assert (o.zeros,o.steps)==(0,0)
        """,
        failing_test="tests/test_behavior.py::test_two_batch_windows",
        expected_failure="optimizer clears and steps every batch instead of per accumulation window",
        tags=("pytorch", "gradient-accumulation", "optimizer", "batch"),
        dependencies=("pytest>=8.2",),
    ),
    Blueprint(
        suite="evaluation", number=113, task_id="device_nested_113",
        title="Move model and nested batch tensors", repository="tensor-runtime-heldout",
        category="device_dtype_mismatch", difficulty="medium",
        package="tensor_runtime", module="device",
        issue="Move the model and every tensor in the batch tuple to the selected device before returning them.",
        buggy="""
        def prepare(model, batch, device):
            return model.to(device), batch
        """,
        fixed="""
        def prepare(model, batch, device):
            return model.to(device), tuple(tensor.to(device) for tensor in batch)
        """,
        visible_tests="""
        from tensor_runtime.device import prepare
        class Movable:
            def __init__(self): self.device=None
            def to(self,d): self.device=d; return self
        def test_model_and_batch_share_device():
            m,x,y=Movable(),Movable(),Movable(); prepared=prepare(m,(x,y),'cuda')
            assert m.device==x.device==y.device=='cuda' and prepared==(m,(x,y))
        def test_batch_tuple_type_preserved():
            m,x=Movable(),Movable(); assert isinstance(prepare(m,(x,),'cpu')[1],tuple)
        """,
        evaluator_tests="""
        from tensor_runtime.device import prepare
        class T:
            def __init__(self): self.calls=[]
            def to(self,d): self.calls.append(d); return self
        def test_evaluator_each_object_moves_once():
            objects=[T(),T(),T()]; prepare(objects[0],tuple(objects[1:]),'mps'); assert [o.calls for o in objects]==[['mps'],['mps'],['mps']]
        def test_evaluator_empty_batch_supported():
            model=T(); assert prepare(model,tuple(),'cpu')==(model,tuple())
        """,
        failing_test="tests/test_behavior.py::test_model_and_batch_share_device",
        expected_failure="model is moved but nested batch tensors remain on the host device",
        tags=("pytorch", "device", "model", "batch"),
        dependencies=("pytest>=8.2",),
    ),
    Blueprint(
        suite="evaluation", number=114, task_id="dtype_labels_114",
        title="Integer class-label dtype", repository="tensor-runtime-heldout",
        category="device_dtype_mismatch", difficulty="easy",
        package="tensor_runtime", module="labels",
        issue="Prepare class-index targets for cross-entropy by converting them to long integer dtype.",
        buggy="""
        def prepare_class_labels(labels):
            return labels.float()
        """,
        fixed="""
        def prepare_class_labels(labels):
            return labels.long()
        """,
        visible_tests="""
        from tensor_runtime.labels import prepare_class_labels
        class Labels:
            def __init__(self): self.dtype='initial'
            def float(self): self.dtype='float'; return self
            def long(self): self.dtype='long'; return self
        def test_labels_are_long_integers():
            labels=Labels(); assert prepare_class_labels(labels) is labels and labels.dtype=='long'
        def test_conversion_happens_once():
            class L(Labels):
                def __init__(self): super().__init__(); self.calls=0
                def long(self): self.calls+=1; return super().long()
            labels=L(); prepare_class_labels(labels); assert labels.calls==1
        """,
        evaluator_tests="""
        from tensor_runtime.labels import prepare_class_labels
        class L:
            def __init__(self): self.used=[]
            def float(self): self.used.append('float'); return self
            def long(self): self.used.append('long'); return self
        def test_evaluator_never_calls_float():
            labels=L(); prepare_class_labels(labels); assert labels.used==['long']
        def test_evaluator_identity_returned():
            labels=L(); assert prepare_class_labels(labels) is labels
        """,
        failing_test="tests/test_behavior.py::test_labels_are_long_integers",
        expected_failure="class-index targets are converted to floating rather than long dtype",
        tags=("pytorch", "dtype", "cross-entropy", "labels"),
        dependencies=("pytest>=8.2",),
    ),
    Blueprint(
        suite="evaluation", number=115, task_id="stable_logsumexp_115",
        title="Overflow-safe log-sum-exp", repository="numerics-heldout",
        category="numerical_stability", difficulty="medium",
        package="numerics", module="logsumexp",
        issue="Implement finite log-sum-exp for large one-dimensional values using a stable shift.",
        buggy="""
        import numpy as np
        def logsumexp(values):
            data = np.asarray(values, dtype=float)
            return float(np.log(np.exp(data).sum()))
        """,
        fixed="""
        import numpy as np
        def logsumexp(values):
            data = np.asarray(values, dtype=float)
            maximum = data.max()
            return float(maximum + np.log(np.exp(data - maximum).sum()))
        """,
        visible_tests="""
        import numpy as np
        from numerics.logsumexp import logsumexp
        def test_large_values_remain_finite():
            assert np.isfinite(logsumexp([1000.0,1001.0]))
        def test_typical_values_match_direct_formula():
            assert logsumexp([0.,1.]) == np.log(1 + np.e)
        """,
        evaluator_tests="""
        import numpy as np
        from numerics.logsumexp import logsumexp
        def test_evaluator_translation_identity():
            np.testing.assert_allclose(logsumexp([10.,11.]), 10 + logsumexp([0.,1.]))
        def test_evaluator_single_value():
            assert logsumexp([-500.]) == -500.0
        """,
        failing_test="tests/test_behavior.py::test_large_values_remain_finite",
        expected_failure="direct exponentiation overflows for large inputs",
        tags=("numpy", "overflow", "logsumexp", "numerical-stability"),
    ),
    Blueprint(
        suite="evaluation", number=116, task_id="constant_minmax_116",
        title="Finite min-max scaling for constants", repository="numerics-heldout",
        category="numerical_stability", difficulty="easy",
        package="numerics", module="scaling",
        issue="Min-max scaling must return finite zeros for a constant input and preserve shape.",
        buggy="""
        import numpy as np
        def minmax_scale(values):
            data = np.asarray(values, dtype=float)
            return (data - data.min()) / (data.max() - data.min())
        """,
        fixed="""
        import numpy as np
        def minmax_scale(values):
            data = np.asarray(values, dtype=float)
            span = data.max() - data.min()
            if span == 0:
                return np.zeros_like(data, dtype=float)
            return (data - data.min()) / span
        """,
        visible_tests="""
        import numpy as np
        from numerics.scaling import minmax_scale
        def test_constant_values_become_finite_zeros():
            np.testing.assert_array_equal(minmax_scale([4,4,4]), [0,0,0])
        def test_nonconstant_range_is_zero_to_one():
            np.testing.assert_allclose(minmax_scale([2,4,6]), [0,.5,1])
        """,
        evaluator_tests="""
        import numpy as np
        from numerics.scaling import minmax_scale
        def test_evaluator_matrix_shape_preserved():
            data=np.full((2,3),9); assert minmax_scale(data).shape==(2,3)
        def test_evaluator_output_is_float():
            assert minmax_scale(np.array([1,1],dtype=int)).dtype.kind=='f'
        """,
        failing_test="tests/test_behavior.py::test_constant_values_become_finite_zeros",
        expected_failure="division by zero makes constant-array scaling nonfinite",
        tags=("numpy", "division-by-zero", "scaling", "constant"),
    ),
    Blueprint(
        suite="evaluation", number=117, task_id="split_seed_117",
        title="Seeded reproducible split indices", repository="reproducibility-heldout",
        category="random_seed_handling", difficulty="easy",
        package="reproducibility_eval", module="split",
        issue="Use the supplied seed when shuffling indices so repeated calls reproduce the same train/test split.",
        buggy="""
        import numpy as np
        def shuffled_indices(size, seed):
            rng = np.random.default_rng()
            return rng.permutation(size)
        """,
        fixed="""
        import numpy as np
        def shuffled_indices(size, seed):
            rng = np.random.default_rng(seed)
            return rng.permutation(size)
        """,
        visible_tests="""
        import numpy as np
        from reproducibility_eval.split import shuffled_indices
        def test_same_seed_same_order():
            np.testing.assert_array_equal(shuffled_indices(20,7),shuffled_indices(20,7))
        def test_result_is_a_permutation():
            assert sorted(shuffled_indices(10,1).tolist())==list(range(10))
        """,
        evaluator_tests="""
        import numpy as np
        from reproducibility_eval.split import shuffled_indices
        def test_evaluator_known_seed():
            np.testing.assert_array_equal(shuffled_indices(5,3),np.random.default_rng(3).permutation(5))
        def test_evaluator_zero_size():
            assert shuffled_indices(0,9).size==0
        """,
        failing_test="tests/test_behavior.py::test_same_seed_same_order",
        expected_failure="supplied seed is ignored when constructing the generator",
        tags=("numpy", "seed", "split", "reproducibility"),
    ),
    Blueprint(
        suite="evaluation", number=118, task_id="kelvin_conversion_118",
        title="Kelvin to Celsius offset", repository="physics-units-heldout",
        category="scientific_unit_conversion", difficulty="easy",
        package="physics_units", module="temperature",
        issue="Convert Kelvin to Celsius using the exact conventional offset and support NumPy arrays.",
        buggy="""
        import numpy as np
        def kelvin_to_celsius(value):
            return np.asarray(value, dtype=float) - 273.0
        """,
        fixed="""
        import numpy as np
        def kelvin_to_celsius(value):
            return np.asarray(value, dtype=float) - 273.15
        """,
        visible_tests="""
        import numpy as np
        from physics_units.temperature import kelvin_to_celsius
        def test_freezing_point():
            assert kelvin_to_celsius(273.15) == 0.0
        def test_array_conversion():
            np.testing.assert_allclose(kelvin_to_celsius([273.15,373.15]),[0,100])
        """,
        evaluator_tests="""
        from physics_units.temperature import kelvin_to_celsius
        def test_evaluator_absolute_zero():
            assert kelvin_to_celsius(0.0) == -273.15
        def test_evaluator_scalar_is_zero_dimensional():
            assert kelvin_to_celsius(300.0).shape == ()
        """,
        failing_test="tests/test_behavior.py::test_freezing_point",
        expected_failure="temperature conversion uses a rounded 273 degree offset",
        tags=("physics", "temperature", "kelvin", "units"),
    ),
    Blueprint(
        suite="evaluation", number=119, task_id="periodic_boundary_119",
        title="Periodic one-dimensional Laplacian", repository="pde-boundaries-heldout",
        category="scientific_boundary_conditions", difficulty="medium",
        package="pde_boundaries", module="laplacian",
        issue="Compute the second difference with periodic boundary conditions, including wrapped neighbors at both ends.",
        buggy="""
        import numpy as np
        def periodic_laplacian(values, dx=1.0):
            data=np.asarray(values,dtype=float)
            result=np.zeros_like(data)
            result[1:-1]=(data[:-2]-2*data[1:-1]+data[2:])/(dx*dx)
            return result
        """,
        fixed="""
        import numpy as np
        def periodic_laplacian(values, dx=1.0):
            data=np.asarray(values,dtype=float)
            return (np.roll(data,1)-2*data+np.roll(data,-1))/(dx*dx)
        """,
        visible_tests="""
        import numpy as np
        from pde_boundaries.laplacian import periodic_laplacian
        def test_edges_use_wrapped_neighbors():
            np.testing.assert_allclose(periodic_laplacian([1,0,0]),[-2,1,1])
        def test_constant_field_has_zero_laplacian():
            np.testing.assert_allclose(periodic_laplacian([4,4,4]),[0,0,0])
        """,
        evaluator_tests="""
        import numpy as np
        from pde_boundaries.laplacian import periodic_laplacian
        def test_evaluator_dx_scaling():
            np.testing.assert_allclose(periodic_laplacian([1,0,0],.5),[-8,4,4])
        def test_evaluator_sum_is_zero_periodic():
            assert abs(periodic_laplacian([1,3,-2,4]).sum()) < 1e-12
        """,
        failing_test="tests/test_behavior.py::test_edges_use_wrapped_neighbors",
        expected_failure="edge points use zero-filled rather than periodic neighbors",
        tags=("pde", "boundary-condition", "periodic", "finite-difference"),
    ),
    Blueprint(
        suite="evaluation", number=120, task_id="json_numpy_120",
        title="Strict NumPy result serialization", repository="results-io-heldout",
        category="result_serialization", difficulty="hard",
        package="results_io", module="json_results",
        issue="Serialize nested NumPy scalars and arrays to standards-compliant JSON, rejecting NaN and infinity.",
        buggy="""
        import json
        def serialize_results(results):
            return json.dumps(results)
        """,
        fixed="""
        import json
        import numpy as np
        def _plain(value):
            if isinstance(value, np.ndarray):
                return [_plain(item) for item in value.tolist()]
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {key: _plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [_plain(item) for item in value]
            return value
        def serialize_results(results):
            return json.dumps(_plain(results), allow_nan=False, sort_keys=True)
        """,
        visible_tests="""
        import json
        import numpy as np
        import pytest
        from results_io.json_results import serialize_results
        def test_numpy_values_are_serialized():
            encoded=serialize_results({'score':np.float64(.5),'labels':np.array([1,2])})
            assert json.loads(encoded)=={'labels':[1,2],'score':.5}
        def test_nonfinite_values_are_rejected():
            with pytest.raises(ValueError): serialize_results({'loss':np.nan})
        """,
        evaluator_tests="""
        import json
        import numpy as np
        import pytest
        from results_io.json_results import serialize_results
        def test_evaluator_nested_numpy_scalars():
            assert json.loads(serialize_results({'x':[np.int64(3)]}))=={'x':[3]}
        def test_evaluator_infinity_rejected():
            with pytest.raises(ValueError): serialize_results(np.array([np.inf]))
        """,
        failing_test="tests/test_behavior.py::test_numpy_values_are_serialized",
        expected_failure="standard json encoder cannot handle NumPy values or enforce finite numbers",
        tags=("numpy", "json", "serialization", "nan"),
    ),
]


def reference_patch(blueprint: Blueprint) -> str:
    return "".join(
        difflib.unified_diff(
            clean(blueprint.buggy).splitlines(keepends=True),
            clean(blueprint.fixed).splitlines(keepends=True),
            fromfile=f"a/{blueprint.source_path}",
            tofile=f"b/{blueprint.source_path}",
        )
    )


def write_blueprint(root: Path, blueprint: Blueprint) -> None:
    task = root / "benchmarks" / blueprint.suite / blueprint.directory_name
    source = task / "repo" / blueprint.source_path
    visible = task / "repo" / "tests" / "test_behavior.py"
    evaluator = task / "tests" / "test_evaluator.py"
    for directory in (source.parent, visible.parent, evaluator.parent):
        directory.mkdir(parents=True, exist_ok=True)
    source.write_text(clean(blueprint.buggy), encoding="utf-8")
    (source.parent / "__init__.py").write_text("", encoding="utf-8")
    visible.write_text(clean(blueprint.visible_tests), encoding="utf-8")
    evaluator.write_text(clean(blueprint.evaluator_tests), encoding="utf-8")
    (task / "issue.md").write_text(blueprint.issue.strip() + "\n", encoding="utf-8")
    patch = reference_patch(blueprint)
    (task / "reference_patch.diff").write_text(patch, encoding="utf-8")
    metadata = {
        "task_id": blueprint.task_id,
        "title": blueprint.title,
        "repository": blueprint.repository,
        "category": blueprint.category,
        "difficulty": blueprint.difficulty,
        "visible_test_command": ["python", "-m", "pytest", "-q", "tests"],
        "evaluator_test_command": [
            "python", "-m", "pytest", "-q", "tests", ".evalforge_evaluator_tests"
        ],
        "expected_failure": blueprint.expected_failure,
        "dependencies": list(blueprint.dependencies),
        "tags": list(blueprint.tags),
        "schema_version": "1.0",
    }
    (task / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    actions = [
        {
            "kind": "tool",
            "plan": "Inspect the implementation named in the issue.",
            "tool_name": "read_file",
            "arguments": {"path": blueprint.source_path},
        },
        {
            "kind": "tool",
            "plan": "Establish verifier evidence before editing.",
            "tool_name": "run_tests",
            "arguments": {},
        },
        {
            "kind": "tool",
            "diagnosis": {
                "failed_test": blueprint.failing_test,
                "evidence": [blueprint.failing_test],
                "mechanism": blueprint.expected_failure,
                "next_change": blueprint.issue,
            },
            "tool_name": "apply_patch",
            "arguments": {"patch": patch},
        },
        {
            "kind": "tool",
            "plan": "Run the visible verifier after the repair.",
            "tool_name": "run_tests",
            "arguments": {},
        },
        {
            "kind": "submit",
            "plan": "Submit only after the visible verifier passes.",
            "lesson": (
                f"For {blueprint.category}, diagnose {blueprint.expected_failure.lower()} "
                "from test evidence and make the smallest behavior-preserving repair."
            ),
        },
    ]
    (task / "scripted_actions.json").write_text(
        json.dumps(actions, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for blueprint in BLUEPRINTS:
        write_blueprint(root, blueprint)
    print(f"generated {len(BLUEPRINTS)} benchmark tasks")


if __name__ == "__main__":
    main()
