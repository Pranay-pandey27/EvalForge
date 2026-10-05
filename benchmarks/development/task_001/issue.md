# Standardize a feature matrix column-wise

`scientific_stats.zscore_features` should standardize every feature column in a
two-dimensional NumPy array. The current implementation raises a broadcasting
error for ordinary rectangular inputs and produces incorrect behavior for
constant-valued features.

Repair the implementation without changing its public API. Each non-constant
feature should have mean approximately zero and standard deviation approximately
one. Constant feature columns should produce finite zeros.

