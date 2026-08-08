"""Shared array validation and linear-algebra utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def validate_regression_data(
    X: ArrayLike,
    y: ArrayLike,
    *,
    estimator_name: str = "regression",
    require_more_samples_than_features: bool = False,
    require_full_column_rank: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Validate a finite, no-intercept regression problem.

    The returned arrays are contiguous float64 arrays.  Estimators select the
    sample-size and rank conditions required by their own mathematics.
    """

    design = np.asarray(X, dtype=float)
    response = np.asarray(y, dtype=float)
    if design.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if response.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if design.shape[0] != response.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if design.shape[0] == 0 or design.shape[1] == 0:
        raise ValueError("X must have at least one row and one column")
    if require_more_samples_than_features:
        if design.shape[0] <= design.shape[1]:
            raise ValueError(f"{estimator_name} requires n_samples > n_features")
    elif design.shape[0] < design.shape[1]:
        raise ValueError(f"{estimator_name} requires n_samples >= n_features")
    if not np.all(np.isfinite(design)):
        raise ValueError("X must contain only finite values")
    if not np.all(np.isfinite(response)):
        raise ValueError("y must contain only finite values")
    if (
        require_full_column_rank
        and np.linalg.matrix_rank(design) < design.shape[1]
    ):
        raise ValueError("X must have full column rank")
    return (
        np.ascontiguousarray(design, dtype=float),
        np.ascontiguousarray(response, dtype=float),
    )


def validate_prediction_data(
    X: ArrayLike,
    *,
    n_features: int,
) -> FloatArray:
    """Validate a finite design matrix used for prediction."""

    design = np.asarray(X, dtype=float)
    if design.ndim != 2 or design.shape[1] != n_features:
        raise ValueError("X has an incompatible shape")
    if not np.all(np.isfinite(design)):
        raise ValueError("X must contain only finite values")
    return np.ascontiguousarray(design, dtype=float)


def ordinary_least_squares(X: FloatArray, y: FloatArray) -> FloatArray:
    """Return the minimum-norm no-intercept OLS coefficient."""

    return np.asarray(np.linalg.lstsq(X, y, rcond=None)[0], dtype=float)


__all__ = [
    "FloatArray",
    "ordinary_least_squares",
    "validate_prediction_data",
    "validate_regression_data",
]

