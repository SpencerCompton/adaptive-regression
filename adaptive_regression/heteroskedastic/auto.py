"""Whitening and automatic-radius wrapper for residual-balance descent."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._validation import FloatArray, validate_prediction_data, validate_regression_data
from .rbdesc import RBDescRegressor, RBDescResult, RBDescVariant, _validate_variant

_RADIUS_GROWTH = 2.0
_MAX_RADIUS_FITS = 24
_TOLERANCE_MULTIPLIER = 128.0


@dataclass(frozen=True)
class AutoRBDescResult:
    """Automatic estimate and its short radius-search audit trail."""

    coef: FloatArray
    ols_coef: FloatArray
    correction: FloatArray
    core: RBDescResult | None
    radii: FloatArray
    interiors: NDArray[np.bool_]
    residual_rms: float
    condition_number: float
    whitening_error: float
    radius_stop_reason: str

    @property
    def selected_path(self) -> str:
        return "ols" if self.core is None else self.core.selected_path

    @property
    def certified(self) -> bool:
        return self.core is None or self.core.certified


@dataclass(frozen=True)
class _Whitening:
    X: FloatArray
    map: FloatArray
    ols: FloatArray
    residual: FloatArray
    condition_number: float
    error: float


def _whiten(X: FloatArray, y: FloatArray) -> _Whitening:
    n, d = X.shape
    left, singular, right = np.linalg.svd(X, full_matrices=False)
    tolerance = np.finfo(float).eps * max(n, d) * singular[0]
    if singular.size != d or singular[-1] <= tolerance:
        raise ValueError("automatic RB-Desc requires X to have full column rank")
    root_n = math.sqrt(n)
    design = np.ascontiguousarray(root_n * left)
    coefficient_map = np.ascontiguousarray(right.T) * (root_n / singular)[None, :]
    residual = np.ascontiguousarray(y - left @ (left.T @ y))
    ols = coefficient_map @ (design.T @ y / n)
    return _Whitening(
        design,
        coefficient_map,
        ols,
        residual,
        float(singular[0] / singular[-1]),
        float(np.linalg.norm(design.T @ design / n - np.eye(d), ord=2)),
    )


def _tolerance(*scales: float) -> float:
    scale = max((abs(value) for value in scales), default=0.0)
    return _TOLERANCE_MULTIPLIER * np.finfo(float).eps * scale


class AutoRBDescRegressor:
    """General-purpose RB-Desc with empirical whitening and automatic radius."""

    def __init__(
        self,
        *,
        confidence_level: float = 0.95,
        variant: RBDescVariant = "standard",
    ) -> None:
        if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
            raise ValueError("confidence_level must lie between zero and one")
        self.confidence_level = float(confidence_level)
        self.variant = _validate_variant(variant)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "AutoRBDescRegressor":
        X, y = validate_regression_data(X, y, estimator_name="automatic RB-Desc")
        n, d = X.shape
        if n <= d:
            raise ValueError("automatic RB-Desc requires at least d + 1 observations")
        whitening = _whiten(X, y)
        residual_norm = float(np.linalg.norm(whitening.residual))
        residual_rms = residual_norm / math.sqrt(n)
        residual_floor = _tolerance(
            float(np.linalg.norm(y)), float(np.linalg.norm(y - whitening.residual))
        )

        if residual_norm <= residual_floor:
            result = AutoRBDescResult(
                whitening.ols.copy(),
                whitening.ols.copy(),
                np.zeros(d),
                None,
                np.empty(0),
                np.empty(0, dtype=bool),
                residual_rms,
                whitening.condition_number,
                whitening.error,
                "radius_search_not_needed",
            )
        else:
            radius = max(residual_rms, _tolerance(residual_rms))
            radii: list[float] = []
            interiors: list[bool] = []
            for _ in range(_MAX_RADIUS_FITS):
                fitted = RBDescRegressor(
                    radius,
                    confidence_level=self.confidence_level,
                    variant=self.variant,
                ).fit(whitening.X, whitening.residual)
                core = fitted.result_
                base = core.aggressive if self.variant == "aggressive" else core.standard
                assert base is not None
                interior = base.projections == 0
                radii.append(radius)
                interiors.append(interior)
                if core.certified or interior:
                    reason = "exact_residual_certificate" if core.certified else "interior_radius"
                    correction = whitening.map @ core.coef
                    result = AutoRBDescResult(
                        whitening.ols + correction,
                        whitening.ols.copy(),
                        correction,
                        core,
                        np.asarray(radii),
                        np.asarray(interiors),
                        residual_rms,
                        whitening.condition_number,
                        whitening.error,
                        reason,
                    )
                    break
                radius *= _RADIUS_GROWTH
            else:
                raise RuntimeError("automatic radius search did not find an interior or exact fit")

        self.result_ = result
        self.coef_, self.n_features_in_ = result.coef.copy(), d
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before predict")
        return validate_prediction_data(X, n_features=self.n_features_in_) @ self.coef_


__all__ = ["AutoRBDescRegressor", "AutoRBDescResult"]
